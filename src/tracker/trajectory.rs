use std::collections::VecDeque;

/// Sliding window size for velocity estimation
const VELOCITY_WINDOW: usize = 10;

/// Minimum displacement per frame to consider an object moving
const STATIONARY_THRESHOLD: f64 = 0.005;

/// Trajectory forecaster with EMA smoothing and linear projection.
#[derive(Debug, Clone)]
pub struct TrajectoryForecaster {
  /// EMA smoothing factor (higher = more weight on recent observations)
  alpha: f64,
  /// Raw position history (normalized 0..1 coordinates)
  positions: VecDeque<(f64, f64)>,
  /// EMA-smoothed position history
  smoothed: VecDeque<(f64, f64)>,
  /// Maximum history length
  max_history: usize,
}

impl TrajectoryForecaster {
  pub fn new(alpha: f64, max_history: usize) -> Self {
    Self {
      alpha: alpha.clamp(0.01, 0.99),
      positions: VecDeque::with_capacity(max_history),
      smoothed: VecDeque::with_capacity(max_history),
      max_history,
    }
  }

  /// Default forecaster with alpha=0.7 and 60-frame history.
  pub fn default_forecaster() -> Self {
    Self::new(0.7, 60)
  }

  /// Push a new observed position and update EMA.
  pub fn push(&mut self, x: f64, y: f64) {
    self.positions.push_back((x, y));
    if self.positions.len() > self.max_history {
      self.positions.pop_front();
    }

    let smoothed_pt = match self.smoothed.back() {
      Some(&(sx, sy)) => {
        let nx = self.alpha * x + (1.0 - self.alpha) * sx;
        let ny = self.alpha * y + (1.0 - self.alpha) * sy;
        (nx, ny)
      }
      None => (x, y),
    };

    self.smoothed.push_back(smoothed_pt);
    if self.smoothed.len() > self.max_history {
      self.smoothed.pop_front();
    }
  }

  /// Returns the latest EMA-smoothed position, if available.
  pub fn current_smoothed(&self) -> Option<(f64, f64)> {
    self.smoothed.back().copied()
  }

  /// Estimate median velocity over the sliding window using smoothed positions.
  /// Returns (vx, vy) in normalized units per frame.
  pub fn median_velocity(&self) -> (f64, f64) {
    let n = self.smoothed.len();
    if n < 2 {
      return (0.0, 0.0);
    }

    let window_start = if n > VELOCITY_WINDOW { n - VELOCITY_WINDOW } else { 0 };
    let slice: Vec<(f64, f64)> = self.smoothed.iter().skip(window_start).copied().collect();

    if slice.len() < 2 {
      return (0.0, 0.0);
    }

    let mut vx_samples: Vec<f64> = Vec::with_capacity(slice.len() - 1);
    let mut vy_samples: Vec<f64> = Vec::with_capacity(slice.len() - 1);

    for pair in slice.windows(2) {
      vx_samples.push(pair[1].0 - pair[0].0);
      vy_samples.push(pair[1].1 - pair[0].1);
    }

    (median(&mut vx_samples), median(&mut vy_samples))
  }

  /// Returns true if the object is essentially stationary.
  pub fn is_stationary(&self) -> bool {
    let (vx, vy) = self.median_velocity();
    let speed = (vx * vx + vy * vy).sqrt();
    speed < STATIONARY_THRESHOLD
  }

  /// Predict position `steps_ahead` frames into the future using
  /// linear projection from the current smoothed position + median velocity.
  pub fn predict_position(&self, steps_ahead: u32) -> Option<(f64, f64)> {
    let current = self.current_smoothed()?;
    let (vx, vy) = self.median_velocity();
    let s = steps_ahead as f64;
    Some((current.0 + vx * s, current.1 + vy * s))
  }

  /// Predict positions for the next N frames, returning the full trajectory.
  pub fn predict_trajectory(&self, steps: u32) -> Vec<(f64, f64)> {
    (1..=steps).filter_map(|s| self.predict_position(s)).collect()
  }

  /// Number of observed positions.
  pub fn len(&self) -> usize {
    self.positions.len()
  }

  pub fn is_empty(&self) -> bool {
    self.positions.is_empty()
  }

  /// Linear regression slope on the smoothed x and y positions independently.
  /// Returns (slope_x, slope_y) or None if insufficient data.
  pub fn linear_regression_slope(&self) -> Option<(f64, f64)> {
    let n = self.smoothed.len();
    if n < 3 {
      return None;
    }

    let xs: Vec<f64> = self.smoothed.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = self.smoothed.iter().map(|p| p.1).collect();

    let slope_x = lin_reg_slope(&xs);
    let slope_y = lin_reg_slope(&ys);

    Some((slope_x, slope_y))
  }
}

/// Compute median of a mutable slice (sorts in place).
fn median(v: &mut Vec<f64>) -> f64 {
  if v.is_empty() {
    return 0.0;
  }
  v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let mid = v.len() / 2;
  if v.len() % 2 == 0 {
    (v[mid - 1] + v[mid]) / 2.0
  } else {
    v[mid]
  }
}

/// Simple linear regression slope for a 1-D series indexed by 0..n.
fn lin_reg_slope(vals: &[f64]) -> f64 {
  let n = vals.len() as f64;
  if n < 2.0 {
    return 0.0;
  }
  let sum_x: f64 = (0..vals.len()).map(|i| i as f64).sum();
  let sum_y: f64 = vals.iter().sum();
  let sum_xy: f64 = vals.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
  let sum_x2: f64 = (0..vals.len()).map(|i| (i as f64) * (i as f64)).sum();

  let denom = n * sum_x2 - sum_x * sum_x;
  if denom.abs() < 1e-12 {
    return 0.0;
  }
  (n * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_stationary_object() {
    let mut f = TrajectoryForecaster::default_forecaster();
    for _ in 0..20 {
      f.push(0.5, 0.5);
    }
    assert!(f.is_stationary());
  }

  #[test]
  fn test_moving_object_prediction() {
    let mut f = TrajectoryForecaster::default_forecaster();
    for i in 0..20 {
      f.push(0.1 + 0.01 * i as f64, 0.5);
    }
    assert!(!f.is_stationary());
    let pred = f.predict_position(5).unwrap();
    // should be roughly at x ~ 0.29 + 5*0.01 = 0.34
    assert!(pred.0 > 0.28);
  }

  #[test]
  fn test_median() {
    let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
    assert!((median(&mut v) - 3.0).abs() < 1e-9);
  }
}
