pub mod trajectory;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trajectory::TrajectoryForecaster;
use uuid::Uuid;

// ---------- configuration constants ----------

/// Maximum distance (normalized coords) to match a detection to an existing track.
const MAX_MATCH_DISTANCE: f64 = 0.12;

/// Weight given to patch similarity vs spatial distance when scoring matches.
/// Final score = (1 - PATCH_WEIGHT) * spatial_score + PATCH_WEIGHT * patch_score
const PATCH_WEIGHT: f64 = 0.35;

/// How many frames an object can be unseen before the track is dropped.
const MAX_FRAMES_MISSING: u32 = 30;

/// Time window (seconds) for cross-camera linking.
const CROSS_CAM_LINK_WINDOW_SECS: i64 = 30;

/// Patch side length (grayscale).
const PATCH_SIZE: usize = 32;

// ---------- intent classification ----------

/// Intent categories for front-camera objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
  ApproachingGate,
  EnteringCompound,
  PassingTraffic,
  StandingAtBoundary,
  Unknown,
}

// ---------- tracked object ----------

/// A single object being tracked across frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedObject {
  pub id: String,
  pub class: String,
  pub positions: Vec<(f64, f64)>,
  pub frames_seen: u32,
  pub last_seen: DateTime<Utc>,
  pub alerted: bool,
  /// 32x32 grayscale center patch for appearance matching.
  #[serde(skip)]
  pub center_patch: Vec<u8>,
  pub camera_id: u32,
  pub intent: Intent,
  /// If this track was linked from another camera, the original track id.
  pub linked_from: Option<String>,
  /// Internal forecaster (not serialized).
  #[serde(skip)]
  forecaster: Option<TrajectoryForecaster>,
}

impl TrackedObject {
  pub fn new(class: &str, x: f64, y: f64, patch: Vec<u8>, camera_id: u32) -> Self {
    let mut forecaster = TrajectoryForecaster::default_forecaster();
    forecaster.push(x, y);
    Self {
      id: Uuid::new_v4().to_string(),
      class: class.to_string(),
      positions: vec![(x, y)],
      frames_seen: 1,
      last_seen: Utc::now(),
      alerted: false,
      center_patch: patch,
      camera_id,
      intent: Intent::Unknown,
      linked_from: None,
      forecaster: Some(forecaster),
    }
  }

  /// Update position, patch, and internal forecaster.
  pub fn update(&mut self, x: f64, y: f64, patch: Vec<u8>) {
    self.positions.push((x, y));
    self.frames_seen += 1;
    self.last_seen = Utc::now();
    self.center_patch = patch;
    if let Some(ref mut f) = self.forecaster {
      f.push(x, y);
    }
  }

  /// Predict where this object will be `steps` frames from now.
  pub fn predict_position(&self, steps: u32) -> Option<(f64, f64)> {
    self.forecaster.as_ref()?.predict_position(steps)
  }

  /// Get the last known position.
  pub fn last_position(&self) -> (f64, f64) {
    self.positions.last().copied().unwrap_or((0.0, 0.0))
  }

  /// Mark as alerted so we never re-alert for the same object.
  pub fn mark_alerted(&mut self) {
    self.alerted = true;
  }

  /// Whether this object is effectively stationary.
  pub fn is_stationary(&self) -> bool {
    self.forecaster.as_ref().map_or(true, |f| f.is_stationary())
  }
}

// ---------- detection input ----------

/// A single detection from the detector, to be fed into the tracker.
#[derive(Debug, Clone)]
pub struct Detection {
  pub class: String,
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
  pub confidence: f32,
  /// 32x32 grayscale center patch.
  pub center_patch: Vec<u8>,
}

// ---------- multi-object tracker ----------

/// Multi-object tracker that maintains tracks across frames and cameras.
pub struct ObjectTracker {
  /// Active tracks keyed by track id.
  tracks: HashMap<String, TrackedObject>,
  /// Recently disappeared tracks for cross-camera linking.
  disappeared: Vec<TrackedObject>,
  /// Current frame counter.
  frame_count: u64,
  /// Camera IDs considered "front" cameras for intent classification.
  front_camera_ids: Vec<u32>,
}

impl ObjectTracker {
  pub fn new(front_camera_ids: Vec<u32>) -> Self {
    Self {
      tracks: HashMap::new(),
      disappeared: Vec::new(),
      frame_count: 0,
      front_camera_ids,
    }
  }

  /// Process a batch of detections for a single camera frame.
  /// Returns IDs of newly created tracks (potential alert candidates).
  pub fn update(&mut self, camera_id: u32, detections: &[Detection]) -> Vec<String> {
    self.frame_count += 1;
    let mut new_track_ids: Vec<String> = Vec::new();

    // Collect current track ids for this camera
    let cam_track_ids: Vec<String> = self
      .tracks
      .iter()
      .filter(|(_, t)| t.camera_id == camera_id)
      .map(|(id, _)| id.clone())
      .collect();

    let mut matched_track_ids: Vec<String> = Vec::new();
    let mut matched_det_indices: Vec<usize> = Vec::new();

    // Build cost matrix: for each detection, find best matching track
    for (det_idx, det) in detections.iter().enumerate() {
      let mut best_score = f64::MAX;
      let mut best_track_id: Option<String> = None;

      for track_id in &cam_track_ids {
        if matched_track_ids.contains(track_id) {
          continue;
        }
        let track = &self.tracks[track_id];

        // Only match same class
        if track.class != det.class {
          continue;
        }

        let score = match_score(track, det);
        if score < best_score && score < MAX_MATCH_DISTANCE {
          best_score = score;
          best_track_id = Some(track_id.clone());
        }
      }

      if let Some(tid) = best_track_id {
        matched_track_ids.push(tid);
        matched_det_indices.push(det_idx);
      }
    }

    // Update matched tracks
    for (tid, &det_idx) in matched_track_ids.iter().zip(matched_det_indices.iter()) {
      let det = &detections[det_idx];
      if let Some(track) = self.tracks.get_mut(tid) {
        track.update(det.x, det.y, det.center_patch.clone());
      }
    }

    // Create new tracks for unmatched detections
    for (det_idx, det) in detections.iter().enumerate() {
      if matched_det_indices.contains(&det_idx) {
        continue;
      }

      // Try cross-camera linking first
      let linked = self.try_cross_camera_link(camera_id, det);
      if let Some(mut linked_track) = linked {
        linked_track.camera_id = camera_id;
        linked_track.update(det.x, det.y, det.center_patch.clone());
        let id = linked_track.id.clone();
        self.tracks.insert(id, linked_track);
        // Cross-linked tracks are not "new" for alerting purposes
        continue;
      }

      let track = TrackedObject::new(&det.class, det.x, det.y, det.center_patch.clone(), camera_id);
      let id = track.id.clone();
      self.tracks.insert(id.clone(), track);
      new_track_ids.push(id);
    }

    // Prune stale tracks
    self.prune_stale(camera_id);

    // Classify intent for front cameras
    if self.front_camera_ids.contains(&camera_id) {
      self.classify_intents(camera_id);
    }

    new_track_ids
  }

  /// Get a reference to a tracked object by id.
  pub fn get_track(&self, id: &str) -> Option<&TrackedObject> {
    self.tracks.get(id)
  }

  /// Get a mutable reference to a tracked object by id.
  pub fn get_track_mut(&mut self, id: &str) -> Option<&mut TrackedObject> {
    self.tracks.get_mut(id)
  }

  /// Return all active tracks for a given camera.
  pub fn tracks_for_camera(&self, camera_id: u32) -> Vec<&TrackedObject> {
    self.tracks.values().filter(|t| t.camera_id == camera_id).collect()
  }

  /// Return all active tracks.
  pub fn all_tracks(&self) -> Vec<&TrackedObject> {
    self.tracks.values().collect()
  }

  /// Total number of active tracks.
  pub fn track_count(&self) -> usize {
    self.tracks.len()
  }

  /// Remove tracks that have not been seen for MAX_FRAMES_MISSING frames.
  fn prune_stale(&mut self, camera_id: u32) {
    let now = Utc::now();
    let stale_ids: Vec<String> = self
      .tracks
      .iter()
      .filter(|(_, t)| {
        t.camera_id == camera_id
          && (now - t.last_seen).num_seconds() > MAX_FRAMES_MISSING as i64
      })
      .map(|(id, _)| id.clone())
      .collect();

    for id in stale_ids {
      if let Some(track) = self.tracks.remove(&id) {
        self.disappeared.push(track);
        if self.disappeared.len() > 200 {
          self.disappeared.remove(0);
        }
      }
    }
  }

  /// Try to link a new detection to a recently disappeared track from another camera.
  fn try_cross_camera_link(&mut self, target_camera_id: u32, det: &Detection) -> Option<TrackedObject> {
    let now = Utc::now();
    let mut best_idx: Option<usize> = None;
    let mut best_sim: f64 = 0.0;

    for (idx, track) in self.disappeared.iter().enumerate() {
      if track.camera_id == target_camera_id {
        continue;
      }
      if track.class != det.class {
        continue;
      }
      let elapsed = (now - track.last_seen).num_seconds();
      if elapsed > CROSS_CAM_LINK_WINDOW_SECS {
        continue;
      }

      let sim = patch_similarity(&track.center_patch, &det.center_patch);
      if sim > best_sim && sim > 0.6 {
        best_sim = sim;
        best_idx = Some(idx);
      }
    }

    if let Some(idx) = best_idx {
      let mut linked = self.disappeared.remove(idx);
      linked.linked_from = Some(linked.id.clone());
      Some(linked)
    } else {
      None
    }
  }

  /// Classify intent for all active tracks on a front camera.
  fn classify_intents(&mut self, camera_id: u32) {
    let track_ids: Vec<String> = self
      .tracks
      .iter()
      .filter(|(_, t)| t.camera_id == camera_id)
      .map(|(id, _)| id.clone())
      .collect();

    for id in track_ids {
      if let Some(track) = self.tracks.get_mut(&id) {
        track.intent = classify_intent(track);
      }
    }
  }
}

// ---------- scoring / matching helpers ----------

/// Combined spatial + appearance score (lower is better, 0 = perfect match).
fn match_score(track: &TrackedObject, det: &Detection) -> f64 {
  let predicted = track
    .predict_position(1)
    .unwrap_or_else(|| track.last_position());
  let dx = predicted.0 - det.x;
  let dy = predicted.1 - det.y;
  let spatial = (dx * dx + dy * dy).sqrt();

  let sim = patch_similarity(&track.center_patch, &det.center_patch);
  let appearance = 1.0 - sim;

  (1.0 - PATCH_WEIGHT) * spatial + PATCH_WEIGHT * appearance
}

/// Compute normalized cross-correlation between two grayscale patches.
/// Returns 0.0..1.0 (1.0 = identical).
fn patch_similarity(a: &[u8], b: &[u8]) -> f64 {
  let expected_len = PATCH_SIZE * PATCH_SIZE;
  if a.len() != expected_len || b.len() != expected_len {
    return 0.0;
  }

  let mean_a = a.iter().map(|&v| v as f64).sum::<f64>() / expected_len as f64;
  let mean_b = b.iter().map(|&v| v as f64).sum::<f64>() / expected_len as f64;

  let mut num = 0.0f64;
  let mut den_a = 0.0f64;
  let mut den_b = 0.0f64;

  for i in 0..expected_len {
    let da = a[i] as f64 - mean_a;
    let db = b[i] as f64 - mean_b;
    num += da * db;
    den_a += da * da;
    den_b += db * db;
  }

  let denom = (den_a * den_b).sqrt();
  if denom < 1e-9 {
    return if den_a < 1e-9 && den_b < 1e-9 { 1.0 } else { 0.0 };
  }

  (num / denom).max(0.0)
}

/// Classify object intent on a front camera based on trajectory.
fn classify_intent(track: &TrackedObject) -> Intent {
  if track.positions.len() < 5 {
    return Intent::Unknown;
  }

  let (first_x, first_y) = track.positions[0];
  let (last_x, last_y) = track.last_position();
  let dx = last_x - first_x;
  let dy = last_y - first_y;
  let displacement = (dx * dx + dy * dy).sqrt();

  // Stationary or barely moving near the boundary
  if track.is_stationary() || displacement < 0.02 {
    if last_y > 0.8 || last_y < 0.2 || last_x > 0.85 || last_x < 0.15 {
      return Intent::StandingAtBoundary;
    }
    return Intent::Unknown;
  }

  // Moving towards camera (y increasing = approaching in typical gate cam)
  if dy > 0.05 && last_y > 0.6 {
    if let Some((_, pred_y)) = track.predict_position(10) {
      if pred_y > 0.9 {
        return Intent::EnteringCompound;
      }
    }
    return Intent::ApproachingGate;
  }

  // Lateral movement with minimal depth change = passing traffic
  if dx.abs() > dy.abs() * 2.0 && displacement > 0.08 {
    return Intent::PassingTraffic;
  }

  if dy > 0.03 {
    return Intent::ApproachingGate;
  }

  Intent::Unknown
}

#[cfg(test)]
mod tests {
  use super::*;

  fn dummy_patch() -> Vec<u8> {
    vec![128u8; PATCH_SIZE * PATCH_SIZE]
  }

  #[test]
  fn test_new_track_creation() {
    let mut tracker = ObjectTracker::new(vec![1]);
    let dets = vec![Detection {
      class: "person".into(),
      x: 0.5,
      y: 0.5,
      width: 0.1,
      height: 0.2,
      confidence: 0.9,
      center_patch: dummy_patch(),
    }];
    let new_ids = tracker.update(1, &dets);
    assert_eq!(new_ids.len(), 1);
    assert_eq!(tracker.track_count(), 1);
  }

  #[test]
  fn test_track_continuation() {
    let mut tracker = ObjectTracker::new(vec![1]);
    let patch = dummy_patch();

    let dets1 = vec![Detection {
      class: "person".into(),
      x: 0.5,
      y: 0.5,
      width: 0.1,
      height: 0.2,
      confidence: 0.9,
      center_patch: patch.clone(),
    }];
    let new1 = tracker.update(1, &dets1);
    assert_eq!(new1.len(), 1);

    let dets2 = vec![Detection {
      class: "person".into(),
      x: 0.52,
      y: 0.51,
      width: 0.1,
      height: 0.2,
      confidence: 0.88,
      center_patch: patch,
    }];
    let new2 = tracker.update(1, &dets2);
    assert_eq!(new2.len(), 0, "should match existing track, not create new");
    assert_eq!(tracker.track_count(), 1);
  }

  #[test]
  fn test_patch_similarity_identical() {
    let a = dummy_patch();
    let sim = patch_similarity(&a, &a);
    assert!((sim - 1.0).abs() < 1e-6);
  }

  #[test]
  fn test_one_alert_per_object() {
    let mut track = TrackedObject::new("person", 0.5, 0.5, dummy_patch(), 1);
    assert!(!track.alerted);
    track.mark_alerted();
    assert!(track.alerted);
  }
}
