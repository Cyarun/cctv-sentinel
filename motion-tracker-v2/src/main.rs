use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ─── Configuration ───────────────────────────────────────────────────────────

const DVR_IP: &str = "192.168.1.15";
const DVR_USER: &str = "admin";
const DVR_PASS: &str = "Sv@123456";
const FRAME_WIDTH: usize = 352;
const FRAME_HEIGHT: usize = 288;
const FRAME_SIZE: usize = FRAME_WIDTH * FRAME_HEIGHT;
const FPS: f64 = 12.0; // 12fps for smooth tracking on CIF sub-streams
// Shadow-run safety: default to a v2-specific dir so we never overwrite v1 events.
// Override via $MOTION_EVENT_DIR (read once in main()).
const DEFAULT_EVENT_DIR: &str = "/opt/cctvanalytics/motion_events_v2";

// Blob filtering
const MIN_BLOB_PIXELS: usize = 150;
const MAX_BLOB_FRACTION: f64 = 0.55;
const DIFF_THRESHOLD: u8 = 25;
const BG_ALPHA: f64 = 0.02;
const MIN_FRAMES_PERSIST: u8 = 3;
const OSCILLATION_WINDOW: usize = 10;
const OSCILLATION_THRESHOLD: f64 = 3.0;
const COOLDOWN_SECS: u64 = 10;
const FRONT_DAYTIME_COOLDOWN: u64 = 30;

// Trajectory analysis: need 20 frames (10 seconds at 2fps) for direction confidence
const TRAJECTORY_MIN_FRAMES: u8 = 5;    // minimum frames before judging direction
const TRAJECTORY_CONFIDENCE: usize = 15; // frames for high-confidence trajectory

// Stationary detection
const STATIONARY_BLOB_MIN: usize = 600;
const STATIONARY_FRAMES: u8 = 60;  // 5 seconds at 12fps — someone standing still

// Time-based thresholds (in seconds, converted to frames using FPS)
const ROI_MIN_SECS_FRONT_BOTTOM: f64 = 3.0;  // 3 seconds inside bottom ROI to alert
const ROI_MIN_SECS_FRONT_SHOPS: f64 = 7.0;   // 7 seconds inside top ROI (shops) to alert
const ROI_MIN_SECS_INTERIOR: f64 = 1.0;       // 1 second for interior cameras

// Front camera channels
const FRONT_CAMERAS: [u16; 4] = [101, 201, 301, 401];
// Critical cameras — any movement is significant
const CRITICAL_CAMERAS: [u16; 2] = [1001, 1101];

// Sub-stream channels
const CHANNELS: [(u16, u16); 11] = [
    (101, 102), (201, 202), (301, 302), (401, 402),
    (501, 502), (601, 602), (701, 702),
    (801, 802), (901, 902), (1001, 1002), (1101, 1102),
];

// ─── Zone Boundaries (from daytime frame analysis) ───────────────────────────
// Normalized coordinates — where road ends and compound begins

struct CameraZone {
    /// The boundary line. Blobs beyond this are on the road = ignore.
    /// For cam 1,2: y value — below this = our compound
    /// For cam 3: x value — left of this = our compound
    road_boundary: f64,
    /// Axis: true = Y axis (cam 1,2), false = X axis (cam 3)
    is_y_axis: bool,
    /// Direction toward house: true = increasing value, false = decreasing
    toward_house_positive: bool,
    /// Name for logging
    label: &'static str,
}

fn get_camera_zones() -> HashMap<u16, CameraZone> {
    let mut zones = HashMap::new();

    // Cam 1: y > 0.65 = our compound/gate. Road is y 0.35-0.65. Shops y < 0.35.
    zones.insert(101, CameraZone {
        road_boundary: 0.65,
        is_y_axis: true,
        toward_house_positive: true, // moving down (y increases) = toward house
        label: "Cam1-front",
    });

    // Cam 2: y > 0.50 = compound wall/gate. Road is y < 0.50.
    zones.insert(201, CameraZone {
        road_boundary: 0.50,
        is_y_axis: true,
        toward_house_positive: true,
        label: "Cam2-front",
    });

    // Cam 3: x < 0.45 = inside compound. Road is x > 0.55.
    zones.insert(301, CameraZone {
        road_boundary: 0.45,
        is_y_axis: false,
        toward_house_positive: false, // moving left (x decreases) = into compound
        label: "Cam3-compound",
    });

    // Cam 4: parking area — everything is relevant
    zones.insert(401, CameraZone {
        road_boundary: 1.0, // no road boundary
        is_y_axis: true,
        toward_house_positive: true,
        label: "Cam4-parking",
    });

    zones
}

// ─── Intent Classification ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Intent {
    ApproachingGate,      // moving from road toward compound
    EnteringCompound,     // crossed into compound area
    StandingAtBoundary,   // stationary near gate/wall
    LeavingCompound,      // moving from compound toward road
    PersonInCorridor,     // interior camera movement
    PersonAtFlat,         // CRITICAL — family floor
    PersonOnStairs,       // CRITICAL — stairs to family floor
    ParkingActivity,      // movement in parking area
    PassingTraffic,       // lateral movement on road — IGNORE
    RoadActivity,         // movement on road, not approaching — IGNORE
    Unknown,
}

impl Intent {
    fn should_alert(&self) -> bool {
        match self {
            Intent::ApproachingGate => false,    // just watching, don't alert yet
            Intent::EnteringCompound => true,    // confirmed entry — ALERT
            Intent::StandingAtBoundary => true,  // loitering at gate/shops — ALERT
            Intent::PersonInCorridor => true,    // interior movement — ALERT
            Intent::PersonAtFlat => true,        // CRITICAL — ALERT
            Intent::PersonOnStairs => true,      // CRITICAL — ALERT
            Intent::ParkingActivity => true,     // parking area — ALERT
            Intent::LeavingCompound => false,
            Intent::PassingTraffic => false,
            Intent::RoadActivity => false,
            Intent::Unknown => false,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Intent::ApproachingGate => "approaching_gate",
            Intent::EnteringCompound => "entering_compound",
            Intent::StandingAtBoundary => "standing_at_boundary",
            Intent::LeavingCompound => "leaving_compound",
            Intent::PersonInCorridor => "person_in_corridor",
            Intent::PersonAtFlat => "person_at_flat",
            Intent::PersonOnStairs => "person_on_stairs",
            Intent::ParkingActivity => "parking_activity",
            Intent::PassingTraffic => "passing_traffic",
            Intent::RoadActivity => "road_activity",
            Intent::Unknown => "unknown",
        }
    }
}

/// Predict where a blob will be in `steps` frames using linear regression on recent positions.
/// Returns (predicted_x, predicted_y, confidence) where confidence is R² of the fit.
fn predict_trajectory(positions: &[(f64, f64)], steps_ahead: usize) -> Option<(f64, f64, f64)> {
    let n = positions.len();
    if n < 4 { return None; } // need at least 4 points for meaningful prediction

    // Use last 10 positions (5 seconds at 2fps)
    let recent = &positions[n.saturating_sub(10)..];
    let n = recent.len() as f64;

    // Linear regression: y = a + b*t for both x and y coordinates
    let mut sum_t = 0.0;
    let mut sum_t2 = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tx = 0.0;
    let mut sum_ty = 0.0;

    for (i, (x, y)) in recent.iter().enumerate() {
        let t = i as f64;
        sum_t += t;
        sum_t2 += t * t;
        sum_x += x;
        sum_y += y;
        sum_tx += t * x;
        sum_ty += t * y;
    }

    let denom = n * sum_t2 - sum_t * sum_t;
    if denom.abs() < 1e-10 { return None; } // degenerate case

    // Slope and intercept for X
    let bx = (n * sum_tx - sum_t * sum_x) / denom;
    let ax = (sum_x - bx * sum_t) / n;

    // Slope and intercept for Y
    let by = (n * sum_ty - sum_t * sum_y) / denom;
    let ay = (sum_y - by * sum_t) / n;

    // Predict position at t = last + steps_ahead
    let t_pred = (recent.len() - 1) as f64 + steps_ahead as f64;
    let pred_x = ax + bx * t_pred;
    let pred_y = ay + by * t_pred;

    // R² confidence (how well does a line fit the trajectory?)
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut ss_res_x = 0.0;
    let mut ss_tot_x = 0.0;
    let mut ss_res_y = 0.0;
    let mut ss_tot_y = 0.0;
    for (i, (x, y)) in recent.iter().enumerate() {
        let t = i as f64;
        let pred = ax + bx * t;
        ss_res_x += (x - pred) * (x - pred);
        ss_tot_x += (x - mean_x) * (x - mean_x);
        let pred = ay + by * t;
        ss_res_y += (y - pred) * (y - pred);
        ss_tot_y += (y - mean_y) * (y - mean_y);
    }
    let r2_x = if ss_tot_x > 1e-10 { 1.0 - ss_res_x / ss_tot_x } else { 1.0 };
    let r2_y = if ss_tot_y > 1e-10 { 1.0 - ss_res_y / ss_tot_y } else { 1.0 };
    let confidence = (r2_x + r2_y) / 2.0;

    Some((pred_x.clamp(0.0, 1.0), pred_y.clamp(0.0, 1.0), confidence.max(0.0)))
}

// DVR gridMap data — exact 22x18 grids from ISAPI
const GRID_COLS: usize = 22;
const GRID_ROWS: usize = 18;

struct GridMap {
    cells: [[bool; GRID_COLS]; GRID_ROWS],
}

impl GridMap {
    fn from_hex(hex: &str) -> Self {
        let mut cells = [[false; GRID_COLS]; GRID_ROWS];
        for row in 0..GRID_ROWS {
            let row_hex = &hex[row * 6..(row + 1) * 6];
            let bits = u32::from_str_radix(row_hex, 16).unwrap_or(0);
            for col in 0..GRID_COLS {
                cells[row][col] = (bits >> (23 - col)) & 1 == 1;
            }
        }
        GridMap { cells }
    }

    fn is_monitored(&self, x_norm: f64, y_norm: f64) -> bool {
        let col = ((x_norm * GRID_COLS as f64) as usize).min(GRID_COLS - 1);
        let row = ((y_norm * GRID_ROWS as f64) as usize).min(GRID_ROWS - 1);
        self.cells[row][col]
    }

    fn grid_row(&self, y_norm: f64) -> usize {
        ((y_norm * GRID_ROWS as f64) as usize).min(GRID_ROWS - 1)
    }
}

fn get_gridmaps() -> HashMap<u16, GridMap> {
    let mut maps = HashMap::new();
    maps.insert(101, GridMap::from_hex("001ffc001ffc001ffc001ffc001ffc0001f80000080000000000000000000000000000003800007fe0007fff807ffffcfffffcfffffc"));
    maps.insert(201, GridMap::from_hex("0000000000000000000000000000000000000000000000f00003f8001ffc00fffc03fffc1ffffcfffffcfffffcfffffcfffffcfffffc"));
    maps.insert(301, GridMap::from_hex("3f00003f80003fc0003fe0003fe0003ff0003ff8003ffc003ffe007fff007fff807fff807fffc07fffe07ffff07ffff87ffff8fffff8"));
    maps.insert(401, GridMap::from_hex("00000000000000000001ff801ffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc"));
    maps.insert(901, GridMap::from_hex("c00000f80000fe0000ffc0fcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc"));
    // Interior cameras — full frame
    for ch in &[501u16, 601, 701, 801, 1001, 1101] {
        maps.insert(*ch, GridMap::from_hex("fffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc"));
    }
    maps
}

fn classify_intent(camera: u16, positions: &[(f64, f64)], blob_size: usize, gridmaps: &HashMap<u16, GridMap>) -> Intent {
    // Critical cameras — any movement in monitored zone
    if camera == 1001 || camera == 1101 {
        // Even critical cameras: check gridmap
        let grid = match gridmaps.get(&camera) {
            Some(g) => g,
            None => return Intent::PersonAtFlat,
        };
        let latest = positions.last().unwrap_or(&(0.5, 0.5));
        if !grid.is_monitored(latest.0, latest.1) {
            return Intent::RoadActivity; // outside monitored zone
        }
        if camera == 1001 { return Intent::PersonAtFlat; }
        return Intent::PersonOnStairs;
    }

    // Interior cameras — any movement in monitored zone
    match camera {
        501 | 601 | 701 | 801 | 901 => {
            let grid = match gridmaps.get(&camera) {
                Some(g) => g,
                None => return Intent::PersonInCorridor,
            };
            let latest = positions.last().unwrap_or(&(0.5, 0.5));
            if !grid.is_monitored(latest.0, latest.1) {
                return Intent::RoadActivity;
            }
            return Intent::PersonInCorridor;
        }
        401 => {
            let grid = match gridmaps.get(&camera) {
                Some(g) => g,
                None => return Intent::ParkingActivity,
            };
            let latest = positions.last().unwrap_or(&(0.5, 0.5));
            if !grid.is_monitored(latest.0, latest.1) {
                return Intent::RoadActivity;
            }
            return Intent::ParkingActivity;
        }
        _ => {}
    }

    // Front cameras (101, 201, 301) — strict gridMap-based logic
    let grid = match gridmaps.get(&camera) {
        Some(g) => g,
        None => return Intent::Unknown,
    };

    if positions.len() < 2 {
        return Intent::Unknown;
    }

    let latest = positions.last().unwrap();
    let first = &positions[0];
    let n = positions.len();

    // Check if current position is in a monitored cell
    let in_roi_now = grid.is_monitored(latest.0, latest.1);
    if !in_roi_now {
        return Intent::RoadActivity; // not in any monitored zone — ignore completely
    }

    // For Camera 1: determine if in TOP zone (shops, rows 0-4) or BOTTOM zone (gate, rows 12-17)
    if camera == 101 {
        let current_row = grid.grid_row(latest.1);

        // TOP ZONE (rows 0-5): shops area — only alert on loitering >30s
        if current_row <= 5 {
            let is_stationary = n >= (STATIONARY_FRAMES as usize * 3) // 15 seconds
                && positions.iter().skip(n.saturating_sub(20)).all(|p| {
                    let r = grid.grid_row(p.1);
                    r <= 5
                });
            if is_stationary {
                return Intent::StandingAtBoundary; // loitering at shops
            }
            return Intent::RoadActivity; // just activity at shops, don't alert
        }

        // BOTTOM ZONE (rows 12+): gate area
        // Sub-zone: rows 12-14 (y 0.67-0.83) = near compound, just approaching
        // Sub-zone: rows 15-17 (y 0.83-1.0) = AT THE GATE — alert zone
        if current_row >= 15 {
            // AT THE GATE — this is the real alert
            // Check if came from outside (road) → entered compound
            let started_outside = !grid.is_monitored(first.0, first.1)
                || grid.grid_row(first.1) < 12;
            if started_outside {
                return Intent::EnteringCompound; // came from road, now at gate
            }
            // Was already in bottom zone, now at gate
            return Intent::ApproachingGate;
        }

        if current_row >= 12 {
            // Near compound (rows 12-14) — check trajectory
            // Is the object moving downward toward the gate?
            let moving_down = latest.1 > first.1 + 0.03; // significant downward movement
            let has_been_here_long = n >= 16; // 2+ seconds at 8fps

            if moving_down && has_been_here_long {
                return Intent::ApproachingGate;
            }
            // Just entered the near-compound zone briefly — don't alert yet
            return Intent::RoadActivity;
        }

        // Middle zone (rows 6-11) — road, should not be in ROI
        return Intent::PassingTraffic;
    }

    // Camera 2 and 3: simpler — gridMap already defines the compound area
    // Only alert if object has been in the monitored zone for >2 seconds AND moving inward
    let frames_in_roi = positions.iter().rev().take(16) // last 2 seconds
        .filter(|p| grid.is_monitored(p.0, p.1))
        .count();

    if frames_in_roi < 8 {
        // Less than 1 second in the ROI — just touching the edge, ignore
        return Intent::RoadActivity;
    }

    // Check for stationary (standing at gate/boundary)
    let total_movement: f64 = (1..n.min(20)).map(|i| {
        let idx = n - 1 - (n.min(20) - 1 - i);
        let prev_idx = idx.saturating_sub(1);
        let dx = positions[idx].0 - positions[prev_idx].0;
        let dy = positions[idx].1 - positions[prev_idx].1;
        (dx * dx + dy * dy).sqrt()
    }).sum();

    if total_movement < 0.02 && n >= STATIONARY_FRAMES as usize {
        return Intent::StandingAtBoundary;
    }

    // Check if moving deeper into compound (for cam 2: downward, for cam 3: leftward)
    let moving_inward = match camera {
        201 => latest.1 > first.1 + 0.03, // moving down = toward gate
        301 => latest.0 < first.0 - 0.03, // moving left = into compound
        _ => true,
    };

    if moving_inward {
        return Intent::EnteringCompound;
    }

    // In ROI but moving laterally — passing along boundary
    let dx_total = (latest.0 - first.0).abs();
    let dy_total = (latest.1 - first.1).abs();
    if dx_total > dy_total * 1.5 {
        return Intent::PassingTraffic;
    }

    // Default: in ROI, not clearly entering — monitor but don't alert
    Intent::RoadActivity
}

// ─── Baseline Scene Memory ───────────────────────────────────────────────────

const BASELINE_DIR: &str = "/opt/cctvanalytics/baselines";
const BASELINE_SLOTS: [&str; 5] = ["morning", "midday", "afternoon", "evening", "night"];

fn get_time_slot() -> &'static str {
    let hour = {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        ((secs + 19800) % 86400) / 3600 // IST
    };
    match hour {
        7..=9 => "morning",
        10..=13 => "midday",
        14..=17 => "afternoon",
        18..=21 => "evening",
        _ => "night",
    }
}

struct BaselineMemory {
    /// Per-camera, per-timeslot running average frame
    baselines: HashMap<(u16, String), Vec<f64>>,
    /// How many frames contributed to each baseline
    frame_counts: HashMap<(u16, String), u64>,
}

impl BaselineMemory {
    fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            frame_counts: HashMap::new(),
        }
    }

    fn load_from_disk(&mut self, camera: u16) {
        let slot = get_time_slot().to_string();
        let key = (camera, slot.clone());
        if self.baselines.contains_key(&key) {
            return; // already loaded
        }
        let path = format!("{}/cam{}_{}.bin", BASELINE_DIR, camera, slot);
        if let Ok(data) = fs::read(&path) {
            if data.len() == FRAME_SIZE * 8 {
                // f64 = 8 bytes
                let baseline: Vec<f64> = data.chunks(8)
                    .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                self.baselines.insert(key.clone(), baseline);
                self.frame_counts.insert(key, 1000); // pretend we have history
            }
        }
    }

    fn update(&mut self, camera: u16, frame: &[u8], has_motion: bool) {
        let slot = get_time_slot().to_string();
        let key = (camera, slot);

        // Only update baseline from frames WITHOUT motion (quiet scene = baseline)
        if has_motion {
            return;
        }

        let count = self.frame_counts.entry(key.clone()).or_insert(0);
        let baseline = self.baselines.entry(key).or_insert_with(|| vec![128.0; FRAME_SIZE]);

        // Slow update — takes many frames to change baseline
        let alpha = if *count < 100 { 0.05 } else { 0.005 };
        for i in 0..FRAME_SIZE {
            baseline[i] = baseline[i] * (1.0 - alpha) + frame[i] as f64 * alpha;
        }
        *count += 1;
    }

    fn save_to_disk(&self, camera: u16) {
        let slot = get_time_slot().to_string();
        let key = (camera, slot.clone());
        if let Some(baseline) = self.baselines.get(&key) {
            let _ = fs::create_dir_all(BASELINE_DIR);
            let path = format!("{}/cam{}_{}.bin", BASELINE_DIR, camera, slot);
            let data: Vec<u8> = baseline.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let _ = fs::write(path, data);
        }
    }

    fn is_new_object(&self, camera: u16, blob_region: &[(usize, usize)], frame: &[u8]) -> bool {
        let slot = get_time_slot().to_string();
        let key = (camera, slot);
        let baseline = match self.baselines.get(&key) {
            Some(b) => b,
            None => return true, // no baseline = everything is new
        };

        // Check if the blob region differs significantly from baseline
        // If baseline already has something similar here → not new (permanent object)
        let mut diff_count = 0;
        let mut total = 0;
        for &(x, y) in blob_region {
            let idx = y * FRAME_WIDTH + x;
            if idx < FRAME_SIZE {
                total += 1;
                let d = (frame[idx] as f64 - baseline[idx]).abs();
                if d > 30.0 {
                    diff_count += 1;
                }
            }
        }

        if total == 0 { return true; }
        // If >60% of blob pixels differ from baseline → new object
        (diff_count as f64 / total as f64) > 0.6
    }
}

// ─── ROI Polygons ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Polygon {
    points: Vec<(f64, f64)>,
}

impl Polygon {
    fn new(points: Vec<(f64, f64)>) -> Self {
        Self { points }
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        let mut inside = false;
        let n = self.points.len();
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = self.points[i];
            let (xj, yj) = self.points[j];
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

fn get_roi_polygons() -> HashMap<u16, Vec<Polygon>> {
    let mut rois: HashMap<u16, Vec<Polygon>> = HashMap::new();

    // Cam 1: Only compound area (y > 0.65) — excludes ALL road and shops
    rois.insert(101, vec![
        Polygon::new(vec![(0.0, 0.60), (1.0, 0.60), (1.0, 1.0), (0.0, 1.0)]),
    ]);

    // Cam 2: Below road line (y > 0.45) — compound wall, gate, stones
    rois.insert(201, vec![
        Polygon::new(vec![(0.0, 0.45), (1.0, 0.45), (1.0, 1.0), (0.0, 1.0)]),
    ]);

    // Cam 3: Left 45% — inside compound only
    rois.insert(301, vec![
        Polygon::new(vec![(0.0, 0.0), (0.45, 0.0), (0.45, 1.0), (0.0, 1.0)]),
    ]);

    // Cam 4: Bottom 65% — parking
    rois.insert(401, vec![
        Polygon::new(vec![(0.0, 0.35), (1.0, 0.30), (1.0, 1.0), (0.0, 1.0)]),
    ]);

    // Cam 9: Bottom 65% — corridor, exclude canopy
    rois.insert(901, vec![
        Polygon::new(vec![(0.0, 0.35), (1.0, 0.30), (1.0, 1.0), (0.0, 1.0)]),
    ]);

    // Interior cameras: full frame
    for ch in &[501, 601, 701, 801, 1001, 1101] {
        rois.insert(*ch, vec![
            Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        ]);
    }

    rois
}

// ─── Blob Detection ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Blob {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    pixel_count: usize,
    pixels: Vec<(usize, usize)>, // for baseline comparison
}

impl Blob {
    fn centroid_normalized(&self) -> (f64, f64) {
        let cx = (self.min_x + self.max_x) as f64 / 2.0 / FRAME_WIDTH as f64;
        let cy = (self.min_y + self.max_y) as f64 / 2.0 / FRAME_HEIGHT as f64;
        (cx, cy)
    }
}

fn find_blobs(diff: &[u8], width: usize, height: usize) -> Vec<Blob> {
    let mut labels = vec![0u16; width * height];
    let mut blobs: Vec<Blob> = Vec::new();
    let mut label_id: u16 = 0;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if diff[idx] > 0 && labels[idx] == 0 {
                label_id += 1;
                let mut blob = Blob {
                    min_x: x, min_y: y, max_x: x, max_y: y,
                    pixel_count: 0, pixels: Vec::new(),
                };
                let mut stack = vec![(x, y)];
                while let Some((px, py)) = stack.pop() {
                    let pidx = py * width + px;
                    if px >= width || py >= height { continue; }
                    if diff[pidx] == 0 || labels[pidx] != 0 { continue; }
                    labels[pidx] = label_id;
                    blob.pixel_count += 1;
                    blob.min_x = blob.min_x.min(px);
                    blob.min_y = blob.min_y.min(py);
                    blob.max_x = blob.max_x.max(px);
                    blob.max_y = blob.max_y.max(py);
                    // Store some pixels for baseline check (sample every 4th)
                    if blob.pixel_count % 4 == 0 {
                        blob.pixels.push((px, py));
                    }
                    if px > 0 { stack.push((px - 1, py)); }
                    if px + 1 < width { stack.push((px + 1, py)); }
                    if py > 0 { stack.push((px, py - 1)); }
                    if py + 1 < height { stack.push((px, py + 1)); }
                }
                if blob.pixel_count >= MIN_BLOB_PIXELS {
                    blobs.push(blob);
                }
            }
        }
    }
    blobs
}

// ─── Center Patch Matching ───────────────────────────────────────────────────

const PATCH_SIZE: usize = 32;
const PATCH_PIXELS: usize = PATCH_SIZE * PATCH_SIZE; // 1024
const PATCH_MATCH_THRESHOLD: f64 = 0.70;

fn extract_patch(frame: &[u8], cx_norm: f64, cy_norm: f64) -> Vec<u8> {
    let half = PATCH_SIZE / 2;
    let cx = (cx_norm * FRAME_WIDTH as f64) as usize;
    let cy = (cy_norm * FRAME_HEIGHT as f64) as usize;
    let start_x = cx.saturating_sub(half);
    let start_y = cy.saturating_sub(half);
    let mut patch = Vec::with_capacity(PATCH_PIXELS);
    for dy in 0..PATCH_SIZE {
        for dx in 0..PATCH_SIZE {
            let x = (start_x + dx).min(FRAME_WIDTH - 1);
            let y = (start_y + dy).min(FRAME_HEIGHT - 1);
            patch.push(frame[y * FRAME_WIDTH + x]);
        }
    }
    patch
}

fn patch_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let diff_sum: u64 = a.iter().zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    1.0 - (diff_sum as f64 / (a.len() as f64 * 255.0))
}

// ─── Tracked Object ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct TrackedObject {
    id: u32,
    positions: Vec<(f64, f64)>,
    frames_seen: u8,
    frames_missed: u8,
    last_blob_size: usize,
    last_blob_pixels: Vec<(usize, usize)>,
    center_patch: Vec<u8>,     // 32x32 grayscale appearance signature
    velocity: (f64, f64),      // smoothed velocity (vx, vy) per frame
    alerted: bool,             // has this object already triggered an alert?
}

impl TrackedObject {
    fn predict_next(&self) -> (f64, f64) {
        if let Some(last) = self.positions.last() {
            (
                (last.0 + self.velocity.0).clamp(0.0, 1.0),
                (last.1 + self.velocity.1).clamp(0.0, 1.0),
            )
        } else {
            (0.5, 0.5)
        }
    }

    fn update_velocity(&mut self) {
        let n = self.positions.len();
        if n < 2 {
            self.velocity = (0.0, 0.0);
            return;
        }
        // Exponential moving average of velocity from last 5 positions
        let lookback = n.min(5);
        let start = &self.positions[n - lookback];
        let end = self.positions.last().unwrap();
        let vx = (end.0 - start.0) / (lookback - 1).max(1) as f64;
        let vy = (end.1 - start.1) / (lookback - 1).max(1) as f64;
        // Smooth with previous velocity (0.7 new + 0.3 old)
        self.velocity = (
            vx * 0.7 + self.velocity.0 * 0.3,
            vy * 0.7 + self.velocity.1 * 0.3,
        );
    }

    fn is_oscillating(&self) -> bool {
        if self.positions.len() < OSCILLATION_WINDOW {
            return false;
        }
        let recent = &self.positions[self.positions.len() - OSCILLATION_WINDOW..];
        let mut total_dist = 0.0;
        for i in 1..recent.len() {
            let dx = recent[i].0 - recent[i - 1].0;
            let dy = recent[i].1 - recent[i - 1].1;
            total_dist += (dx * dx + dy * dy).sqrt();
        }
        let net_dx = recent.last().unwrap().0 - recent[0].0;
        let net_dy = recent.last().unwrap().1 - recent[0].1;
        let net_dist = (net_dx * net_dx + net_dy * net_dy).sqrt();
        if total_dist < 0.001 { return true; }
        (total_dist / (net_dist + 0.0001)) > OSCILLATION_THRESHOLD
    }

    fn speed(&self) -> f64 {
        (self.velocity.0 * self.velocity.0 + self.velocity.1 * self.velocity.1).sqrt()
    }
}

// ─── Camera Tracker ──────────────────────────────────────────────────────────

struct CameraTracker {
    channel: u16,
    background: Vec<f64>,
    bg_initialized: bool,
    objects: Vec<TrackedObject>,
    next_id: u32,
    last_event_time: Instant,
    roi_polygons: Vec<Polygon>,
    frame_count: u64,
    prev_avg_brightness: f64,
    ir_suppress_frames: u8,
    baseline: BaselineMemory,
    gridmaps: HashMap<u16, GridMap>,
    baseline_save_counter: u64,
}

impl CameraTracker {
    fn new(channel: u16, roi: Vec<Polygon>, gridmaps: HashMap<u16, GridMap>) -> Self {
        let mut baseline = BaselineMemory::new();
        baseline.load_from_disk(channel);
        Self {
            channel,
            background: vec![128.0; FRAME_SIZE],
            bg_initialized: false,
            objects: Vec::new(),
            next_id: 0,
            last_event_time: Instant::now() - Duration::from_secs(60),
            roi_polygons: roi,
            frame_count: 0,
            prev_avg_brightness: 128.0,
            ir_suppress_frames: 0,
            baseline,
            gridmaps,
            baseline_save_counter: 0,
        }
    }

    fn process_frame(&mut self, frame: &[u8]) -> Option<MotionEvent> {
        self.frame_count += 1;

        if !self.bg_initialized {
            for i in 0..FRAME_SIZE {
                self.background[i] = frame[i] as f64;
            }
            self.bg_initialized = true;
            return None;
        }

        if self.frame_count < 10 {
            for i in 0..FRAME_SIZE {
                self.background[i] = self.background[i] * 0.7 + frame[i] as f64 * 0.3;
            }
            return None;
        }

        // IR switch detection
        let avg_brightness: f64 = frame.iter().map(|&p| p as f64).sum::<f64>() / FRAME_SIZE as f64;
        let brightness_delta = (avg_brightness - self.prev_avg_brightness).abs();
        self.prev_avg_brightness = avg_brightness;

        if brightness_delta > 15.0 {
            self.ir_suppress_frames = 10;
            for i in 0..FRAME_SIZE {
                self.background[i] = frame[i] as f64;
            }
            self.objects.clear();
            return None;
        }

        if self.ir_suppress_frames > 0 {
            self.ir_suppress_frames -= 1;
            for i in 0..FRAME_SIZE {
                self.background[i] = self.background[i] * 0.5 + frame[i] as f64 * 0.5;
            }
            return None;
        }

        // Compute diff
        let mut diff = vec![0u8; FRAME_SIZE];
        let mut changed_pixels = 0usize;
        for i in 0..FRAME_SIZE {
            let d = (frame[i] as f64 - self.background[i]).abs() as u8;
            if d > DIFF_THRESHOLD {
                diff[i] = 255;
                changed_pixels += 1;
            }
        }

        // Update background where no motion
        for i in 0..FRAME_SIZE {
            if diff[i] == 0 {
                self.background[i] = self.background[i] * (1.0 - BG_ALPHA) + frame[i] as f64 * BG_ALPHA;
            }
        }

        let changed_fraction = changed_pixels as f64 / FRAME_SIZE as f64;
        let has_motion = changed_pixels >= MIN_BLOB_PIXELS && changed_fraction <= MAX_BLOB_FRACTION;

        // Update baseline scene memory (only from quiet frames)
        self.baseline.update(self.channel, frame, has_motion);
        self.baseline_save_counter += 1;
        if self.baseline_save_counter % 200 == 0 { // save every ~100s at 2fps
            self.baseline.save_to_disk(self.channel);
        }

        if changed_fraction > MAX_BLOB_FRACTION {
            for i in 0..FRAME_SIZE {
                self.background[i] = self.background[i] * 0.8 + frame[i] as f64 * 0.2;
            }
            return None;
        }

        if !has_motion {
            self.age_out_objects();
            return None;
        }

        // Find and filter blobs
        let blobs = find_blobs(&diff, FRAME_WIDTH, FRAME_HEIGHT);
        let valid_blobs: Vec<&Blob> = blobs.iter().filter(|b| {
            if b.pixel_count < MIN_BLOB_PIXELS { return false; }
            let frac = b.pixel_count as f64 / FRAME_SIZE as f64;
            if frac > MAX_BLOB_FRACTION { return false; }
            let (cx, cy) = b.centroid_normalized();
            if self.roi_polygons.is_empty() { return true; }
            self.roi_polygons.iter().any(|p| p.contains(cx, cy))
        }).collect();

        if valid_blobs.is_empty() {
            self.age_out_objects();
            return None;
        }

        // Match blobs to tracked objects using prediction + patch similarity
        // Step 1: Build match candidates (predicted position + patch comparison)
        struct MatchCandidate {
            score: f64,
            obj_idx: usize,
            blob_idx: usize,
            cx: f64,
            cy: f64,
            patch: Vec<u8>,
        }

        let mut candidates: Vec<MatchCandidate> = Vec::new();
        for (oi, obj) in self.objects.iter().enumerate() {
            let (pred_x, pred_y) = obj.predict_next();
            for (bi, blob) in valid_blobs.iter().enumerate() {
                let (cx, cy) = blob.centroid_normalized();
                // Distance to PREDICTED position (not last position)
                let dx = cx - pred_x;
                let dy = cy - pred_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 0.20 { // within 20% of frame from prediction
                    let patch = extract_patch(frame, cx, cy);
                    let sim = patch_similarity(&obj.center_patch, &patch);
                    // Combined score: similarity weighted more than distance
                    let score = sim * 2.0 - dist;
                    candidates.push(MatchCandidate {
                        score, obj_idx: oi, blob_idx: bi, cx, cy, patch,
                    });
                }
            }
        }

        // Step 2: Greedy matching — best scores first
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut matched_objs = vec![false; self.objects.len()];
        let mut matched_blobs = vec![false; valid_blobs.len()];

        for c in &candidates {
            if matched_objs[c.obj_idx] || matched_blobs[c.blob_idx] {
                continue;
            }
            // Accept if patch is similar enough OR prediction is very close
            let dist_to_pred = {
                let (px, py) = self.objects[c.obj_idx].predict_next();
                ((c.cx - px).powi(2) + (c.cy - py).powi(2)).sqrt()
            };
            let sim = patch_similarity(&self.objects[c.obj_idx].center_patch, &c.patch);

            if sim >= PATCH_MATCH_THRESHOLD || dist_to_pred < 0.04 {
                matched_objs[c.obj_idx] = true;
                matched_blobs[c.blob_idx] = true;
                let obj = &mut self.objects[c.obj_idx];
                obj.positions.push((c.cx, c.cy));
                if obj.positions.len() > 240 { obj.positions.remove(0); } // 240 frames = 20s at 12fps
                obj.frames_seen = obj.frames_seen.saturating_add(1);
                obj.frames_missed = 0;
                obj.last_blob_size = valid_blobs[c.blob_idx].pixel_count;
                obj.last_blob_pixels = valid_blobs[c.blob_idx].pixels.clone();
                // Update patch (slow blend: 70% old + 30% new for stability)
                if obj.center_patch.len() == c.patch.len() {
                    for k in 0..obj.center_patch.len() {
                        obj.center_patch[k] = ((obj.center_patch[k] as f64 * 0.7
                            + c.patch[k] as f64 * 0.3) as u8);
                    }
                } else {
                    obj.center_patch = c.patch.clone();
                }
                obj.update_velocity();
            }
        }

        // Age unmatched objects
        for (i, obj) in self.objects.iter_mut().enumerate() {
            if !matched_objs[i] {
                obj.frames_missed += 1;
            }
        }

        // New objects for unmatched blobs
        for (i, blob) in valid_blobs.iter().enumerate() {
            if !matched_blobs[i] {
                let (cx, cy) = blob.centroid_normalized();
                let patch = extract_patch(frame, cx, cy);
                self.next_id += 1;
                self.objects.push(TrackedObject {
                    id: self.next_id,
                    positions: vec![(cx, cy)],
                    frames_seen: 1,
                    frames_missed: 0,
                    last_blob_size: blob.pixel_count,
                    last_blob_pixels: blob.pixels.clone(),
                    center_patch: patch,
                    velocity: (0.0, 0.0),
                    alerted: false,
                });
            }
        }

        self.objects.retain(|o| o.frames_missed < 5);

        // Check cooldown
        let is_front = FRONT_CAMERAS.contains(&self.channel);
        let is_daytime = {
            let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            let hour = ((secs + 19800) % 86400) / 3600;
            hour >= 7 && hour < 23
        };
        let cooldown = if is_front && is_daytime { FRONT_DAYTIME_COOLDOWN } else { COOLDOWN_SECS };
        if self.last_event_time.elapsed() < Duration::from_secs(cooldown) {
            return None;
        }

        // Get gridMap for ROI checking
        let grid = self.gridmaps.get(&self.channel);
        let is_front = FRONT_CAMERAS.contains(&self.channel);

        // Evaluate each tracked object
        for obj in &mut self.objects {
            // Already alerted on this object — ONE alert per object, period
            if obj.alerted { continue; }

            if obj.frames_seen < MIN_FRAMES_PERSIST { continue; }
            if obj.is_oscillating() { continue; }

            // Filter permanent fixtures (>8 seconds, zero speed)
            let speed = obj.speed();
            if obj.frames_seen as f64 > 8.0 * FPS && speed < 0.005 {
                continue;
            }

            // Min blob size for critical cameras
            if (self.channel == 1001 || self.channel == 1101) && obj.last_blob_size < 500 {
                continue;
            }

            let (cx, cy) = obj.positions.last().unwrap();

            // Current position must be inside ROI
            if let Some(g) = grid {
                if !g.is_monitored(*cx, *cy) {
                    continue;
                }
            }

            // ═══════════════════════════════════════════════════════════
            // FRONT CAMERAS: Trajectory-based entry detection
            // Alert ONLY when path shows: outside ROI → inside ROI
            // ═══════════════════════════════════════════════════════════
            if is_front {
                if let Some(g) = grid {
                    // Find if the object has a trajectory FROM outside TO inside
                    let positions = &obj.positions;
                    let n = positions.len();
                    if n < 3 { continue; } // need at least 3 positions

                    // Check: did ANY earlier position start outside the ROI?
                    let started_outside = positions.iter()
                        .take(n / 2) // first half of trajectory
                        .any(|(px, py)| !g.is_monitored(*px, *py));

                    // Check: is the object NOW inside the ROI?
                    let now_inside = g.is_monitored(*cx, *cy);

                    // Check: for Cam 1, is the object in the GATE zone (rows 12+)?
                    let in_gate_zone = if self.channel == 101 {
                        g.grid_row(*cy) >= 12
                    } else {
                        true // cam 2, 3 — all ROI is relevant
                    };

                    if !now_inside || !in_gate_zone {
                        continue; // not in the zone that matters
                    }

                    if started_outside && now_inside {
                        // TRAJECTORY ENTRY: object crossed from outside → inside ROI
                        // This is the real deal — someone entered the compound area
                    } else if !started_outside && now_inside {
                        // Object appeared directly inside ROI (e.g. walked out of gate)
                        // Still relevant but check if it's a permanent object
                        if speed < 0.01 && obj.frames_seen as f64 > 5.0 * FPS {
                            continue; // stationary inside ROI for >5 seconds with no movement = fixture
                        }
                    } else {
                        continue; // not an entry trajectory
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════
            // INTERIOR CAMERAS: Any real movement = alert
            // But filter noise: insects, shadows, light flux
            // ═══════════════════════════════════════════════════════════
            if !is_front {
                // Must be in ROI
                if let Some(g) = grid {
                    if !g.is_monitored(*cx, *cy) { continue; }
                }
                // Must be person-sized (not insect/dust)
                if obj.last_blob_size < 300 { continue; }
                // Must have real movement (not shadow/flux)
                if speed < 0.01 { continue; }
            }

            // Classify intent
            let intent = classify_intent(
                self.channel, &obj.positions, obj.last_blob_size, &self.gridmaps
            );

            if !intent.should_alert() { continue; }

            // Baseline check
            if !obj.last_blob_pixels.is_empty() {
                if !self.baseline.is_new_object(self.channel, &obj.last_blob_pixels, frame) {
                    continue;
                }
            }

            // ═══════════════════════════════════════════════════════════
            // CONFIRMED ALERT — mark object as alerted, never alert again
            // ═══════════════════════════════════════════════════════════
            obj.alerted = true;
            self.last_event_time = Instant::now();

            return Some(MotionEvent {
                camera: self.channel,
                timestamp: unix_timestamp_ms(),
                centroid_x: *cx,
                centroid_y: *cy,
                intent: intent.as_str().to_string(),
                blob_pixels: obj.last_blob_size,
                frames_tracked: obj.frames_seen as u32,
                speed,
            });
        }

        None
    }

    fn age_out_objects(&mut self) {
        for obj in &mut self.objects {
            obj.frames_missed += 1;
        }
        self.objects.retain(|o| o.frames_missed < 5);
    }
}

// ─── Event Output ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
struct MotionEvent {
    camera: u16,
    timestamp: u64,
    centroid_x: f64,
    centroid_y: f64,
    intent: String,       // v2: semantic intent instead of raw direction
    blob_pixels: usize,
    frames_tracked: u32,
    speed: f64,
}

fn event_dir() -> &'static str {
    static DIR: OnceLock<String> = OnceLock::new();
    DIR.get_or_init(|| {
        env::var("MOTION_EVENT_DIR").unwrap_or_else(|_| DEFAULT_EVENT_DIR.to_string())
    })
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

fn write_event(event: &MotionEvent) {
    let dir = Path::new(event_dir());
    let _ = fs::create_dir_all(dir);
    let filename = format!("evt_{}_{}.json", event.camera, event.timestamp);
    let path = dir.join(filename);
    if let Ok(json) = serde_json::to_string(event) {
        let _ = fs::write(&path, json);
        eprintln!("[v2] Event: cam{} intent={} size={} speed={:.4} frames={}",
            event.camera, event.intent, event.blob_pixels, event.speed, event.frames_tracked);
    }
}

// ─── RTSP Stream Reader ─────────────────────────────────────────────────────

fn start_stream_reader(sub_channel: u16) -> Option<std::process::Child> {
    let url = format!(
        "rtsp://{}:{}@{}:554/Streaming/Channels/{}",
        DVR_USER, DVR_PASS, DVR_IP, sub_channel
    );
    Command::new("ffmpeg")
        .args([
            "-rtsp_transport", "tcp",
            "-stimeout", "5000000",
            "-i", &url,
            "-vf", &format!("fps={},format=gray", FPS),
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-s", &format!("{}x{}", FRAME_WIDTH, FRAME_HEIGHT),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}

fn start_playback_reader(main_channel: u16, start_time: &str, end_time: &str) -> Option<std::process::Child> {
    let url = format!(
        "rtsp://{}:{}@{}:554/Streaming/tracks/{}?starttime={}&endtime={}",
        DVR_USER, DVR_PASS, DVR_IP, main_channel, start_time, end_time
    );
    Command::new("ffmpeg")
        .args([
            "-rtsp_transport", "tcp",
            "-stimeout", "10000000",
            "-i", &url,
            "-vf", &format!("fps={},format=gray", FPS),
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-s", &format!("{}x{}", FRAME_WIDTH, FRAME_HEIGHT),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}

// ─── Camera Thread ───────────────────────────────────────────────────────────

fn camera_thread(main_channel: u16, sub_channel: u16, roi: Vec<Polygon>, tx: mpsc::Sender<MotionEvent>) {
    eprintln!("[v2] Starting cam{} (sub={})", main_channel, sub_channel);

    let gridmaps = get_gridmaps();
    let mut tracker = CameraTracker::new(main_channel, roi, gridmaps);

    loop {
        let mut child = match start_stream_reader(sub_channel) {
            Some(c) => c,
            None => {
                eprintln!("[v2] cam{}: stream failed, retry in 5s", main_channel);
                thread::sleep(Duration::from_secs(5));
                continue;
            }
        };

        let mut stdout = match child.stdout.take() {
            Some(s) => s,
            None => { thread::sleep(Duration::from_secs(5)); continue; }
        };

        let mut frame_buf = vec![0u8; FRAME_SIZE];

        loop {
            match stdout.read_exact(&mut frame_buf) {
                Ok(()) => {}
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
            }

            if let Some(event) = tracker.process_frame(&frame_buf) {
                write_event(&event);
                let _ = tx.send(event);
            }
        }

        thread::sleep(Duration::from_secs(3));
    }
}

// ─── Playback Thread ─────────────────────────────────────────────────────────

fn playback_thread(main_channel: u16, start_time: String, end_time: String, roi: Vec<Polygon>, tx: mpsc::Sender<MotionEvent>) {
    eprintln!("[v2-playback] cam{}: {} → {}", main_channel, start_time, end_time);

    let gridmaps = get_gridmaps();
    let mut tracker = CameraTracker::new(main_channel, roi, gridmaps);

    let mut child = match start_playback_reader(main_channel, &start_time, &end_time) {
        Some(c) => c,
        None => { eprintln!("[v2-playback] cam{}: failed", main_channel); return; }
    };

    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => return,
    };

    let mut frame_buf = vec![0u8; FRAME_SIZE];
    let mut frame_num = 0u64;

    loop {
        match stdout.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(_) => { eprintln!("[v2-playback] cam{}: done ({} frames)", main_channel, frame_num); break; }
        }
        frame_num += 1;

        if let Some(event) = tracker.process_frame(&frame_buf) {
            let secs = frame_num as f64 / 4.0;
            eprintln!("[v2-playback] cam{} T+{:.0}s: intent={} size={} speed={:.4}",
                main_channel, secs, event.intent, event.blob_pixels, event.speed);
            write_event(&event);
            let _ = tx.send(event);
        }
    }
    let _ = child.wait();
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    // Playback mode
    if args.len() >= 4 && args[1] == "--playback" {
        let start_time = &args[2];
        let end_time = &args[3];
        let cameras: Vec<u16> = if args.len() >= 5 {
            args[4].split(',').filter_map(|s| s.parse().ok()).collect()
        } else {
            vec![101, 201, 301, 401]
        };

        eprintln!("[v2-playback] {} cameras: {} → {}", cameras.len(), start_time, end_time);
        eprintln!("[v2-playback] events → {}", event_dir());
        let _ = fs::create_dir_all(event_dir());
        let rois = get_roi_polygons();
        let (tx, rx) = mpsc::channel::<MotionEvent>();

        let mut handles = Vec::new();
        for main_ch in &cameras {
            let roi = rois.get(main_ch).cloned().unwrap_or_else(|| {
                vec![Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
            });
            let tx_clone = tx.clone();
            let ch = *main_ch;
            let st = start_time.clone();
            let et = end_time.clone();
            handles.push(thread::spawn(move || {
                playback_thread(ch, st, et, roi, tx_clone);
            }));
        }

        drop(tx);
        let mut events: Vec<MotionEvent> = rx.into_iter().collect();
        events.sort_by_key(|e| e.timestamp);

        eprintln!("\n=== V2 PLAYBACK RESULTS ===");
        eprintln!("Total events: {}", events.len());

        // Count by intent
        let mut intent_counts: HashMap<String, u32> = HashMap::new();
        let mut cam_counts: HashMap<u16, u32> = HashMap::new();
        for e in &events {
            *intent_counts.entry(e.intent.clone()).or_insert(0) += 1;
            *cam_counts.entry(e.camera).or_insert(0) += 1;
        }

        eprintln!("\nBy camera:");
        for (cam, count) in &cam_counts {
            eprintln!("  Cam {}: {}", cam / 100, count);
        }
        eprintln!("\nBy intent:");
        for (intent, count) in &intent_counts {
            eprintln!("  {}: {}", intent, count);
        }
        eprintln!("\nAlertable vs Ignored:");
        let alertable: u32 = events.iter()
            .filter(|e| {
                let intent = match e.intent.as_str() {
                    "approaching_gate" => Intent::ApproachingGate,
                    "entering_compound" => Intent::EnteringCompound,
                    "standing_at_boundary" => Intent::StandingAtBoundary,
                    "person_in_corridor" => Intent::PersonInCorridor,
                    "person_at_flat" => Intent::PersonAtFlat,
                    "person_on_stairs" => Intent::PersonOnStairs,
                    "parking_activity" => Intent::ParkingActivity,
                    _ => Intent::PassingTraffic,
                };
                intent.should_alert()
            })
            .count() as u32;
        eprintln!("  Alertable: {}", alertable);
        eprintln!("  Ignored: {}", events.len() as u32 - alertable);

        for h in handles { let _ = h.join(); }
        return;
    }

    // Live mode
    eprintln!("[v2] Motion tracker v2 starting — intent classification + baseline memory");
    eprintln!("[v2] Events: {}", event_dir());

    let _ = fs::create_dir_all(event_dir());
    let _ = fs::create_dir_all(BASELINE_DIR);
    let rois = get_roi_polygons();
    let (tx, rx) = mpsc::channel::<MotionEvent>();

    let mut handles = Vec::new();
    for (main_ch, sub_ch) in CHANNELS.iter() {
        let roi = rois.get(main_ch).cloned().unwrap_or_else(|| {
            vec![Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
        });
        let tx_clone = tx.clone();
        let main_ch = *main_ch;
        let sub_ch = *sub_ch;
        handles.push(thread::spawn(move || {
            camera_thread(main_ch, sub_ch, roi, tx_clone);
        }));
    }

    drop(tx);
    let mut event_count = 0u64;
    for _event in rx {
        event_count += 1;
        if event_count % 10 == 0 {
            eprintln!("[v2] Total events: {}", event_count);
        }
    }

    for h in handles { let _ = h.join(); }
}
