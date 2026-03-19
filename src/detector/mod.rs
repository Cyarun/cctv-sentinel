//! YOLO detector — runs ONNX inference via a Python helper subprocess.
//!
//! Since the `ort` crate has version/linking issues on this platform,
//! inference is delegated to a Python script that uses onnxruntime.
//! The detector writes the frame as a temporary raw file, invokes
//! `inference.py`, and parses the JSON detection output.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, warn};

use crate::camera::gridmap::{self, GridMap};
use crate::camera::Frame;

// ─── Configuration ──────────────────────────────────────────────────────────

const DEFAULT_MODEL_PATH: &str = "/opt/cctvanalytics/models/yolo26n_cctv/bestv4.onnx";
const CONFIDENCE_THRESHOLD: f32 = 0.30;
const MODEL_INPUT_SIZE: u32 = 352;

/// The 10 classes the custom YOLO model was trained on.
pub const CLASSES: [&str; 10] = [
  "person", "car", "motorcycle", "auto_rickshaw", "dog",
  "cat", "cow", "bicycle", "truck", "bus",
];

// ─── Detection Structs ─────────────────────────────────────────────────────

/// A single object detection from YOLO inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
  /// Class name (e.g., "person", "car")
  pub class_name: String,
  /// Class index in the CLASSES array
  pub class_id: u32,
  /// Confidence score 0.0..1.0
  pub confidence: f32,
  /// Bounding box center X, normalized 0.0..1.0
  pub x: f32,
  /// Bounding box center Y, normalized 0.0..1.0
  pub y: f32,
  /// Bounding box width, normalized 0.0..1.0
  pub w: f32,
  /// Bounding box height, normalized 0.0..1.0
  pub h: f32,
  /// Whether the detection center falls inside the camera's ROI gridmap
  pub in_roi: bool,
}

/// Raw detection output from the Python inference script (before ROI check).
#[derive(Debug, Deserialize)]
struct RawDetection {
  class: String,
  class_id: u32,
  confidence: f32,
  x: f32,
  y: f32,
  w: f32,
  h: f32,
}

// ─── YOLO Detector ──────────────────────────────────────────────────────────

/// YOLO object detector that delegates ONNX inference to a Python subprocess.
pub struct YoloDetector {
  model_path: PathBuf,
  script_path: PathBuf,
  confidence_threshold: f32,
  gridmaps: HashMap<u16, GridMap>,
  tmp_dir: PathBuf,
}

impl YoloDetector {
  /// Create a new detector with default model path.
  ///
  /// The `script_dir` should point to the directory containing `inference.py`.
  pub fn new(script_dir: &Path) -> Result<Self> {
    Self::with_model(script_dir, Path::new(DEFAULT_MODEL_PATH))
  }

  /// Create a detector with a custom model path.
  pub fn with_model(script_dir: &Path, model_path: &Path) -> Result<Self> {
    let script_path = script_dir.join("inference.py");
    if !script_path.exists() {
      anyhow::bail!(
        "inference.py not found at {}",
        script_path.display()
      );
    }

    let tmp_dir = std::env::temp_dir().join("cctv-sentinel");
    std::fs::create_dir_all(&tmp_dir)
      .context("failed to create temp directory for detector")?;

    Ok(Self {
      model_path: model_path.to_path_buf(),
      script_path,
      confidence_threshold: CONFIDENCE_THRESHOLD,
      gridmaps: gridmap::get_gridmaps(),
      tmp_dir,
    })
  }

  /// Set a custom confidence threshold (default: 0.30).
  pub fn set_confidence_threshold(&mut self, threshold: f32) {
    self.confidence_threshold = threshold;
  }

  /// Run YOLO detection on a single frame.
  ///
  /// Workflow:
  /// 1. Write the raw RGB frame to a temporary file
  /// 2. Invoke inference.py with the file path, model path, and dimensions
  /// 3. Parse JSON detections from stdout
  /// 4. Apply ROI gridmap check to each detection
  /// 5. Filter by confidence threshold
  pub async fn detect(&self, frame: &Frame) -> Result<Vec<Detection>> {
    // Write frame RGB data to a temp file
    let tmp_path = self.tmp_dir.join(format!(
      "frame_{}_{}.raw",
      frame.camera_id, frame.timestamp_ms
    ));
    {
      let mut f = std::fs::File::create(&tmp_path)
        .context("failed to create temp frame file")?;
      f.write_all(&frame.rgb_data)
        .context("failed to write frame data")?;
    }

    debug!(
      camera = frame.camera_id,
      path = %tmp_path.display(),
      "running inference"
    );

    // Invoke the Python inference script
    let output = Command::new("python3")
      .args([
        self.script_path.to_str().unwrap_or("inference.py"),
        tmp_path.to_str().unwrap_or(""),
        self.model_path.to_str().unwrap_or(DEFAULT_MODEL_PATH),
        &frame.width.to_string(),
        &frame.height.to_string(),
        &MODEL_INPUT_SIZE.to_string(),
        &self.confidence_threshold.to_string(),
      ])
      .output()
      .await
      .context("failed to spawn python3 inference")?;

    // Clean up temp file regardless of result
    let _ = std::fs::remove_file(&tmp_path);

    if !output.status.success() {
      let stderr = String::from_utf8_lossy(&output.stderr);
      anyhow::bail!(
        "inference.py failed for camera {}: {}",
        frame.camera_id,
        stderr.trim()
      );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw_detections: Vec<RawDetection> = serde_json::from_str(stdout.trim())
      .context("failed to parse inference output JSON")?;

    // Apply ROI gridmap and build final detections
    let gridmap = self.gridmaps.get(&frame.camera_id);
    let detections: Vec<Detection> = raw_detections
      .into_iter()
      .filter(|d| d.confidence >= self.confidence_threshold)
      .map(|d| {
        let in_roi = gridmap
          .map(|g| g.is_monitored(d.x, d.y))
          .unwrap_or(true); // no gridmap = assume monitored

        Detection {
          class_name: d.class,
          class_id: d.class_id,
          confidence: d.confidence,
          x: d.x,
          y: d.y,
          w: d.w,
          h: d.h,
          in_roi,
        }
      })
      .collect();

    if detections.is_empty() {
      debug!(camera = frame.camera_id, "no detections above threshold");
    } else {
      debug!(
        camera = frame.camera_id,
        count = detections.len(),
        roi_count = detections.iter().filter(|d| d.in_roi).count(),
        "detections found"
      );
    }

    Ok(detections)
  }

  /// Run detection on multiple frames sequentially.
  pub async fn detect_batch(
    &self,
    frames: &[Frame],
  ) -> HashMap<u16, Vec<Detection>> {
    let mut results = HashMap::new();
    for frame in frames {
      match self.detect(frame).await {
        Ok(dets) => {
          results.insert(frame.camera_id, dets);
        }
        Err(e) => {
          warn!(
            camera = frame.camera_id,
            error = %e,
            "detection failed"
          );
        }
      }
    }
    results
  }

  /// Return only detections that are inside the ROI.
  pub fn filter_roi(detections: &[Detection]) -> Vec<&Detection> {
    detections.iter().filter(|d| d.in_roi).collect()
  }

  /// Check if any detection of a given class is in the ROI.
  pub fn has_class_in_roi(detections: &[Detection], class: &str) -> bool {
    detections
      .iter()
      .any(|d| d.in_roi && d.class_name == class)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn classes_count() {
    assert_eq!(CLASSES.len(), 10);
  }

  #[test]
  fn detection_serialization() {
    let det = Detection {
      class_name: "person".to_string(),
      class_id: 0,
      confidence: 0.85,
      x: 0.5,
      y: 0.6,
      w: 0.1,
      h: 0.3,
      in_roi: true,
    };
    let json = serde_json::to_string(&det).unwrap();
    assert!(json.contains("person"));
    assert!(json.contains("0.85"));
  }

  #[test]
  fn filter_roi_works() {
    let dets = vec![
      Detection {
        class_name: "person".into(),
        class_id: 0,
        confidence: 0.9,
        x: 0.5, y: 0.5, w: 0.1, h: 0.2,
        in_roi: true,
      },
      Detection {
        class_name: "car".into(),
        class_id: 1,
        confidence: 0.8,
        x: 0.1, y: 0.1, w: 0.2, h: 0.3,
        in_roi: false,
      },
    ];
    let filtered = YoloDetector::filter_roi(&dets);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].class_name, "person");
  }

  #[test]
  fn has_class_in_roi_check() {
    let dets = vec![
      Detection {
        class_name: "dog".into(),
        class_id: 4,
        confidence: 0.7,
        x: 0.5, y: 0.5, w: 0.1, h: 0.1,
        in_roi: true,
      },
    ];
    assert!(YoloDetector::has_class_in_roi(&dets, "dog"));
    assert!(!YoloDetector::has_class_in_roi(&dets, "person"));
  }
}
