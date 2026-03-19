//! Camera manager — RTSP frame grabbing via ffmpeg subprocess.
//!
//! Manages 11 Hikvision DVR cameras. Each camera has a main stream and a
//! sub-stream (CIF H.264). Frame capture uses ffmpeg to decode one RTSP
//! frame into raw RGB bytes.

pub mod gridmap;

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::task::JoinSet;
use tracing::{debug, error, warn};

// ─── DVR Configuration ─────────────────────────────────────────────────────

const DVR_IP: &str = "192.168.1.15";
const DVR_USER: &str = "admin";
const DVR_PASS: &str = "Sv@123456";
const DVR_PORT: u16 = 554;

/// CIF sub-stream resolution
pub const FRAME_WIDTH: u32 = 352;
pub const FRAME_HEIGHT: u32 = 288;
/// Expected byte count for one raw RGB frame
pub const FRAME_BYTES: usize = (FRAME_WIDTH as usize) * (FRAME_HEIGHT as usize) * 3;

/// ffmpeg RTSP connection timeout in microseconds (5 seconds)
const RTSP_TIMEOUT_US: &str = "5000000";

// ─── Camera Definitions ────────────────────────────────────────────────────

/// All 11 DVR channels (main stream). Sub-stream = main + 1.
const ALL_CHANNELS: [u16; 11] = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101];

/// Front perimeter cameras — processed in parallel for speed.
const FRONT_CAMERAS: [u16; 3] = [101, 201, 301];

/// Camera metadata with human-readable names and zone assignments.
#[derive(Debug, Clone)]
pub struct CameraInfo {
  pub id: u16,
  pub name: &'static str,
  pub zone: &'static str,
}

fn camera_registry() -> Vec<CameraInfo> {
  vec![
    CameraInfo { id: 101,  name: "Cam 1 - Front Gate Left",           zone: "front" },
    CameraInfo { id: 201,  name: "Cam 2 - Front Gate Right",          zone: "front" },
    CameraInfo { id: 301,  name: "Cam 3 - Front Corridor Right",      zone: "front" },
    CameraInfo { id: 401,  name: "Cam 4 - Car Parking",               zone: "front" },
    CameraInfo { id: 501,  name: "Cam 5 - Left Corridor",             zone: "ground_left" },
    CameraInfo { id: 601,  name: "Cam 6 - Back Lift/Scooter",         zone: "back" },
    CameraInfo { id: 701,  name: "Cam 7 - Back Lift Other Side",      zone: "back" },
    CameraInfo { id: 801,  name: "Cam 8 - Right Corridor Back",       zone: "ground_right" },
    CameraInfo { id: 901,  name: "Cam 9 - Right Corridor Front",      zone: "ground_right" },
    CameraInfo { id: 1001, name: "Cam 10 - 2nd Floor Flat Entrance",  zone: "upper_floor" },
    CameraInfo { id: 1101, name: "Cam 11 - 1st Floor Stairs",         zone: "upper_floor" },
  ]
}

// ─── Frame ──────────────────────────────────────────────────────────────────

/// A single captured video frame in raw RGB format.
#[derive(Debug, Clone)]
pub struct Frame {
  pub camera_id: u16,
  pub timestamp_ms: u64,
  pub width: u32,
  pub height: u32,
  pub rgb_data: Vec<u8>,
}

impl Frame {
  /// Current unix timestamp in milliseconds.
  fn now_ms() -> u64 {
    SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_millis() as u64
  }
}

// ─── Camera Manager ─────────────────────────────────────────────────────────

/// Manages RTSP connections to all DVR cameras.
///
/// Frame capture uses ffmpeg as a subprocess to avoid linking against
/// libavcodec. Each call spawns ffmpeg, grabs exactly one frame, and
/// returns the raw RGB bytes.
pub struct CameraManager {
  cameras: Vec<CameraInfo>,
  dvr_ip: String,
  dvr_user: String,
  dvr_pass: String,
  dvr_port: u16,
}

impl CameraManager {
  /// Create a new CameraManager with default DVR credentials.
  pub fn new() -> Self {
    Self {
      cameras: camera_registry(),
      dvr_ip: DVR_IP.to_string(),
      dvr_user: DVR_USER.to_string(),
      dvr_pass: DVR_PASS.to_string(),
      dvr_port: DVR_PORT,
    }
  }

  /// Create with custom DVR connection parameters.
  pub fn with_config(ip: &str, user: &str, pass: &str, port: u16) -> Self {
    Self {
      cameras: camera_registry(),
      dvr_ip: ip.to_string(),
      dvr_user: user.to_string(),
      dvr_pass: pass.to_string(),
      dvr_port: port,
    }
  }

  /// All configured camera channel IDs.
  pub fn channel_ids(&self) -> Vec<u16> {
    self.cameras.iter().map(|c| c.id).collect()
  }

  /// Look up camera info by channel ID.
  pub fn camera_info(&self, channel: u16) -> Option<&CameraInfo> {
    self.cameras.iter().find(|c| c.id == channel)
  }

  /// Build the RTSP URL for the sub-stream of a given main channel.
  /// Sub-stream channel = main channel + 1 (e.g., 101 -> 102).
  fn sub_stream_url(&self, main_channel: u16) -> String {
    let sub_channel = main_channel + 1;
    format!(
      "rtsp://{}:{}@{}:{}/Streaming/Channels/{}",
      self.dvr_user, self.dvr_pass, self.dvr_ip, self.dvr_port, sub_channel
    )
  }

  /// Build the RTSP URL for the main stream of a given channel.
  fn main_stream_url(&self, main_channel: u16) -> String {
    format!(
      "rtsp://{}:{}@{}:{}/Streaming/Channels/{}",
      self.dvr_user, self.dvr_pass, self.dvr_ip, self.dvr_port, main_channel
    )
  }

  /// Grab a single frame from a camera's sub-stream via ffmpeg.
  ///
  /// Spawns `ffmpeg` to connect via RTSP/TCP, decode one video frame,
  /// scale to CIF (352x288), and output raw RGB24 bytes to stdout.
  pub async fn grab_frame(&self, channel: u16) -> Result<Frame> {
    let url = self.sub_stream_url(channel);

    debug!(channel, "grabbing frame via ffmpeg");

    let output = Command::new("ffmpeg")
      .args([
        "-rtsp_transport", "tcp",
        "-stimeout", RTSP_TIMEOUT_US,
        "-i", &url,
        "-frames:v", "1",
        "-vf", &format!("scale={}:{}", FRAME_WIDTH, FRAME_HEIGHT),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "-",
      ])
      .output()
      .await
      .context("failed to spawn ffmpeg")?;

    if !output.status.success() {
      let stderr = String::from_utf8_lossy(&output.stderr);
      anyhow::bail!(
        "ffmpeg exited with status {} for channel {}: {}",
        output.status,
        channel,
        stderr.trim()
      );
    }

    let rgb_data = output.stdout;
    if rgb_data.len() != FRAME_BYTES {
      anyhow::bail!(
        "unexpected frame size for channel {}: got {} bytes, expected {}",
        channel,
        rgb_data.len(),
        FRAME_BYTES
      );
    }

    Ok(Frame {
      camera_id: channel,
      timestamp_ms: Frame::now_ms(),
      width: FRAME_WIDTH,
      height: FRAME_HEIGHT,
      rgb_data,
    })
  }

  /// Grab frames from multiple cameras in parallel using tokio tasks.
  ///
  /// Returns a map of channel_id -> Frame for cameras that succeeded.
  /// Failed cameras are logged as warnings and omitted from the result.
  pub async fn grab_frames_parallel(&self, channels: &[u16]) -> HashMap<u16, Frame> {
    let mut join_set = JoinSet::new();

    for &channel in channels {
      let url = self.sub_stream_url(channel);
      join_set.spawn(async move {
        let result = grab_frame_subprocess(&url, channel).await;
        (channel, result)
      });
    }

    let mut frames = HashMap::new();
    while let Some(result) = join_set.join_next().await {
      match result {
        Ok((channel, Ok(frame))) => {
          frames.insert(channel, frame);
        }
        Ok((channel, Err(e))) => {
          warn!(channel, error = %e, "failed to grab frame");
        }
        Err(e) => {
          error!(error = %e, "task join error during frame grab");
        }
      }
    }
    frames
  }

  /// Grab frames following the cascade order:
  /// 1. Front cameras (101, 201, 301) — in parallel
  /// 2. Remaining cameras — sequentially
  ///
  /// This matches the processing priority: front perimeter first,
  /// then ground floor, then upper floors.
  pub async fn grab_frames_cascade(&self) -> HashMap<u16, Frame> {
    let mut all_frames = HashMap::new();

    // Phase 1: front cameras in parallel
    let front_frames = self.grab_frames_parallel(&FRONT_CAMERAS).await;
    all_frames.extend(front_frames);

    // Phase 2: remaining cameras sequentially
    let remaining: Vec<u16> = ALL_CHANNELS
      .iter()
      .copied()
      .filter(|ch| !FRONT_CAMERAS.contains(ch))
      .collect();

    for channel in remaining {
      match self.grab_frame(channel).await {
        Ok(frame) => {
          all_frames.insert(channel, frame);
        }
        Err(e) => {
          warn!(channel, error = %e, "failed to grab frame (sequential)");
        }
      }
    }

    all_frames
  }
}

impl Default for CameraManager {
  fn default() -> Self {
    Self::new()
  }
}

/// Standalone async function to grab a frame — used by parallel tasks
/// since CameraManager is not Send+Sync across spawn boundaries.
async fn grab_frame_subprocess(rtsp_url: &str, channel: u16) -> Result<Frame> {
  let output = Command::new("ffmpeg")
    .args([
      "-rtsp_transport", "tcp",
      "-stimeout", RTSP_TIMEOUT_US,
      "-i", rtsp_url,
      "-frames:v", "1",
      "-vf", &format!("scale={}:{}", FRAME_WIDTH, FRAME_HEIGHT),
      "-f", "rawvideo",
      "-pix_fmt", "rgb24",
      "-loglevel", "error",
      "-",
    ])
    .output()
    .await
    .context("failed to spawn ffmpeg")?;

  if !output.status.success() {
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!(
      "ffmpeg exited {} for channel {}: {}",
      output.status, channel, stderr.trim()
    );
  }

  let rgb_data = output.stdout;
  if rgb_data.len() != FRAME_BYTES {
    anyhow::bail!(
      "bad frame size for channel {}: {} bytes (expected {})",
      channel, rgb_data.len(), FRAME_BYTES
    );
  }

  Ok(Frame {
    camera_id: channel,
    timestamp_ms: Frame::now_ms(),
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    rgb_data,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn camera_registry_has_11_cameras() {
    let reg = camera_registry();
    assert_eq!(reg.len(), 11);
  }

  #[test]
  fn sub_stream_channel_is_main_plus_one() {
    let mgr = CameraManager::new();
    let url = mgr.sub_stream_url(101);
    assert!(url.contains("/Streaming/Channels/102"));
  }

  #[test]
  fn main_stream_url_uses_main_channel() {
    let mgr = CameraManager::new();
    let url = mgr.main_stream_url(101);
    assert!(url.contains("/Streaming/Channels/101"));
  }

  #[test]
  fn all_channels_count() {
    assert_eq!(ALL_CHANNELS.len(), 11);
  }

  #[test]
  fn front_cameras_are_subset_of_all() {
    for fc in &FRONT_CAMERAS {
      assert!(ALL_CHANNELS.contains(fc));
    }
  }
}
