pub mod telegram;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use telegram::TelegramBot;

// ---------- constants ----------

const TELEGRAM_BOT_TOKEN: &str = "8287023378:AAGNZ4gX8cKdkO4wxeWBCT1hhg25zAR6PA4";
const CHAT_IDS: &[&str] = &["6052029324", "6258808484"];

/// Cooldown in seconds: front cameras get 30s, interior gets 15s.
const FRONT_CAM_COOLDOWN_SECS: i64 = 30;
const INTERIOR_CAM_COOLDOWN_SECS: i64 = 15;

/// Clip duration: 5s before + 7s after = 12s total.
const CLIP_PRE_SECS: u32 = 5;
const CLIP_POST_SECS: u32 = 7;
const CLIP_TOTAL_SECS: u32 = CLIP_PRE_SECS + CLIP_POST_SECS;

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_MODEL: &str = "google/gemini-2.0-flash-001";

// ---------- alert event ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
  pub id: String,
  pub camera_id: u32,
  pub camera_name: String,
  pub track_id: String,
  pub object_class: String,
  pub timestamp: DateTime<Utc>,
  pub clip_path: Option<String>,
  pub ai_description: Option<String>,
  pub intent: Option<String>,
}

// ---------- alert manager ----------

/// Manages alert lifecycle: cooldowns, clip capture, Telegram delivery, AI analysis, DB storage.
pub struct AlertManager {
  bot: TelegramBot,
  chat_ids: Vec<String>,
  /// DVR connection info for RTSP clip grabbing.
  dvr_ip: String,
  dvr_user: String,
  dvr_pass: String,
  /// Directory to store captured clips.
  clips_dir: PathBuf,
  /// Last alert timestamp per camera for cooldown enforcement.
  last_alert: HashMap<u32, DateTime<Utc>>,
  /// Camera IDs considered "front" cameras (30s cooldown).
  front_camera_ids: Vec<u32>,
  /// SQLite connection for persisting alert events.
  db: Arc<Mutex<Connection>>,
  /// OpenRouter API key (read from OPENROUTER_API_KEY env var).
  openrouter_key: Option<String>,
  /// HTTP client reused across requests.
  client: reqwest::Client,
}

impl AlertManager {
  /// Create a new AlertManager. Initializes SQLite schema.
  pub fn new(
    dvr_ip: &str,
    dvr_user: &str,
    dvr_pass: &str,
    clips_dir: &Path,
    front_camera_ids: Vec<u32>,
    db_path: &str,
  ) -> Result<Self, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(clips_dir)?;

    let conn = Connection::open(db_path)?;
    conn.execute_batch(
      "CREATE TABLE IF NOT EXISTS alert_events (
        id TEXT PRIMARY KEY,
        camera_id INTEGER NOT NULL,
        camera_name TEXT NOT NULL,
        track_id TEXT NOT NULL,
        object_class TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        clip_path TEXT,
        ai_description TEXT,
        intent TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_alert_ts ON alert_events(timestamp);
      CREATE INDEX IF NOT EXISTS idx_alert_cam ON alert_events(camera_id);",
    )?;

    let openrouter_key = std::env::var("OPENROUTER_API_KEY").ok();
    if openrouter_key.is_none() {
      warn!("OPENROUTER_API_KEY not set; AI analysis will be skipped");
    }

    Ok(Self {
      bot: TelegramBot::new(TELEGRAM_BOT_TOKEN),
      chat_ids: CHAT_IDS.iter().map(|s| s.to_string()).collect(),
      dvr_ip: dvr_ip.to_string(),
      dvr_user: dvr_user.to_string(),
      dvr_pass: dvr_pass.to_string(),
      clips_dir: clips_dir.to_path_buf(),
      last_alert: HashMap::new(),
      front_camera_ids,
      db: Arc::new(Mutex::new(conn)),
      openrouter_key,
      client: reqwest::Client::new(),
    })
  }

  /// Check whether an alert is allowed for this camera (cooldown check).
  pub fn is_cooldown_active(&self, camera_id: u32) -> bool {
    let cooldown = if self.front_camera_ids.contains(&camera_id) {
      FRONT_CAM_COOLDOWN_SECS
    } else {
      INTERIOR_CAM_COOLDOWN_SECS
    };

    if let Some(last) = self.last_alert.get(&camera_id) {
      let elapsed = (Utc::now() - *last).num_seconds();
      elapsed < cooldown
    } else {
      false
    }
  }

  /// Full alert pipeline: capture clip, send Telegram, run AI analysis, store in DB.
  /// Returns the AlertEvent on success.
  pub async fn trigger_alert(
    &mut self,
    camera_id: u32,
    camera_name: &str,
    channel: u32,
    track_id: &str,
    object_class: &str,
    intent: Option<&str>,
  ) -> Result<AlertEvent, Box<dyn std::error::Error + Send + Sync>> {
    // Enforce cooldown
    if self.is_cooldown_active(camera_id) {
      return Err(format!("cooldown active for camera {}", camera_id).into());
    }

    let now = Utc::now();
    let event_id = uuid::Uuid::new_v4().to_string();
    let ts_str = now.format("%Y%m%d_%H%M%S").to_string();

    // 1. Capture 12-second clip from DVR main stream via RTSP
    let clip_filename = format!("alert_{}_{}.mp4", camera_id, ts_str);
    let clip_path = self.clips_dir.join(&clip_filename);
    let clip_result = self.capture_clip(channel, &clip_path).await;

    let clip_path_str = match clip_result {
      Ok(()) => {
        info!(camera_id, clip = %clip_path.display(), "clip captured");
        Some(clip_path.to_string_lossy().to_string())
      }
      Err(e) => {
        error!(camera_id, error = %e, "clip capture failed, sending text-only alert");
        None
      }
    };

    // 2. Build caption
    let caption = format!(
      "<b>Alert: {} detected</b>\n\
       Camera: {} (cam {})\n\
       Time: {}\n\
       Track: {}\n\
       {}",
      object_class,
      camera_name,
      camera_id,
      now.format("%Y-%m-%d %H:%M:%S UTC"),
      &track_id[..8.min(track_id.len())],
      intent.map_or(String::new(), |i| format!("Intent: {}", i)),
    );

    // 3. Send to Telegram
    if let Some(ref cp) = clip_path_str {
      let video_path = Path::new(cp);
      let results = self.bot.broadcast_video(&self.chat_ids, video_path, &caption).await;
      for (i, r) in results.iter().enumerate() {
        if let Err(e) = r {
          error!(chat_id = %self.chat_ids[i], error = %e, "failed to send video");
        }
      }
    } else {
      let results = self.bot.broadcast_message(&self.chat_ids, &caption).await;
      for (i, r) in results.iter().enumerate() {
        if let Err(e) = r {
          error!(chat_id = %self.chat_ids[i], error = %e, "failed to send message");
        }
      }
    }

    // 4. AI analysis via OpenRouter (non-blocking on failure)
    let ai_desc = if let Some(ref cp) = clip_path_str {
      match self.ai_analyze_clip(cp, camera_name, object_class).await {
        Ok(desc) => {
          // Send the AI analysis as a follow-up message
          let ai_msg = format!(
            "<b>AI Analysis ({})</b>\n{}",
            camera_name, desc
          );
          let _ = self.bot.broadcast_message(&self.chat_ids, &ai_msg).await;
          Some(desc)
        }
        Err(e) => {
          warn!(error = %e, "AI analysis failed");
          None
        }
      }
    } else {
      None
    };

    // 5. Build and store event
    let event = AlertEvent {
      id: event_id,
      camera_id,
      camera_name: camera_name.to_string(),
      track_id: track_id.to_string(),
      object_class: object_class.to_string(),
      timestamp: now,
      clip_path: clip_path_str,
      ai_description: ai_desc,
      intent: intent.map(|s| s.to_string()),
    };

    self.store_event(&event).await?;

    // 6. Update cooldown
    self.last_alert.insert(camera_id, now);

    info!(camera_id, track_id, "alert pipeline complete");
    Ok(event)
  }

  /// Capture a 12-second RTSP clip from the DVR using ffmpeg.
  async fn capture_clip(
    &self,
    channel: u32,
    output_path: &Path,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let rtsp_url = format!(
      "rtsp://{}:{}@{}/Streaming/Channels/{}01",
      self.dvr_user, self.dvr_pass, self.dvr_ip, channel
    );

    let output_str = output_path.to_string_lossy().to_string();

    let result = Command::new("ffmpeg")
      .args([
        "-y",
        "-rtsp_transport", "tcp",
        "-i", &rtsp_url,
        "-t", &CLIP_TOTAL_SECS.to_string(),
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        &output_str,
      ])
      .output()
      .await?;

    if !result.status.success() {
      let stderr = String::from_utf8_lossy(&result.stderr);
      return Err(format!("ffmpeg failed: {}", stderr).into());
    }

    Ok(())
  }

  /// Send a clip to Gemini 2.0 Flash via OpenRouter for scene description.
  async fn ai_analyze_clip(
    &self,
    clip_path: &str,
    camera_name: &str,
    object_class: &str,
  ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let api_key = self
      .openrouter_key
      .as_ref()
      .ok_or("OPENROUTER_API_KEY not set")?;

    // Read clip and base64 encode for the vision API
    let clip_bytes = tokio::fs::read(clip_path).await?;
    let b64_clip = base64_encode(&clip_bytes);

    let prompt = format!(
      "You are a CCTV security analyst. Analyze this surveillance clip from camera '{}'. \
       A '{}' was detected. Describe:\n\
       1. What is happening in the scene\n\
       2. Number of people/vehicles visible\n\
       3. Any suspicious or noteworthy behavior\n\
       4. Movement direction and intent\n\
       Be concise (3-5 sentences).",
      camera_name, object_class
    );

    let payload = serde_json::json!({
      "model": OPENROUTER_MODEL,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": format!("data:video/mp4;base64,{}", b64_clip)
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    });

    let resp = self
      .client
      .post(OPENROUTER_API_URL)
      .header("Authorization", format!("Bearer {}", api_key))
      .header("Content-Type", "application/json")
      .json(&payload)
      .send()
      .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
      let err_msg = body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("unknown error");
      return Err(format!("OpenRouter API error ({}): {}", status, err_msg).into());
    }

    let description = body
      .get("choices")
      .and_then(|c| c.get(0))
      .and_then(|c| c.get("message"))
      .and_then(|m| m.get("content"))
      .and_then(|c| c.as_str())
      .unwrap_or("No analysis available")
      .to_string();

    Ok(description)
  }

  /// Persist an alert event to SQLite.
  async fn store_event(
    &self,
    event: &AlertEvent,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db = self.db.lock().await;
    db.execute(
      "INSERT INTO alert_events (id, camera_id, camera_name, track_id, object_class, timestamp, clip_path, ai_description, intent)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
      params![
        event.id,
        event.camera_id,
        event.camera_name,
        event.track_id,
        event.object_class,
        event.timestamp.to_rfc3339(),
        event.clip_path,
        event.ai_description,
        event.intent,
      ],
    )?;
    Ok(())
  }

  /// Query recent alert events from the database.
  pub async fn recent_events(
    &self,
    limit: usize,
  ) -> Result<Vec<AlertEvent>, Box<dyn std::error::Error + Send + Sync>> {
    let db = self.db.lock().await;
    let mut stmt = db.prepare(
      "SELECT id, camera_id, camera_name, track_id, object_class, timestamp, clip_path, ai_description, intent
       FROM alert_events ORDER BY timestamp DESC LIMIT ?1",
    )?;

    let events = stmt
      .query_map(params![limit as i64], |row| {
        let ts_str: String = row.get(5)?;
        let timestamp = DateTime::parse_from_rfc3339(&ts_str)
          .map(|dt| dt.with_timezone(&Utc))
          .unwrap_or_else(|_| Utc::now());

        Ok(AlertEvent {
          id: row.get(0)?,
          camera_id: row.get(1)?,
          camera_name: row.get(2)?,
          track_id: row.get(3)?,
          object_class: row.get(4)?,
          timestamp,
          clip_path: row.get(6)?,
          ai_description: row.get(7)?,
          intent: row.get(8)?,
        })
      })?
      .collect::<Result<Vec<_>, _>>()?;

    Ok(events)
  }

  /// Get a reference to the Telegram bot for command handling.
  pub fn bot(&self) -> &TelegramBot {
    &self.bot
  }

  /// Get the configured chat IDs.
  pub fn chat_ids(&self) -> &[String] {
    &self.chat_ids
  }
}

/// Simple base64 encoder (avoids pulling in the base64 crate for one call).
fn base64_encode(data: &[u8]) -> String {
  const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let mut result = String::with_capacity((data.len() + 2) / 3 * 4);

  for chunk in data.chunks(3) {
    let b0 = chunk[0] as u32;
    let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
    let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
    let triple = (b0 << 16) | (b1 << 8) | b2;

    result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
    result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

    if chunk.len() > 1 {
      result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
    } else {
      result.push('=');
    }

    if chunk.len() > 2 {
      result.push(CHARS[(triple & 0x3F) as usize] as char);
    } else {
      result.push('=');
    }
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_base64_encode() {
    assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
    assert_eq!(base64_encode(b"Hi"), "SGk=");
    assert_eq!(base64_encode(b"ABC"), "QUJD");
  }

  #[test]
  fn test_cooldown_initially_inactive() {
    let mgr = AlertManager::new(
      "192.168.1.4",
      "admin",
      "pass",
      Path::new("/tmp/sentinel_clips_test"),
      vec![1, 2],
      ":memory:",
    )
    .unwrap();
    assert!(!mgr.is_cooldown_active(1));
    assert!(!mgr.is_cooldown_active(5));
  }
}
