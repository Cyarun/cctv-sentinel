use reqwest::multipart;
use serde::Deserialize;
use std::path::Path;
use tracing::{error, info, warn};

const TELEGRAM_API_BASE: &str = "https://api.telegram.org/bot";

/// Telegram bot client for sending messages and videos.
#[derive(Debug, Clone)]
pub struct TelegramBot {
  bot_token: String,
  client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct TgResponse {
  ok: bool,
  description: Option<String>,
}

/// Parsed inline command from a Telegram update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BotCommand {
  Status,
  Snap { camera_id: Option<u32> },
  Help,
  Unknown(String),
}

impl TelegramBot {
  pub fn new(bot_token: &str) -> Self {
    let client = reqwest::Client::builder()
      .timeout(std::time::Duration::from_secs(120))
      .build()
      .expect("failed to build reqwest client");

    Self {
      bot_token: bot_token.to_string(),
      client,
    }
  }

  /// Default bot using the project's Telegram token.
  pub fn default_bot() -> Self {
    Self::new("8287023378:AAGNZ4gX8cKdkO4wxeWBCT1hhg25zAR6PA4")
  }

  fn api_url(&self, method: &str) -> String {
    format!("{}{}/{}", TELEGRAM_API_BASE, self.bot_token, method)
  }

  /// Send a text message to the specified chat.
  pub async fn send_message(&self, chat_id: &str, text: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = self.api_url("sendMessage");
    let resp = self
      .client
      .post(&url)
      .json(&serde_json::json!({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
      }))
      .send()
      .await?;

    let status = resp.status();
    let body: TgResponse = resp.json().await?;
    if !body.ok {
      let desc = body.description.unwrap_or_default();
      error!(chat_id, %status, desc, "telegram sendMessage failed");
      return Err(format!("telegram sendMessage failed: {}", desc).into());
    }

    info!(chat_id, "telegram message sent");
    Ok(())
  }

  /// Send a video file to the specified chat with a caption.
  pub async fn send_video(
    &self,
    chat_id: &str,
    video_path: &Path,
    caption: &str,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !video_path.exists() {
      return Err(format!("video file not found: {}", video_path.display()).into());
    }

    let file_bytes = tokio::fs::read(video_path).await?;
    let file_name = video_path
      .file_name()
      .and_then(|n| n.to_str())
      .unwrap_or("clip.mp4")
      .to_string();

    let file_part = multipart::Part::bytes(file_bytes)
      .file_name(file_name)
      .mime_str("video/mp4")?;

    let form = multipart::Form::new()
      .text("chat_id", chat_id.to_string())
      .text("caption", caption.to_string())
      .text("parse_mode", "HTML".to_string())
      .part("video", file_part);

    let url = self.api_url("sendVideo");
    let resp = self.client.post(&url).multipart(form).send().await?;

    let status = resp.status();
    let body: TgResponse = resp.json().await?;
    if !body.ok {
      let desc = body.description.unwrap_or_default();
      error!(chat_id, %status, desc, "telegram sendVideo failed");
      return Err(format!("telegram sendVideo failed: {}", desc).into());
    }

    info!(chat_id, "telegram video sent");
    Ok(())
  }

  /// Send a video to all configured chat IDs.
  pub async fn broadcast_video(
    &self,
    chat_ids: &[String],
    video_path: &Path,
    caption: &str,
  ) -> Vec<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let mut results = Vec::new();
    for cid in chat_ids {
      results.push(self.send_video(cid, video_path, caption).await);
    }
    results
  }

  /// Send a text message to all configured chat IDs.
  pub async fn broadcast_message(
    &self,
    chat_ids: &[String],
    text: &str,
  ) -> Vec<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let mut results = Vec::new();
    for cid in chat_ids {
      results.push(self.send_message(cid, text).await);
    }
    results
  }

  /// Parse a bot command from message text.
  pub fn parse_command(text: &str) -> BotCommand {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    if lower == "/status" {
      BotCommand::Status
    } else if lower == "/help" {
      BotCommand::Help
    } else if lower.starts_with("/snap") {
      let parts: Vec<&str> = trimmed.split_whitespace().collect();
      let camera_id = parts.get(1).and_then(|s| s.parse::<u32>().ok());
      BotCommand::Snap { camera_id }
    } else {
      BotCommand::Unknown(trimmed.to_string())
    }
  }

  /// Generate help text for bot commands.
  pub fn help_text() -> &'static str {
    concat!(
      "<b>CCTV Sentinel Bot</b>\n\n",
      "/status - Show system status (cameras, tracks, uptime)\n",
      "/snap [cam_id] - Take a snapshot from a camera\n",
      "/help - Show this help message\n",
    )
  }

  /// Poll for incoming updates (long-polling). Returns parsed commands with chat_id.
  pub async fn poll_updates(
    &self,
    offset: &mut i64,
  ) -> Result<Vec<(String, BotCommand)>, Box<dyn std::error::Error + Send + Sync>> {
    let url = self.api_url("getUpdates");
    let resp = self
      .client
      .get(&url)
      .query(&[
        ("offset", offset.to_string()),
        ("timeout", "30".to_string()),
      ])
      .send()
      .await?;

    let body: serde_json::Value = resp.json().await?;
    let mut commands = Vec::new();

    if let Some(results) = body.get("result").and_then(|r| r.as_array()) {
      for update in results {
        let update_id = update
          .get("update_id")
          .and_then(|v| v.as_i64())
          .unwrap_or(0);
        if update_id >= *offset {
          *offset = update_id + 1;
        }

        if let Some(msg) = update.get("message") {
          let chat_id = msg
            .get("chat")
            .and_then(|c| c.get("id"))
            .and_then(|v| v.as_i64())
            .map(|id| id.to_string())
            .unwrap_or_default();

          if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
            if text.starts_with('/') {
              let cmd = Self::parse_command(text);
              commands.push((chat_id, cmd));
            }
          }
        }
      }
    }

    Ok(commands)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_parse_status() {
    assert_eq!(TelegramBot::parse_command("/status"), BotCommand::Status);
  }

  #[test]
  fn test_parse_help() {
    assert_eq!(TelegramBot::parse_command("/help"), BotCommand::Help);
  }

  #[test]
  fn test_parse_snap_with_cam() {
    assert_eq!(
      TelegramBot::parse_command("/snap 3"),
      BotCommand::Snap { camera_id: Some(3) }
    );
  }

  #[test]
  fn test_parse_snap_no_cam() {
    assert_eq!(
      TelegramBot::parse_command("/snap"),
      BotCommand::Snap { camera_id: None }
    );
  }

  #[test]
  fn test_parse_unknown() {
    match TelegramBot::parse_command("/foo") {
      BotCommand::Unknown(_) => {}
      other => panic!("expected Unknown, got {:?}", other),
    }
  }
}
