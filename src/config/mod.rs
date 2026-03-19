use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
  pub dvr: DvrConfig,
  pub cameras: Vec<CameraConfig>,
  pub server: ServerConfig,
  pub telegram: TelegramConfig,
  pub detection: DetectionConfig,
  pub auth: AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DvrConfig {
  pub ip: String,
  pub username: String,
  pub password: String,
  pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
  pub id: u32,
  pub channel: u32,
  pub name: String,
  pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
  pub host: String,
  pub port: u16,
  pub static_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
  pub enabled: bool,
  pub bot_token: String,
  pub chat_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
  pub model_path: String,
  pub confidence_threshold: f32,
  pub nms_threshold: f32,
  pub capture_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
  pub jwt_secret: String,
  pub token_expiry_hours: u64,
}

impl Default for AppConfig {
  fn default() -> Self {
    let cameras = vec![
      CameraConfig { id: 1,  channel: 101,  name: "Front Gate".into(),       enabled: true },
      CameraConfig { id: 2,  channel: 201,  name: "Parking Lot".into(),      enabled: true },
      CameraConfig { id: 3,  channel: 301,  name: "Main Entrance".into(),    enabled: true },
      CameraConfig { id: 4,  channel: 401,  name: "Lobby".into(),            enabled: true },
      CameraConfig { id: 5,  channel: 501,  name: "Corridor East".into(),    enabled: true },
      CameraConfig { id: 6,  channel: 601,  name: "Corridor West".into(),    enabled: true },
      CameraConfig { id: 7,  channel: 701,  name: "Server Room".into(),      enabled: true },
      CameraConfig { id: 8,  channel: 801,  name: "Warehouse".into(),        enabled: true },
      CameraConfig { id: 9,  channel: 901,  name: "Loading Dock".into(),     enabled: true },
      CameraConfig { id: 10, channel: 1001, name: "Perimeter North".into(),  enabled: true },
      CameraConfig { id: 11, channel: 1101, name: "Perimeter South".into(),  enabled: true },
    ];

    Self {
      dvr: DvrConfig {
        ip: "192.168.1.15".into(),
        username: "admin".into(),
        password: "Sv@123456".into(),
        protocol: "http".into(),
      },
      cameras,
      server: ServerConfig {
        host: "0.0.0.0".into(),
        port: 8080,
        static_dir: "./static".into(),
      },
      telegram: TelegramConfig {
        enabled: false,
        bot_token: String::new(),
        chat_ids: Vec::new(),
      },
      detection: DetectionConfig {
        model_path: "./models/yolov8n.onnx".into(),
        confidence_threshold: 0.45,
        nms_threshold: 0.5,
        capture_interval_secs: 5,
      },
      auth: AuthConfig {
        jwt_secret: uuid::Uuid::new_v4().to_string(),
        token_expiry_hours: 24,
      },
    }
  }
}

impl AppConfig {
  pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
      let content = std::fs::read_to_string(path)?;
      let config: AppConfig = toml::from_str(&content)?;
      Ok(config)
    } else {
      let config = AppConfig::default();
      config.save(path)?;
      Ok(config)
    }
  }

  pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = toml::to_string_pretty(self)?;
    std::fs::write(path, content)?;
    Ok(())
  }
}
