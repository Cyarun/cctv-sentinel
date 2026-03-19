use axum::{
  extract::{Json, Path, State, WebSocketUpgrade, ws::{Message, WebSocket}},
  http::{HeaderMap, StatusCode, header},
  response::{Html, IntoResponse, Response},
  body::Body,
};
use chrono::Utc;
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

use std::sync::atomic::Ordering;
use crate::AppState;

// --- Data types ---

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
  pub sub: String,
  pub exp: usize,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
  pub username: String,
  pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
  pub token: String,
  pub username: String,
  pub expires_in: u64,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
  pub error: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionEvent {
  pub id: String,
  pub camera_id: u32,
  pub camera_name: String,
  pub timestamp: String,
  pub object_label: String,
  pub confidence: f32,
  pub bbox: [f32; 4],
}

#[derive(Debug, Deserialize)]
pub struct SettingsUpdate {
  pub dvr_ip: Option<String>,
  pub dvr_username: Option<String>,
  pub dvr_password: Option<String>,
  pub telegram_enabled: Option<bool>,
  pub telegram_bot_token: Option<String>,
  pub confidence_threshold: Option<f32>,
  pub capture_interval_secs: Option<u64>,
}

// --- Auth helpers ---

pub fn create_jwt(username: &str, secret: &str, expiry_hours: u64) -> Result<String, StatusCode> {
  let expiration = Utc::now()
    .checked_add_signed(chrono::Duration::hours(expiry_hours as i64))
    .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?
    .timestamp() as usize;

  let claims = Claims {
    sub: username.to_owned(),
    exp: expiration,
  };

  encode(
    &Header::default(),
    &claims,
    &EncodingKey::from_secret(secret.as_bytes()),
  )
  .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub fn validate_jwt(token: &str, secret: &str) -> Result<Claims, StatusCode> {
  decode::<Claims>(
    token,
    &DecodingKey::from_secret(secret.as_bytes()),
    &Validation::default(),
  )
  .map(|data| data.claims)
  .map_err(|_| StatusCode::UNAUTHORIZED)
}

pub fn extract_token(headers: &HeaderMap) -> Result<String, StatusCode> {
  headers
    .get("Authorization")
    .and_then(|v| v.to_str().ok())
    .and_then(|v| v.strip_prefix("Bearer "))
    .map(|s| s.to_string())
    .ok_or(StatusCode::UNAUTHORIZED)
}

// --- Handlers ---

pub async fn login_handler(
  State(state): State<Arc<AppState>>,
  Json(body): Json<LoginRequest>,
) -> Response {
  let db = state.db.lock().unwrap();

  let result = db.query_row(
    "SELECT password_hash FROM users WHERE username = ?1",
    params![body.username],
    |row| row.get::<_, String>(0),
  );

  match result {
    Ok(hash) => {
      if bcrypt::verify(&body.password, &hash).unwrap_or(false) {
        match create_jwt(
          &body.username,
          &state.config.auth.jwt_secret,
          state.config.auth.token_expiry_hours,
        ) {
          Ok(token) => Json(LoginResponse {
            token,
            username: body.username,
            expires_in: state.config.auth.token_expiry_hours * 3600,
          })
          .into_response(),
          Err(status) => (status, Json(ApiError { error: "Token generation failed".into() })).into_response(),
        }
      } else {
        (StatusCode::UNAUTHORIZED, Json(ApiError { error: "Invalid credentials".into() })).into_response()
      }
    }
    Err(_) => {
      (StatusCode::UNAUTHORIZED, Json(ApiError { error: "Invalid credentials".into() })).into_response()
    }
  }
}

pub async fn dashboard_handler() -> Html<String> {
  match tokio::fs::read_to_string("./static/index.html").await {
    Ok(content) => Html(content),
    Err(_) => Html("<h1>Dashboard not found</h1>".into()),
  }
}

pub async fn cameras_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }
  Json(&state.config.cameras).into_response()
}

pub async fn events_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  let db = state.db.lock().unwrap();
  let mut stmt = db
    .prepare(
      "SELECT id, camera_id, camera_name, timestamp, object_label, confidence, bbox_x, bbox_y, bbox_w, bbox_h \
       FROM events ORDER BY timestamp DESC LIMIT 100",
    )
    .unwrap();

  let events: Vec<DetectionEvent> = stmt
    .query_map([], |row| {
      Ok(DetectionEvent {
        id: row.get(0)?,
        camera_id: row.get(1)?,
        camera_name: row.get(2)?,
        timestamp: row.get(3)?,
        object_label: row.get(4)?,
        confidence: row.get(5)?,
        bbox: [
          row.get(6)?,
          row.get(7)?,
          row.get(8)?,
          row.get(9)?,
        ],
      })
    })
    .unwrap()
    .filter_map(|e| e.ok())
    .collect();

  Json(events).into_response()
}

pub async fn get_settings_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  let config = &state.config;
  Json(serde_json::json!({
    "dvr_ip": config.dvr.ip,
    "dvr_username": config.dvr.username,
    "telegram_enabled": config.telegram.enabled,
    "telegram_bot_token": config.telegram.bot_token,
    "confidence_threshold": config.detection.confidence_threshold,
    "capture_interval_secs": config.detection.capture_interval_secs,
    "cameras": config.cameras,
  }))
  .into_response()
}

pub async fn put_settings_handler(
  State(_state): State<Arc<AppState>>,
  headers: HeaderMap,
  Json(body): Json<SettingsUpdate>,
) -> Response {
  if let Err(status) = auth_check(&headers, &_state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  // In production, we would mutate the config and save it.
  // For now, log what was received.
  tracing::info!("Settings update received: {:?}", body);

  Json(serde_json::json!({ "status": "ok", "message": "Settings updated" })).into_response()
}

pub async fn ws_handler(
  ws: WebSocketUpgrade,
  State(state): State<Arc<AppState>>,
) -> Response {
  ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>) {
  let mut rx = state.event_tx.subscribe();

  // Send initial connection acknowledgement
  let ack = serde_json::json!({
    "type": "connected",
    "timestamp": Utc::now().to_rfc3339(),
    "cameras": state.config.cameras.len(),
  });
  if socket.send(Message::Text(ack.to_string().into())).await.is_err() {
    return;
  }

  // Forward broadcast events to WebSocket client
  loop {
    tokio::select! {
      msg = rx.recv() => {
        match msg {
          Ok(event_json) => {
            if socket.send(Message::Text(event_json.into())).await.is_err() {
              break;
            }
          }
          Err(broadcast::error::RecvError::Lagged(n)) => {
            tracing::warn!("WebSocket client lagged by {} messages", n);
          }
          Err(_) => break,
        }
      }
      msg = socket.recv() => {
        match msg {
          Some(Ok(Message::Close(_))) | None => break,
          Some(Ok(Message::Ping(data))) => {
            if socket.send(Message::Pong(data)).await.is_err() {
              break;
            }
          }
          _ => {}
        }
      }
    }
  }
}

// --- MJPEG streaming (no browser polling needed) ---

pub async fn mjpeg_handler(
  Path(channel): Path<u16>,
  State(state): State<Arc<AppState>>,
) -> Response {
  let dvr = state.config.dvr.clone();
  let boundary = "cctvsentinelframe";

  let dvr_ip = dvr.ip.clone();
  let dvr_user = dvr.username.clone();
  let dvr_pass = dvr.password.clone();

  let stream = async_stream::stream! {
    loop {
      // Grab one snapshot per cycle using curl (reliable, handles digest auth)
      let output = tokio::process::Command::new("curl")
        .args(["-s", "--digest", "-u",
               &format!("{}:{}", dvr_user, dvr_pass),
               "-m", "3",
               &format!("http://{}/ISAPI/Streaming/channels/{}/picture", dvr_ip, channel)])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .await;

      if let Ok(out) = output {
        if out.status.success() && out.stdout.len() > 1000 {
          let header = format!(
            "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            boundary, out.stdout.len()
          );
          yield Ok::<_, std::io::Error>(bytes::Bytes::from(header));
          yield Ok(bytes::Bytes::from(out.stdout));
          yield Ok(bytes::Bytes::from("\r\n"));
        }
      }
      // 200ms between frames = ~5fps
      tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
  };

  let body = Body::from_stream(stream);
  Response::builder()
    .status(200)
    .header(header::CONTENT_TYPE, format!("multipart/x-mixed-replace; boundary={}", boundary))
    .header(header::CACHE_CONTROL, "no-cache, no-store")
    .body(body)
    .unwrap()
}

fn auth_check(headers: &HeaderMap, state: &Arc<AppState>) -> Result<Claims, StatusCode> {
  let token = extract_token(headers)?;
  validate_jwt(&token, &state.config.auth.jwt_secret)
}

// --- Snapshot proxy ---

pub async fn snapshot_handler(
  Path(channel): Path<u16>,
  State(state): State<Arc<AppState>>,
) -> Response {
  let dvr = &state.config.dvr;

  // Use curl with digest auth to grab ISAPI snapshot
  let output = tokio::process::Command::new("curl")
    .args(["-s", "--digest", "-u",
           &format!("{}:{}", dvr.username, dvr.password),
           "-m", "5",
           &format!("http://{}/ISAPI/Streaming/channels/{}/picture", dvr.ip, channel)])
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::null())
    .output()
    .await;

  match output
  {
    Ok(out) => {
      if out.status.success() && out.stdout.len() > 1000 {
        (
          StatusCode::OK,
          [(header::CONTENT_TYPE, "image/jpeg"),
           (header::CACHE_CONTROL, "no-cache, no-store")],
          out.stdout,
        ).into_response()
      } else {
        StatusCode::BAD_GATEWAY.into_response()
      }
    }
    Err(e) => {
      tracing::error!("Snapshot failed for channel {}: {}", channel, e);
      StatusCode::BAD_GATEWAY.into_response()
    }
  }
}

// --- Detection API ---

/// Grab a snapshot from the DVR and run YOLO inference, returning JSON detections.
pub async fn detect_handler(
  Path(channel): Path<u16>,
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  let dvr = &state.config.dvr;
  let conf_threshold = state.config.detection.confidence_threshold;
  let model_path = state.config.detection.model_path.clone();

  // 1. Grab snapshot via curl (ISAPI HTTP, reliable digest auth)
  let snapshot = tokio::process::Command::new("curl")
    .args(["-s", "--digest", "-u",
           &format!("{}:{}", dvr.username, dvr.password),
           "-m", "5",
           &format!("http://{}/ISAPI/Streaming/channels/{}/picture", dvr.ip, channel)])
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::null())
    .output()
    .await;

  let jpeg_bytes = match snapshot {
    Ok(out) if out.status.success() && out.stdout.len() > 1000 => out.stdout,
    _ => {
      return (
        StatusCode::BAD_GATEWAY,
        Json(ApiError { error: format!("Failed to grab snapshot from channel {}", channel) }),
      ).into_response();
    }
  };

  // 2. Write JPEG to temp file, decode to raw RGB via Python, run inference
  let tmp_dir = std::env::temp_dir().join("cctv-sentinel");
  let _ = std::fs::create_dir_all(&tmp_dir);
  let ts = chrono::Utc::now().timestamp_millis();
  let jpeg_path = tmp_dir.join(format!("snap_{}_{}.jpg", channel, ts));
  if std::fs::write(&jpeg_path, &jpeg_bytes).is_err() {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      Json(ApiError { error: "Failed to write temp snapshot".into() }),
    ).into_response();
  }

  // Use the inference_jpeg.py wrapper that handles JPEG input directly
  let script_path = std::path::Path::new("src/detector/inference_jpeg.py");
  let effective_model = if std::path::Path::new(&model_path).exists() {
    model_path
  } else {
    "/opt/cctvanalytics/models/yolo26n_cctv/bestv4.onnx".to_string()
  };

  let output = tokio::process::Command::new("python3")
    .args([
      script_path.to_str().unwrap_or("src/detector/inference_jpeg.py"),
      jpeg_path.to_str().unwrap_or(""),
      &effective_model,
      "352",  // model input size
      &conf_threshold.to_string(),
    ])
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .output()
    .await;

  // Clean up temp file
  let _ = std::fs::remove_file(&jpeg_path);

  match output {
    Ok(out) => {
      if out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        // Return the raw JSON array from inference
        (
          StatusCode::OK,
          [(header::CONTENT_TYPE, "application/json")],
          stdout.trim().to_string(),
        ).into_response()
      } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        tracing::error!("Detection failed for channel {}: {}", channel, stderr);
        (
          StatusCode::INTERNAL_SERVER_ERROR,
          Json(ApiError { error: format!("Inference failed: {}", stderr.trim()) }),
        ).into_response()
      }
    }
    Err(e) => {
      tracing::error!("Failed to spawn inference: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError { error: "Failed to run detection".into() }),
      ).into_response()
    }
  }
}

/// Start the background detection loop that runs YOLO round-robin on all cameras.
pub async fn detect_start_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  // Check if already running
  if state.detection_running.load(Ordering::SeqCst) {
    return Json(serde_json::json!({
      "status": "already_running",
      "message": "Detection loop is already active"
    })).into_response();
  }

  state.detection_running.store(true, Ordering::SeqCst);

  let state_clone = state.clone();
  tokio::spawn(async move {
    detection_loop(state_clone).await;
  });

  Json(serde_json::json!({
    "status": "started",
    "message": "Detection loop started"
  })).into_response()
}

/// Stop the background detection loop.
pub async fn detect_stop_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  state.detection_running.store(false, Ordering::SeqCst);

  Json(serde_json::json!({
    "status": "stopped",
    "message": "Detection loop will stop after current cycle"
  })).into_response()
}

/// Get detection loop status.
pub async fn detect_status_handler(
  State(state): State<Arc<AppState>>,
  headers: HeaderMap,
) -> Response {
  if let Err(status) = auth_check(&headers, &state) {
    return (status, Json(ApiError { error: "Unauthorized".into() })).into_response();
  }

  let running = state.detection_running.load(Ordering::SeqCst);
  Json(serde_json::json!({
    "running": running
  })).into_response()
}

/// Background detection loop: round-robin YOLO on all enabled cameras.
/// Grabs snapshot, runs inference, broadcasts detections via WebSocket, stores in SQLite.
async fn detection_loop(state: Arc<AppState>) {
  tracing::info!("Detection loop started");

  // Simple centroid tracker state per camera: track_id -> {class, positions, first_seen}
  let mut tracker_state: std::collections::HashMap<u32, std::collections::HashMap<u32, TrackerEntry>> =
    std::collections::HashMap::new();
  let mut next_track_id: u32 = 1;

  loop {
    if !state.detection_running.load(Ordering::SeqCst) {
      tracing::info!("Detection loop stopped");
      break;
    }

    let cameras = state.config.cameras.clone();
    let dvr = state.config.dvr.clone();
    let conf_threshold = state.config.detection.confidence_threshold;
    let model_path = state.config.detection.model_path.clone();

    for cam in &cameras {
      if !cam.enabled {
        continue;
      }
      if !state.detection_running.load(Ordering::SeqCst) {
        break;
      }

      let channel = cam.channel;
      let cam_name = cam.name.clone();
      let cam_id = cam.id;

      // 1. Grab snapshot
      let snapshot = tokio::process::Command::new("curl")
        .args(["-s", "--digest", "-u",
               &format!("{}:{}", dvr.username, dvr.password),
               "-m", "3",
               &format!("http://{}/ISAPI/Streaming/channels/{}/picture", dvr.ip, channel)])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .await;

      let jpeg_bytes = match snapshot {
        Ok(out) if out.status.success() && out.stdout.len() > 1000 => out.stdout,
        _ => {
          tracing::warn!("Snapshot grab failed for camera {} (ch {})", cam_id, channel);
          continue;
        }
      };

      // 2. Write to temp, run inference
      let tmp_dir = std::env::temp_dir().join("cctv-sentinel");
      let _ = std::fs::create_dir_all(&tmp_dir);
      let ts = chrono::Utc::now().timestamp_millis();
      let jpeg_path = tmp_dir.join(format!("det_{}_{}.jpg", channel, ts));
      if std::fs::write(&jpeg_path, &jpeg_bytes).is_err() {
        continue;
      }

      let effective_model = if std::path::Path::new(&model_path).exists() {
        model_path.clone()
      } else {
        "/opt/cctvanalytics/models/yolo26n_cctv/bestv4.onnx".to_string()
      };

      let output = tokio::process::Command::new("python3")
        .args([
          "src/detector/inference_jpeg.py",
          jpeg_path.to_str().unwrap_or(""),
          &effective_model,
          "352",
          &conf_threshold.to_string(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .await;

      let _ = std::fs::remove_file(&jpeg_path);

      let detections: Vec<serde_json::Value> = match output {
        Ok(out) if out.status.success() => {
          let stdout = String::from_utf8_lossy(&out.stdout);
          serde_json::from_str(stdout.trim()).unwrap_or_default()
        }
        _ => continue,
      };

      if detections.is_empty() {
        // Broadcast empty detection to clear overlays
        let empty_msg = serde_json::json!({
          "type": "detection",
          "camera": channel,
          "camera_id": cam_id,
          "camera_name": cam_name,
          "timestamp": chrono::Utc::now().to_rfc3339(),
          "objects": []
        });
        let _ = state.event_tx.send(empty_msg.to_string());
        continue;
      }

      // 3. Simple centroid tracking: match detections to existing tracks
      let cam_tracks = tracker_state.entry(cam_id).or_insert_with(std::collections::HashMap::new);
      let now = chrono::Utc::now();
      let mut matched_track_ids: Vec<u32> = Vec::new();
      let mut objects: Vec<serde_json::Value> = Vec::new();

      for det in &detections {
        let class = det.get("class").and_then(|c| c.as_str()).unwrap_or("unknown");
        let conf = det.get("confidence").and_then(|c| c.as_f64()).unwrap_or(0.0);
        let x = det.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let y = det.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let w = det.get("w").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let h = det.get("h").and_then(|v| v.as_f64()).unwrap_or(0.0);

        // Find closest existing track of same class
        let mut best_tid: Option<u32> = None;
        let mut best_dist = 0.15_f64; // max match distance
        for (&tid, entry) in cam_tracks.iter() {
          if matched_track_ids.contains(&tid) { continue; }
          if entry.class != class { continue; }
          let (lx, ly) = entry.last_pos();
          let dist = ((x - lx).powi(2) + (y - ly).powi(2)).sqrt();
          if dist < best_dist {
            best_dist = dist;
            best_tid = Some(tid);
          }
        }

        let track_id = if let Some(tid) = best_tid {
          matched_track_ids.push(tid);
          let entry = cam_tracks.get_mut(&tid).unwrap();
          entry.positions.push((x, y));
          if entry.positions.len() > 10 {
            entry.positions.remove(0);
          }
          entry.last_seen = now;
          tid
        } else {
          let tid = next_track_id;
          next_track_id += 1;
          cam_tracks.insert(tid, TrackerEntry {
            class: class.to_string(),
            positions: vec![(x, y)],
            first_seen: now,
            last_seen: now,
          });
          tid
        };

        let entry = &cam_tracks[&track_id];
        let trail: Vec<Vec<f64>> = entry.positions.iter().map(|&(px, py)| vec![px, py]).collect();
        let duration_s = (now - entry.first_seen).num_milliseconds() as f64 / 1000.0;

        // Simple intent classification based on motion
        let intent = if entry.positions.len() >= 3 {
          let (fx, fy) = entry.positions[0];
          let dy_total = y - fy;
          let dx_total = x - fx;
          if dy_total > 0.05 && y > 0.6 {
            "approaching_gate"
          } else if dx_total.abs() > dy_total.abs() * 2.0 && (dx_total.abs() > 0.08) {
            "passing"
          } else if (dx_total.abs() < 0.02) && (dy_total.abs() < 0.02) {
            "stationary"
          } else {
            "moving"
          }
        } else {
          "unknown"
        };

        objects.push(serde_json::json!({
          "class": class,
          "confidence": conf,
          "x": x,
          "y": y,
          "w": w,
          "h": h,
          "track_id": track_id,
          "trail": trail,
          "intent": intent,
          "duration_s": duration_s,
        }));

        // Store event in SQLite
        let event_id = uuid::Uuid::new_v4().to_string();
        let timestamp = now.to_rfc3339();
        if let Ok(db) = state.db.lock() {
          let _ = db.execute(
            "INSERT OR IGNORE INTO events (id, camera_id, camera_name, timestamp, object_label, confidence, bbox_x, bbox_y, bbox_w, bbox_h) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![event_id, cam_id, cam_name, timestamp, class, conf, x, y, w, h],
          );
        }
      }

      // Prune stale tracks (not seen in last 10 seconds)
      cam_tracks.retain(|_, entry| {
        (now - entry.last_seen).num_seconds() < 10
      });

      // 4. Broadcast detection event via WebSocket
      let ws_msg = serde_json::json!({
        "type": "detection",
        "camera": channel,
        "camera_id": cam_id,
        "camera_name": cam_name,
        "timestamp": now.to_rfc3339(),
        "objects": objects,
      });
      let _ = state.event_tx.send(ws_msg.to_string());

      // ~130ms pause between cameras
      tokio::time::sleep(std::time::Duration::from_millis(130)).await;
    }

    // Small pause between full rounds
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
  }
}

/// Internal tracker entry for the detection loop's simple centroid tracker.
struct TrackerEntry {
  class: String,
  positions: Vec<(f64, f64)>,
  first_seen: chrono::DateTime<chrono::Utc>,
  last_seen: chrono::DateTime<chrono::Utc>,
}

impl TrackerEntry {
  fn last_pos(&self) -> (f64, f64) {
    self.positions.last().copied().unwrap_or((0.0, 0.0))
  }
}
