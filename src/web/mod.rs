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

  // Use ffmpeg to continuously decode RTSP main stream to MJPEG frames
  let rtsp_url = format!(
    "rtsp://{}:{}@{}:554/Streaming/Channels/{}",
    dvr.username, dvr.password, dvr.ip, channel
  );

  let stream = async_stream::stream! {
    // Spawn ONE persistent ffmpeg process that outputs continuous JPEG frames
    let mut child = match tokio::process::Command::new("ffmpeg")
      .args(["-rtsp_transport", "tcp", "-stimeout", "5000000",
             "-i", &rtsp_url,
             "-vf", "fps=5,scale=480:-1",
             "-f", "image2pipe", "-vcodec", "mjpeg",
             "-q:v", "5",
             "-"])
      .stdout(std::process::Stdio::piped())
      .stderr(std::process::Stdio::null())
      .spawn()
    {
      Ok(c) => c,
      Err(_) => return,
    };

    let stdout = match child.stdout.take() {
      Some(s) => s,
      None => return,
    };

    use tokio::io::AsyncReadExt;
    let mut reader = tokio::io::BufReader::new(stdout);
    let mut buf = vec![0u8; 512 * 1024]; // 512KB buffer

    loop {
      // Read JPEG frames from ffmpeg pipe
      // JPEG starts with FF D8, ends with FF D9
      let mut frame_data = Vec::new();
      let mut found_start = false;

      loop {
        let n = match reader.read(&mut buf).await {
          Ok(0) => break, // EOF
          Ok(n) => n,
          Err(_) => break,
        };

        for i in 0..n {
          if !found_start {
            if i + 1 < n && buf[i] == 0xFF && buf[i + 1] == 0xD8 {
              found_start = true;
              frame_data.clear();
              frame_data.push(buf[i]);
            }
          } else {
            frame_data.push(buf[i]);
            if frame_data.len() >= 2
              && frame_data[frame_data.len() - 2] == 0xFF
              && frame_data[frame_data.len() - 1] == 0xD9
            {
              // Complete JPEG frame
              let header = format!(
                "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                boundary, frame_data.len()
              );
              yield Ok::<_, std::io::Error>(bytes::Bytes::from(header));
              yield Ok(bytes::Bytes::from(frame_data.clone()));
              yield Ok(bytes::Bytes::from("\r\n"));
              found_start = false;
              frame_data.clear();
            }
          }
        }

        if !found_start && frame_data.is_empty() {
          continue;
        }
      }
    }

    let _ = child.kill().await;
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
  let url = format!(
    "http://{}/ISAPI/Streaming/channels/{}/picture",
    dvr.ip, channel
  );

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
