mod alerts;
mod camera;
mod config;
mod detector;
mod tracker;
mod web;

use axum::{
  routing::{get, post},
  Router,
};
use config::AppConfig;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct AppState {
  pub config: AppConfig,
  pub db: Mutex<Connection>,
  pub event_tx: broadcast::Sender<String>,
}

fn init_database(db: &Connection) {
  db.execute_batch(
    "CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      camera_id INTEGER NOT NULL,
      camera_name TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      object_label TEXT NOT NULL,
      confidence REAL NOT NULL,
      bbox_x REAL NOT NULL,
      bbox_y REAL NOT NULL,
      bbox_w REAL NOT NULL,
      bbox_h REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);",
  )
  .expect("Failed to initialize database schema");

  // Seed default admin user if none exists (password: admin)
  let user_count: i64 = db
    .query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0))
    .unwrap_or(0);

  if user_count == 0 {
    let hash = bcrypt::hash("admin", bcrypt::DEFAULT_COST).expect("Failed to hash password");
    db.execute(
      "INSERT INTO users (username, password_hash) VALUES (?1, ?2)",
      rusqlite::params!["admin", hash],
    )
    .expect("Failed to seed admin user");
    tracing::info!("Seeded default admin user (admin/admin)");
  }
}

#[tokio::main]
async fn main() {
  // Initialize tracing
  tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer())
    .with(
      tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "cctv_sentinel=info,tower_http=info".into()),
    )
    .init();

  // Load configuration
  let config = AppConfig::load("config.toml").unwrap_or_else(|e| {
    tracing::warn!("Failed to load config.toml: {}, using defaults", e);
    AppConfig::default()
  });

  let bind_addr = format!("{}:{}", config.server.host, config.server.port);
  let static_dir = config.server.static_dir.clone();

  // Initialize SQLite
  let db = Connection::open("sentinel.db").expect("Failed to open SQLite database");
  db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
    .expect("Failed to set SQLite pragmas");
  init_database(&db);

  // Broadcast channel for real-time WebSocket events
  let (event_tx, _) = broadcast::channel::<String>(256);

  let state = Arc::new(AppState {
    config,
    db: Mutex::new(db),
    event_tx,
  });

  // Build router
  let app = Router::new()
    // Pages
    .route("/", get(web::dashboard_handler))
    .route("/login", post(web::login_handler))
    // API
    .route("/api/cameras", get(web::cameras_handler))
    .route("/api/events", get(web::events_handler))
    .route(
      "/api/settings",
      get(web::get_settings_handler).put(web::put_settings_handler),
    )
    // Camera feeds
    .route("/api/snapshot/:channel", get(web::snapshot_handler))
    .route("/api/mjpeg/:channel", get(web::mjpeg_handler))
    // WebSocket
    .route("/ws", get(web::ws_handler))
    // Static files
    .nest_service("/static", ServeDir::new(&static_dir))
    .with_state(state);

  tracing::info!("CCTV Sentinel starting on {}", bind_addr);
  tracing::info!("Dashboard: http://localhost:8080");
  tracing::info!("Default credentials: admin / admin");

  let listener = tokio::net::TcpListener::bind(&bind_addr)
    .await
    .expect("Failed to bind to address");

  axum::serve(listener, app)
    .await
    .expect("Server error");
}
