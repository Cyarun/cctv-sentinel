#!/usr/bin/env bash
# Deploy motion-tracker-v2 to NUC (shadow run, never touches v1)
# Run this from the NUC: bash DEPLOY_NUC.sh
# Or from dev machine: ssh zeek@192.168.1.14 'bash -s' < DEPLOY_NUC.sh

set -e

NUC_V2_DIR="/opt/cctvanalytics/motion-tracker-v2"
SENTINEL_REPO="https://github.com/Cyarun/cctv-sentinel.git"
EVENT_DIR="/opt/cctvanalytics/motion_events_v2"
V1_BIN="/opt/cctvanalytics/motion-tracker/target/release/motion-tracker"
V2_BIN="$NUC_V2_DIR/target/release/motion-tracker-v2"

echo "=== CCTV motion-tracker-v2 NUC deployment ==="
echo "Shadow run: writes to $EVENT_DIR, never touches v1"
echo ""

# 1. Clone or update
if [ -d "$NUC_V2_DIR/.git" ]; then
    echo "[1/5] Updating existing repo..."
    cd "$NUC_V2_DIR"
    git pull
else
    echo "[1/5] Cloning cctv-sentinel (motion-tracker-v2 subdirectory)..."
    TMP=$(mktemp -d)
    git clone --depth=1 "$SENTINEL_REPO" "$TMP/sentinel"
    sudo mkdir -p "$NUC_V2_DIR"
    sudo cp -r "$TMP/sentinel/motion-tracker-v2/." "$NUC_V2_DIR/"
    sudo chown -R zeek:zeek "$NUC_V2_DIR"
    rm -rf "$TMP"
fi

# 2. Build
echo "[2/5] Building (release)..."
cd "$NUC_V2_DIR"
cargo build --release 2>&1
echo "  Binary: $V2_BIN"
ls -lh "$V2_BIN"

# 3. Create event dir
echo "[3/5] Creating event dir $EVENT_DIR..."
mkdir -p "$EVENT_DIR"

# 4. Quick sanity: run in playback mode on a 2-minute window
# Adjust START/END to a time window that has archived events on your NUC
# Format: YYYY-MM-DDTHH:MM:SS
# Example (update to match actual archived event timestamps):
echo "[4/5] Playback sanity check (edit START/END in script if needed)..."
START=${1:-"2025-01-01T00:00:00"}
END=${2:-"2025-01-01T00:02:00"}
echo "  Playback: $START → $END (cams 1,2,3,10)"
MOTION_EVENT_DIR="$EVENT_DIR/playback_test" \
    "$V2_BIN" --playback "$START" "$END" 101,201,301,1001 || true

# 5. Print shadow run instructions
echo ""
echo "=== SHADOW RUN INSTRUCTIONS ==="
echo ""
echo "Run v2 alongside v1 (both write to separate dirs, no interference):"
echo ""
echo "  # Terminal 1 — v1 still running (production)"
echo "  systemctl status motion-tracker  # or however v1 is started"
echo ""
echo "  # Terminal 2 — v2 shadow run"
echo "  MOTION_EVENT_DIR=$EVENT_DIR $V2_BIN"
echo ""
echo "Monitor v2 events:"
echo "  watch -n 5 'ls -lt $EVENT_DIR | head -20'"
echo ""
echo "After ≥48h shadow run, review v2 events vs v1 events."
echo "Switchover: update /opt/cctvanalytics/cctv_watcher.py"
echo "  EVENT_DIR env var → $EVENT_DIR"
echo "  Binary → $V2_BIN"
echo "  Keep v1 binary as: $V1_BIN.bak"
echo ""
echo "=== Done ==="
