//! DVR GridMap ROI — 22x18 grid from Hikvision ISAPI motionDetection config.
//!
//! Each camera has a hex-encoded grid string (22 cols x 18 rows).
//! Each row is 6 hex chars = 24 bits, of which 22 are used (MSB-first).
//! A set bit means that grid cell is monitored (inside ROI).

use std::collections::HashMap;

const GRID_COLS: usize = 22;
const GRID_ROWS: usize = 18;
/// Each row = 6 hex chars = 24 bits; only upper 22 bits used
const HEX_PER_ROW: usize = 6;
/// Total bits per parsed row (24 from 6 hex digits)
const BITS_PER_ROW: u32 = 24;

/// DVR motion-detection grid parsed from ISAPI hex string.
#[derive(Debug, Clone)]
pub struct GridMap {
  cells: [[bool; GRID_COLS]; GRID_ROWS],
}

impl GridMap {
  /// Parse a 108-char hex string (18 rows x 6 hex chars) into a boolean grid.
  ///
  /// Bit layout per row: bits 23..2 map to columns 0..21 (MSB = col 0).
  /// Bits 1..0 are padding (always 0 in DVR output).
  pub fn from_hex(hex: &str) -> Self {
    let mut cells = [[false; GRID_COLS]; GRID_ROWS];
    for row in 0..GRID_ROWS {
      let start = row * HEX_PER_ROW;
      let end = start + HEX_PER_ROW;
      if end > hex.len() {
        break;
      }
      let row_hex = &hex[start..end];
      let bits = u32::from_str_radix(row_hex, 16).unwrap_or(0);
      for col in 0..GRID_COLS {
        // bit 23 = col 0, bit 2 = col 21
        let shift = (BITS_PER_ROW - 1) - col as u32;
        cells[row][col] = (bits >> shift) & 1 == 1;
      }
    }
    GridMap { cells }
  }

  /// Check whether a normalized coordinate (0.0..1.0) falls inside a monitored cell.
  pub fn is_monitored(&self, x_norm: f32, y_norm: f32) -> bool {
    let col = ((x_norm * GRID_COLS as f32) as usize).min(GRID_COLS - 1);
    let row = ((y_norm * GRID_ROWS as f32) as usize).min(GRID_ROWS - 1);
    self.cells[row][col]
  }

  /// Return the grid row index for a given normalized y coordinate.
  pub fn grid_row(&self, y_norm: f32) -> usize {
    ((y_norm * GRID_ROWS as f32) as usize).min(GRID_ROWS - 1)
  }

  /// Return the grid col index for a given normalized x coordinate.
  pub fn grid_col(&self, x_norm: f32) -> usize {
    ((x_norm * GRID_COLS as f32) as usize).min(GRID_COLS - 1)
  }

  /// Full-frame grid (all cells monitored). Used for interior cameras.
  pub fn full_frame() -> Self {
    GridMap {
      cells: [[true; GRID_COLS]; GRID_ROWS],
    }
  }
}

/// Build gridmaps for all 11 cameras from exact DVR ISAPI hex strings.
///
/// Front cameras (101, 201, 301, 401) and right corridor (901) have selective ROI.
/// Interior cameras (501, 601, 701, 801, 1001, 1101) use full-frame monitoring.
pub fn get_gridmaps() -> HashMap<u16, GridMap> {
  let mut maps = HashMap::new();

  // Front gate cameras — selective ROI from ISAPI
  maps.insert(
    101,
    GridMap::from_hex(
      "001ffc001ffc001ffc001ffc001ffc0001f80000080000000000000000000000000000003800007fe0007fff807ffffcfffffcfffffc",
    ),
  );
  maps.insert(
    201,
    GridMap::from_hex(
      "0000000000000000000000000000000000000000000000f00003f8001ffc00fffc03fffc1ffffcfffffcfffffcfffffcfffffcfffffc",
    ),
  );
  maps.insert(
    301,
    GridMap::from_hex(
      "3f00003f80003fc0003fe0003fe0003ff0003ff8003ffc003ffe007fff007fff807fff807fffc07fffe07ffff07ffff87ffff8fffff8",
    ),
  );
  maps.insert(
    401,
    GridMap::from_hex(
      "00000000000000000001ff801ffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc",
    ),
  );

  // Right corridor cam 9 — selective ROI (exclude road beyond compound wall)
  maps.insert(
    901,
    GridMap::from_hex(
      "c00000f80000fe0000ffc0fcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc",
    ),
  );

  // Interior cameras — full frame monitoring
  let full_hex = "fffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffcfffffc";
  for ch in [501_u16, 601, 701, 801, 1001, 1101] {
    maps.insert(ch, GridMap::from_hex(full_hex));
  }

  maps
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn full_frame_all_monitored() {
    let grid = GridMap::full_frame();
    assert!(grid.is_monitored(0.0, 0.0));
    assert!(grid.is_monitored(0.5, 0.5));
    assert!(grid.is_monitored(0.99, 0.99));
  }

  #[test]
  fn cam101_top_left_not_monitored() {
    let maps = get_gridmaps();
    let g = maps.get(&101).unwrap();
    // Top-left corner of cam 101 is road area — should NOT be monitored
    assert!(!g.is_monitored(0.0, 0.0));
  }

  #[test]
  fn cam101_bottom_right_monitored() {
    let maps = get_gridmaps();
    let g = maps.get(&101).unwrap();
    // Bottom-right is gate area — should be monitored
    assert!(g.is_monitored(0.9, 0.95));
  }

  #[test]
  fn interior_full_frame() {
    let maps = get_gridmaps();
    for ch in [501_u16, 601, 701, 801, 1001, 1101] {
      let g = maps.get(&ch).unwrap();
      assert!(g.is_monitored(0.1, 0.1));
      assert!(g.is_monitored(0.9, 0.9));
    }
  }

  #[test]
  fn hex_length_correct() {
    // 18 rows x 6 hex chars = 108 characters
    let hex = "001ffc001ffc001ffc001ffc001ffc0001f80000080000000000000000000000000000003800007fe0007fff807ffffcfffffcfffffc";
    assert_eq!(hex.len(), 108);
  }
}
