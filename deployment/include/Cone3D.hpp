#pragma once

/**
 * @file Cone3D.hpp
 * @brief 3D Cone detection output structure
 *
 * This is the final output of the perception pipeline - ready for
 * SLAM/Planning.
 */

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace unina {

/**
 * @brief Cone color classification
 */
enum class ConeColor : uint8_t {
  YELLOW = 0,
  BLUE = 1,
  SMALL_ORANGE = 2,
  BIG_ORANGE = 3,
  LARGE_ORANGE = 3,
  UNKNOWN = 255
};

/**
 * @brief 3D cone detection result
 *
 * Contains both the 2D detection info and the 3D position.
 * Position is in the camera coordinate frame (meters):
 *   X = right, Y = down, Z = forward (OpenCV convention)
 * Or configurable to ROS convention (X=forward, Y=left, Z=up).
 */
struct Cone3D {
  // === 3D Position (meters, camera frame) ===
  float x = 0.0f; ///< Right (or forward in ROS)
  float y = 0.0f; ///< Down (or left in ROS)
  float z = 0.0f; ///< Forward (or up in ROS)

  // === Classification ===
  ConeColor color = ConeColor::UNKNOWN;
  float confidence = 0.0f;

  // === 2D Bounding Box (pixels, for visualization) ===
  int bbox_x1 = 0;
  int bbox_y1 = 0;
  int bbox_x2 = 0;
  int bbox_y2 = 0;

  // === Tracking (optional, from ZED SDK) ===
  int track_id = -1; ///< -1 if not tracked

  // === Validity ===
  bool is_valid = false; ///< False if 3D position couldn't be computed

  /**
   * @brief Get color as string
   */
  const char *color_name() const {
    switch (color) {
    case ConeColor::YELLOW:
      return "Yellow";
    case ConeColor::BLUE:
      return "Blue";
    case ConeColor::SMALL_ORANGE:
      return "SmallOrange";
    case ConeColor::BIG_ORANGE:
      return "LargeOrange";
    default:
      return "Unknown";
    }
  }

  /**
   * @brief Distance from camera (Euclidean)
   */
  float distance() const { return std::sqrt(x * x + y * y + z * z); }
};

/**
 * @brief Collection of 3D cones from a single frame
 */
struct FrameResult {
  uint64_t timestamp_ns = 0; ///< Frame timestamp (nanoseconds)
  int frame_id = 0;
  std::vector<Cone3D> cones;
  double latency_ms = 0.0; ///< Total processing time
};

} // namespace unina
