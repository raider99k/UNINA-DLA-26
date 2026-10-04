/**
 * @file main_zed.cpp
 * @brief Live 3D Perception with ZED X Camera (Modular Version)
 *
 * This is the production entry point for live camera perception.
 * Uses the modular Perception3D and ZedImageSource classes.
 *
 * Build with: cmake -DUSE_ZED=ON .. && make
 */

#ifdef USE_ZED_SDK

#include "Config.hpp"
#include "Perception3D.hpp"
#include "ZedImageSource.hpp"

#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

// ============================================================================
// Signal Handling
// ============================================================================

static volatile bool g_running = true;

void signal_handler(int signum) {
  std::cerr << "\nReceived signal " << signum << ", shutting down..."
            << std::endl;
  g_running = false;
}

void setup_signal_handlers() {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  setup_signal_handlers();

  if (argc < 2) {
    std::cerr << "Usage: ./unina_perception_zed <engine_path> [config_path]"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string config_path = (argc > 2) ? argv[2] : "config.yaml";

  // ========== Load Configuration ==========
  unina::Config config;
  if (!config.load(config_path)) {
    std::cout << "Warning: Config file not found, using defaults" << std::endl;
  }

  // ========== Initialize ZED Camera ==========
  unina::ZedConfig zed_config;
  zed_config.resolution = config.get_string("zed_resolution", "HD720");
  zed_config.fps = config.get_int("zed_fps", 30);
  zed_config.enable_depth = config.get_bool("zed_depth_enabled", true);
  zed_config.depth_mode = config.get_string("zed_depth_mode", "PERFORMANCE");

  unina::ZedImageSource camera;
  if (!camera.init(zed_config)) {
    std::cerr << "Failed to initialize ZED camera" << std::endl;
    return -1;
  }

  // ========== Initialize Perception Pipeline ==========
  unina::Perception3DConfig perc_config;
  perc_config.engine_path = engine_path;
  perc_config.conf_threshold = config.get_float("conf_threshold", 0.25f);
  perc_config.score_threshold = config.get_float("score_threshold", 0.45f);
  perc_config.nms_iou_threshold = config.get_float("nms_iou_threshold", 0.60f);
  perc_config.max_detections = config.get_int("max_detections", 1000);
  if (perc_config.max_detections < 1) {
    perc_config.max_detections = 1;
  }
  perc_config.use_nms = config.get_bool("use_nms", false);
  perc_config.enable_3d = zed_config.enable_depth;

  unina::Perception3D perception;
  if (!perception.init(perc_config)) {
    std::cerr << "Failed to initialize perception pipeline" << std::endl;
    return -1;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Starting 3D perception loop. Press Ctrl+C to stop."
            << std::endl;
  std::cout << "========================================\n" << std::endl;

  // ========== Main Loop ==========
  unina::ImageFrame frame;
  int frame_count = 0;
  double total_latency_ms = 0;

  while (g_running) {
    // Grab frame from ZED
    if (!camera.grab(frame)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Run 2D detection
    auto detections = perception.detect_2d(frame);

    // Fuse with depth to get 3D
    auto cones = camera.fuse_3d(detections);

    auto t1 = std::chrono::high_resolution_clock::now();
    double latency_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_latency_ms += latency_ms;

    // Output
    if (frame_count % 30 == 0) {
      int valid_3d = 0;
      for (const auto &cone : cones) {
        if (cone.is_valid)
          valid_3d++;
      }

      std::cout << "Frame " << frame_count
                << " | FPS: " << int(1000.0 / latency_ms)
                << " | 2D: " << detections.size() << " | 3D: " << valid_3d
                << " | Latency: " << latency_ms << "ms" << std::endl;

      // Print 3D positions
      for (const auto &cone : cones) {
        if (cone.is_valid) {
          std::cout << "  [" << cone.color_name() << "] "
                    << "X=" << cone.x << "m, "
                    << "Y=" << cone.y << "m, "
                    << "Z=" << cone.z << "m "
                    << "(dist=" << cone.distance() << "m)" << std::endl;
        }
      }
    }

    frame_count++;
  }

  // ========== Summary ==========
  if (frame_count > 0) {
    std::cout << "\n===== Summary (" << frame_count
              << " frames) =====" << std::endl;
    std::cout << "Avg Latency: " << (total_latency_ms / frame_count) << " ms"
              << std::endl;
    std::cout << "Avg FPS: " << (1000.0 / (total_latency_ms / frame_count))
              << std::endl;
  }

  // ========== Cleanup ==========
  camera.close();

  return 0;
}

#else

#include <iostream>

int main() {
  std::cerr << "ZED SDK not enabled. Build with: cmake -DUSE_ZED=ON .."
            << std::endl;
  return -1;
}

#endif // USE_ZED_SDK
