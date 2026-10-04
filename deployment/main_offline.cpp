/**
 * @file main_offline.cpp
 * @brief Offline Perception Testing (Image Files / ROS2 Bag Frames)
 *
 * Use this to test the perception pipeline without a live camera:
 * - Single image: ./unina_perception_offline engine.plan image.jpg
 * - Directory:    ./unina_perception_offline engine.plan ./frames/ --dir
 * - Video:        ./unina_perception_offline engine.plan video.mp4
 *
 * For ROS2 bags: Extract frames to a directory first, then use --dir mode.
 */

#include "Config.hpp"
#include "FileImageSource.hpp"
#include "Perception3D.hpp"

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

// ============================================================================
// Signal Handling
// ============================================================================

static volatile bool g_running = true;

void signal_handler(int signum) {
  std::cerr << "\nReceived signal " << signum << ", stopping..." << std::endl;
  g_running = false;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  if (argc < 3) {
    std::cerr << "Usage: ./unina_perception_offline <engine_path> <input_path> "
                 "[--dir] [--output output_dir]"
              << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  ./unina_perception_offline model.plan test.jpg"
              << std::endl;
    std::cerr << "  ./unina_perception_offline model.plan ./frames/ --dir"
              << std::endl;
    std::cerr << "  ./unina_perception_offline model.plan video.mp4 --output "
                 "./results/"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string input_path = argv[2];
  bool is_directory = false;
  std::string output_dir = "";

  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--dir") {
      is_directory = true;
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    }
  }

  // ========== Initialize Image Source ==========
  unina::FileImageSource source(input_path, is_directory);
  if (!source.init()) {
    std::cerr << "Failed to initialize image source" << std::endl;
    return -1;
  }

  // ========== Initialize Perception ==========
  unina::Perception3DConfig config;
  config.engine_path = engine_path;
  config.conf_threshold = 0.25f;
  config.enable_3d = false; // No 3D for offline (no depth)

  unina::Perception3D perception;
  if (!perception.init(config)) {
    std::cerr << "Failed to initialize perception pipeline" << std::endl;
    return -1;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Processing " << source.total_frames() << " frames..."
            << std::endl;
  std::cout << "========================================\n" << std::endl;

  // ========== Process Loop ==========
  unina::ImageFrame frame;
  double total_latency_ms = 0;
  int total_detections = 0;
  int frame_count = 0;

  // CSV output for analysis
  std::ofstream csv_out;
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
    csv_out.open(output_dir + "/detections.csv");
    csv_out << "frame_id,class,x1,y1,x2,y2,confidence\n";
  }

  while (g_running && source.has_more()) {
    if (!source.grab(frame)) {
      break;
    }

    // Read image for visualization (FileImageSource provides CPU data)
    cv::Mat vis_image(frame.height, frame.width, CV_8UC3,
                      const_cast<uint8_t *>(frame.cpu_data));

    auto result = perception.process_2d(frame);

    total_latency_ms += result.latency_ms;
    total_detections += result.cones.size();

    // Draw detections
    for (const auto &cone : result.cones) {
      cv::Scalar color(0, 255, 0);
      switch (cone.color) {
      case unina::ConeColor::BIG_ORANGE:
        color = cv::Scalar(0, 0, 255);
        break;
      case unina::ConeColor::BLUE:
        color = cv::Scalar(255, 0, 0);
        break;
      case unina::ConeColor::SMALL_ORANGE:
        color = cv::Scalar(0, 165, 255);
        break;
      case unina::ConeColor::YELLOW:
        color = cv::Scalar(0, 255, 255);
        break;
      default:
        break;
      }

      cv::rectangle(vis_image, cv::Point(cone.bbox_x1, cone.bbox_y1),
                    cv::Point(cone.bbox_x2, cone.bbox_y2), color, 2);

      std::string label = std::string(cone.color_name()) + " " +
                          std::to_string(cone.confidence).substr(0, 4);
      cv::putText(vis_image, label, cv::Point(cone.bbox_x1, cone.bbox_y1 - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

      // CSV output
      if (csv_out.is_open()) {
        csv_out << frame_count << "," << cone.color_name() << ","
                << cone.bbox_x1 << "," << cone.bbox_y1 << "," << cone.bbox_x2
                << "," << cone.bbox_y2 << "," << cone.confidence << "\n";
      }
    }

    // Save or display
    if (!output_dir.empty()) {
      std::string out_path =
          output_dir + "/frame_" + std::to_string(frame_count) + ".jpg";
      cv::imwrite(out_path, vis_image);
    }

    // Progress
    if (frame_count % 10 == 0 || source.total_frames() == 1) {
      std::cout << "Frame " << frame_count << "/" << source.total_frames()
                << " | Detections: " << result.cones.size()
                << " | Latency: " << result.latency_ms << "ms" << std::endl;
    }

    frame_count++;
  }

  // ========== Summary ==========
  std::cout << "\n===== Processing Complete =====" << std::endl;
  std::cout << "Frames processed: " << frame_count << std::endl;
  std::cout << "Total detections: " << total_detections << std::endl;
  std::cout << "Avg latency: " << (total_latency_ms / frame_count) << " ms"
            << std::endl;
  std::cout << "Avg FPS: " << (1000.0 / (total_latency_ms / frame_count))
            << std::endl;

  if (!output_dir.empty()) {
    std::cout << "Results saved to: " << output_dir << std::endl;
  }

  // If single image, save output
  if (source.total_frames() == 1 && output_dir.empty()) {
    cv::Mat img = cv::imread(input_path);
    auto result = perception.process_2d(frame);
    for (const auto &cone : result.cones) {
      cv::Scalar color(0, 255, 0);
      cv::rectangle(img, cv::Point(cone.bbox_x1, cone.bbox_y1),
                    cv::Point(cone.bbox_x2, cone.bbox_y2), color, 2);
    }
    cv::imwrite("output.jpg", img);
    std::cout << "Saved output.jpg" << std::endl;
  }

  return 0;
}
