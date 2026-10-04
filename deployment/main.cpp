#include "Config.hpp"
#include "CudaPostprocess.cuh"
#include "CudaPreprocess.cuh"
#include "UninaPerception.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>

// ============================================================================
// Signal Handling for Graceful Shutdown
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
// Dynamic Color Palette for Arbitrary Class Counts
// ============================================================================

cv::Scalar get_palette_color(int class_id) {
  // Canonical palette for FS Driverless classes 0..3 + distinct palette for generic classes
  static const std::vector<cv::Scalar> palette = {
      cv::Scalar(0, 255, 255),   // Class 0: Yellow (BGR)
      cv::Scalar(255, 0, 0),     // Class 1: Blue
      cv::Scalar(0, 140, 255),   // Class 2: SmallOrange
      cv::Scalar(0, 69, 255),    // Class 3: LargeOrange
      cv::Scalar(0, 255, 0),     // Class 4: Green
      cv::Scalar(255, 0, 255),   // Class 5: Magenta
      cv::Scalar(255, 255, 0),   // Class 6: Cyan
      cv::Scalar(180, 105, 255), // Class 7: Hot Pink
      cv::Scalar(255, 191, 0),   // Class 8: Deep Sky Blue
      cv::Scalar(50, 205, 50),   // Class 9: Lime Green
      cv::Scalar(203, 192, 255), // Class 10: Light Pink
      cv::Scalar(0, 215, 255),   // Class 11: Gold
      cv::Scalar(238, 130, 238), // Class 12: Violet
      cv::Scalar(75, 0, 130),    // Class 13: Indigo
      cv::Scalar(128, 128, 0),   // Class 14: Teal
      cv::Scalar(0, 128, 128),   // Class 15: Olive
  };
  if (class_id >= 0 && class_id < static_cast<int>(palette.size())) {
    return palette[class_id];
  }
  // Deterministic color hashing for arbitrary C >= 16 classes
  uint32_t hash = static_cast<uint32_t>(class_id) * 2654435761u;
  uint8_t r = static_cast<uint8_t>((hash & 0xFF) | 0x40);
  uint8_t g = static_cast<uint8_t>(((hash >> 8) & 0xFF) | 0x40);
  uint8_t b = static_cast<uint8_t>(((hash >> 16) & 0xFF) | 0x40);
  return cv::Scalar(b, g, r);
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char **argv) {
  // Setup signal handlers for graceful shutdown
  setup_signal_handlers();

  if (argc < 3) {
    std::cerr
        << "Usage: ./unina_perception <engine_path> <image_path> [config_path]"
        << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string image_path = argv[2];
  std::string config_path = (argc > 3) ? argv[3] : "config.yaml";

  // ========== Load Configuration ==========
  unina::Config config;
  float conf_threshold = 0.25f;
  float score_threshold = 0.45f;
  float nms_iou_threshold = 0.60f;
  int max_detections = unina::kMaxDetections;
  bool use_nms = false;

  if (config.load(config_path)) {
    std::cout << "Loaded config from: " << config_path << std::endl;
    conf_threshold = config.get_float("conf_threshold", 0.25f);
    score_threshold = config.get_float("score_threshold", 0.45f);
    nms_iou_threshold = config.get_float("nms_iou_threshold", 0.60f);
    max_detections = std::max(1, config.get_int("max_detections", unina::kMaxDetections));
    use_nms = config.get_bool("use_nms", false);
    std::cout << "  conf_threshold: " << conf_threshold << std::endl;
    std::cout << "  use_nms: " << (use_nms ? "true" : "false") << std::endl;
    std::cout << "  score_threshold: " << score_threshold << std::endl;
    std::cout << "  nms_iou_threshold: " << nms_iou_threshold << std::endl;
    std::cout << "  max_detections: " << max_detections << std::endl;
  } else {
    std::cout << "Config file not found, using defaults" << std::endl;
  }

  // ========== 1. Initialize Engine ==========
  unina::UninaPerception runner;
  std::cout << "Initializing Engine..." << std::endl;
  if (!runner.init(engine_path)) {
    std::cerr << "Failed to init engine." << std::endl;
    return -1;
  }

  // ========== 2. Initialize CUDA Resources ==========
  unina::cuda_preprocess_init(4096 * 4096 * 3);
  unina::cuda_postprocess_init();

  // Use the runner's internal CUDA stream for all operations
  cudaStream_t stream = runner.get_stream();

  // ========== 3. Load Image (CPU) ==========
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Failed to load image." << std::endl;
    unina::cuda_preprocess_destroy();
    unina::cuda_postprocess_destroy();
    return -1;
  }
  std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

  // ========== 4. Preprocess (CUDA) ==========
  void *trt_input = runner.get_input_gpu_ptr();
  float scale;

  if (runner.input_is_fp16()) {
    // FP16 input for DLA
    scale = unina::cuda_preprocess_fp16(img.data, img.cols, img.rows, trt_input,
                                        unina::kInputW, unina::kInputH, false,
                                        stream);
  } else {
    // FP32 input
    scale = unina::cuda_preprocess(
        img.data, img.cols, img.rows, static_cast<float *>(trt_input),
        unina::kInputW, unina::kInputH, false, stream);
  }

  // ========== 5. Inference (DLA) ==========
  std::cout << "Running Inference..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  runner.inference();

  // ========== 6. Postprocess (CUDA) ==========
  auto output_ptrs = runner.get_output_gpu_ptrs();
  int num_scales = runner.get_num_scales();
  std::vector<const void *> reg_buffers(num_scales);
  std::vector<const void *> cls_buffers(num_scales);
  for (int i = 0; i < num_scales; ++i) {
    reg_buffers[i] = output_ptrs[i];
    cls_buffers[i] = output_ptrs[i + num_scales];
  }

  auto detections =
      unina::cuda_decode(cls_buffers.data(), reg_buffers.data(), num_scales,
                         runner.get_strides().data(), unina::kInputW,
                         unina::kInputH, conf_threshold, runner.get_num_classes(),
                         use_nms, score_threshold, nms_iou_threshold,
                         max_detections, runner.output_is_fp16(), stream);

  runner.synchronize();

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count() /
              1000.0;
  std::cout << "Total Latency (Pre+Inf+Post): " << ms << " ms" << std::endl;
  std::cout << "Detections: " << detections.size() << std::endl;

  // ========== 7. Coordinate Adjustment (CPU) ==========
  int x_off, y_off;
  unina::cuda_preprocess_get_offsets(x_off, y_off);
  unina::adjust_detections_for_letterbox(detections, x_off, y_off, scale,
                                         img.cols, img.rows);

  // ========== 8. Visualize (CPU) ==========
  // Get class names from config if available
  auto class_names = config.get_array("class_names");

  for (const auto &det : detections) {
    cv::Scalar color = get_palette_color(det.class_id);

    cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2),
                  color, 2);

    // Draw label
    std::string class_str;
    if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names.size())) {
      class_str = class_names[det.class_id];
    } else {
      class_str = "Class " + std::to_string(det.class_id);
    }
    std::string label = class_str + " " +
                        std::to_string(det.confidence).substr(0, 4);
    cv::putText(img, label, cv::Point(det.x1, std::max(12, det.y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }

  cv::imwrite("output.jpg", img);
  std::cout << "Saved output.jpg" << std::endl;

  // ========== Cleanup ==========
  unina::cuda_preprocess_destroy();
  unina::cuda_postprocess_destroy();

  return 0;
}
