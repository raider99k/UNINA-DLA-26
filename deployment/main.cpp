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

cv::Scalar get_palette_color(int class_id) {
  static const std::vector<cv::Scalar> palette = {
      cv::Scalar(0, 255, 255),   // Class 0
      cv::Scalar(255, 0, 0),     // Class 1
      cv::Scalar(0, 140, 255),   // Class 2
      cv::Scalar(0, 69, 255),    // Class 3
      cv::Scalar(0, 255, 0),     // Class 4
      cv::Scalar(255, 0, 255),   // Class 5
      cv::Scalar(255, 255, 0),   // Class 6
      cv::Scalar(180, 105, 255), // Class 7
      cv::Scalar(255, 191, 0),   // Class 8
      cv::Scalar(50, 205, 50),   // Class 9
      cv::Scalar(203, 192, 255), // Class 10
      cv::Scalar(0, 215, 255),   // Class 11
      cv::Scalar(238, 130, 238), // Class 12
      cv::Scalar(75, 0, 130),    // Class 13
      cv::Scalar(128, 128, 0),   // Class 14
      cv::Scalar(0, 128, 128),   // Class 15
  };
  if (class_id >= 0 && class_id < static_cast<int>(palette.size())) {
    return palette[class_id];
  }
  uint32_t hash = static_cast<uint32_t>(class_id) * 2654435761u;
  uint8_t r = static_cast<uint8_t>((hash & 0xFF) | 0x40);
  uint8_t g = static_cast<uint8_t>(((hash >> 8) & 0xFF) | 0x40);
  uint8_t b = static_cast<uint8_t>(((hash >> 16) & 0xFF) | 0x40);
  return cv::Scalar(b, g, r);
}

int main(int argc, char **argv) {
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

  unina::UninaPerception runner;
  std::cout << "Initializing Engine..." << std::endl;
  if (!runner.init(engine_path)) {
    std::cerr << "Failed to init engine." << std::endl;
    return -1;
  }

  int input_w = runner.get_input_width() > 0
                    ? runner.get_input_width()
                    : config.get_int("input_width", unina::kInputW);
  int input_h = runner.get_input_height() > 0
                    ? runner.get_input_height()
                    : config.get_int("input_height", unina::kInputH);

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Failed to load image." << std::endl;
    return -1;
  }
  std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

  size_t src_bytes = static_cast<size_t>(img.cols) *
                     static_cast<size_t>(img.rows) * 3;
  if (unina::cuda_preprocess_init(src_bytes) != cudaSuccess ||
      unina::cuda_postprocess_init() != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA buffers." << std::endl;
    unina::cuda_preprocess_destroy();
    unina::cuda_postprocess_destroy();
    return -1;
  }

  cudaStream_t stream = runner.get_stream();

  void *trt_input = runner.get_input_gpu_ptr();
  float scale;

  if (runner.input_is_fp16()) {
    scale = unina::cuda_preprocess_fp16(img.data, img.cols, img.rows, trt_input,
                                        input_w, input_h, false, stream);
  } else {
    scale = unina::cuda_preprocess(
        img.data, img.cols, img.rows, static_cast<float *>(trt_input),
        input_w, input_h, false, stream);
  }
  if (scale <= 0.0f) {
    std::cerr << "CUDA preprocess failed." << std::endl;
    unina::cuda_preprocess_destroy();
    unina::cuda_postprocess_destroy();
    return -1;
  }

  std::cout << "Running Inference..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  runner.inference();

  auto output_ptrs = runner.get_output_gpu_ptrs();
  const auto &output_layouts = runner.get_output_layouts();
  int num_scales = runner.get_num_scales();
  std::vector<const void *> reg_buffers(num_scales);
  std::vector<const void *> cls_buffers(num_scales);
  std::vector<unina::OutputTensorLayout> reg_layouts(num_scales);
  std::vector<unina::OutputTensorLayout> cls_layouts(num_scales);
  for (int i = 0; i < num_scales; ++i) {
    reg_buffers[i] = output_ptrs[i];
    cls_buffers[i] = output_ptrs[i + num_scales];
    reg_layouts[i] = output_layouts[i];
    cls_layouts[i] = output_layouts[i + num_scales];
  }

  auto detections =
      unina::cuda_decode(cls_buffers.data(), reg_buffers.data(),
                         cls_layouts.data(), reg_layouts.data(), num_scales,
                         runner.get_strides().data(), input_w, input_h,
                         conf_threshold, runner.get_num_classes(), use_nms,
                         score_threshold, nms_iou_threshold, max_detections,
                         runner.output_is_fp16(), stream);

  runner.synchronize();

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count() /
              1000.0;
  std::cout << "Total Latency (Pre+Inf+Post): " << ms << " ms" << std::endl;
  std::cout << "Detections: " << detections.size() << std::endl;

  int x_off, y_off;
  unina::cuda_preprocess_get_offsets(x_off, y_off);
  unina::adjust_detections_for_letterbox(detections, x_off, y_off, scale,
                                         img.cols, img.rows);

  auto class_names = config.get_array("class_names");

  for (const auto &det : detections) {
    cv::Scalar color = get_palette_color(det.class_id);

    cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2),
                  color, 2);

    std::string class_str;
    if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names.size())) {
      class_str = class_names[det.class_id];
    } else {
      class_str = "Class " + std::to_string(det.class_id);
    }
    std::string label = class_str + " " +
                        std::to_string(det.confidence).substr(0, 4);
    cv::putText(img, label, cv::Point(det.x1, std::max(12, static_cast<int>(det.y1) - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }

  cv::imwrite("output.jpg", img);
  std::cout << "Saved output.jpg" << std::endl;

  unina::cuda_preprocess_destroy();
  unina::cuda_postprocess_destroy();

  return 0;
}
