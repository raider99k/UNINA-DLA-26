/**
 * @file main_offline.cpp
 * @brief Offline 2D Perception Testing (Image Files, Directories, and Videos)
 *
 * - Single image: ./unina_perception_offline engine.engine image.jpg
 * - Directory:    ./unina_perception_offline engine.engine ./frames/ --dir
 * - Video:        ./unina_perception_offline engine.engine video.mp4
 */

#include "Config.hpp"
#include "CudaPostprocess.cuh"
#include "CudaPreprocess.cuh"
#include "FileImageSource.hpp"
#include "UninaPerception.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

static volatile bool g_running = true;

void signal_handler(int signum) {
  std::cerr << "\nReceived signal " << signum << ", stopping..." << std::endl;
  g_running = false;
}

static cv::Scalar get_palette_color(int class_id) {
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
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  if (argc < 3) {
    std::cerr << "Usage: ./unina_perception_offline <engine_path> <input_path> "
                 "[--dir] [--output output_dir] [--config config.yaml]"
              << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  ./unina_perception_offline model.engine test.jpg"
              << std::endl;
    std::cerr << "  ./unina_perception_offline model.engine ./frames/ --dir"
              << std::endl;
    std::cerr << "  ./unina_perception_offline model.engine video.mp4 --output "
                 "./results/"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string input_path = argv[2];
  bool is_directory = false;
  std::string output_dir;
  std::string config_path = "config.yaml";

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--dir") {
      is_directory = true;
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    }
  }

  unina::Config config;
  config.load(config_path);

  float conf_threshold = config.get_float("conf_threshold", 0.25f);
  float score_threshold = config.get_float("score_threshold", 0.45f);
  bool use_nms = config.get_bool("use_nms", false);
  float nms_iou_threshold = config.get_float("nms_iou_threshold", 0.60f);
  int max_detections = config.get_int("max_detections", 1000);
  int config_w = config.get_int("input_width", unina::kInputW);
  int config_h = config.get_int("input_height", unina::kInputH);
  auto class_names = config.get_array("class_names");

  unina::FileImageSource source(input_path, is_directory);
  if (!source.init()) {
    std::cerr << "Failed to initialize image source" << std::endl;
    return -1;
  }

  unina::UninaPerception runner;
  if (!runner.init(engine_path)) {
    std::cerr << "Failed to initialize TensorRT engine." << std::endl;
    return -1;
  }

  int input_w = runner.get_input_width() > 0 ? runner.get_input_width() : config_w;
  int input_h = runner.get_input_height() > 0 ? runner.get_input_height() : config_h;

  size_t src_buf_bytes = 4096ULL * 4096ULL * 3;
  if (unina::cuda_preprocess_init(src_buf_bytes) != cudaSuccess ||
      unina::cuda_postprocess_init() != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA buffers." << std::endl;
    unina::cuda_preprocess_destroy();
    unina::cuda_postprocess_destroy();
    return -1;
  }
  cudaStream_t stream = runner.get_stream();

  std::cout << "\n========================================" << std::endl;
  std::cout << "Processing " << source.total_frames() << " frames..."
            << std::endl;
  std::cout << "========================================\n" << std::endl;

  unina::ImageFrame frame;
  double total_latency_ms = 0;
  int total_detections = 0;
  int frame_count = 0;

  std::ofstream csv_out;
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
    csv_out.open(output_dir + "/detections.csv");
    csv_out << "frame_id,class_id,class_name,x1,y1,x2,y2,confidence\n";
  }

  while (g_running && source.has_more()) {
    if (!source.grab(frame)) {
      break;
    }

    cv::Mat vis_image(frame.height, frame.width, CV_8UC3,
                      const_cast<uint8_t *>(frame.cpu_data));

    size_t needed_bytes = static_cast<size_t>(frame.width) *
                          static_cast<size_t>(frame.height) * 3;
    if (needed_bytes > src_buf_bytes) {
      if (unina::cuda_preprocess_init(needed_bytes) != cudaSuccess) {
        std::cerr << "Failed to grow CUDA preprocess buffer to "
                  << needed_bytes << " bytes; skipping frame " << frame_count
                  << std::endl;
        frame_count++;
        continue;
      }
      src_buf_bytes = needed_bytes;
    }

    auto start = std::chrono::high_resolution_clock::now();

    void *trt_input = runner.get_input_gpu_ptr();
    float scale = 1.0f;
    if (runner.input_is_fp16()) {
      scale = unina::cuda_preprocess_fp16(frame.cpu_data, frame.width, frame.height,
                                          trt_input, input_w, input_h, false, stream);
    } else {
      scale = unina::cuda_preprocess(frame.cpu_data, frame.width, frame.height,
                                     static_cast<float *>(trt_input), input_w,
                                     input_h, false, stream);
    }
    if (scale <= 0.0f) {
      std::cerr << "CUDA preprocess failed on frame " << frame_count
                << "; skipping." << std::endl;
      frame_count++;
      continue;
    }

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

    auto detections = unina::cuda_decode(
        cls_buffers.data(), reg_buffers.data(), cls_layouts.data(),
        reg_layouts.data(), num_scales, runner.get_strides().data(), input_w,
        input_h, conf_threshold, runner.get_num_classes(), use_nms,
        score_threshold, nms_iou_threshold, max_detections,
        runner.output_is_fp16(), stream);

    runner.synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;

    int x_off = 0, y_off = 0;
    unina::cuda_preprocess_get_offsets(x_off, y_off);
    unina::adjust_detections_for_letterbox(detections, x_off, y_off, scale,
                                           frame.width, frame.height);

    total_latency_ms += latency_ms;
    total_detections += static_cast<int>(detections.size());

    for (const auto &det : detections) {
      cv::Scalar color = get_palette_color(det.class_id);
      cv::rectangle(vis_image, cv::Point(det.x1, det.y1),
                    cv::Point(det.x2, det.y2), color, 2);

      std::string class_str;
      if (det.class_id >= 0 &&
          det.class_id < static_cast<int>(class_names.size())) {
        class_str = class_names[det.class_id];
      } else {
        class_str = "class_" + std::to_string(det.class_id);
      }

      std::string label =
          class_str + " " + std::to_string(det.confidence).substr(0, 4);
      cv::putText(vis_image, label,
                  cv::Point(det.x1, std::max(12, static_cast<int>(det.y1) - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

      if (csv_out.is_open()) {
        csv_out << frame_count << "," << det.class_id << "," << class_str << ","
                << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2
                << "," << det.confidence << "\n";
      }
    }

    if (!output_dir.empty()) {
      std::string out_path =
          output_dir + "/frame_" + std::to_string(frame_count) + ".jpg";
      cv::imwrite(out_path, vis_image);
    } else if (source.total_frames() == 1) {
      cv::imwrite("output.jpg", vis_image);
      std::cout << "Saved output.jpg" << std::endl;
    }

    if (frame_count % 10 == 0 || source.total_frames() == 1) {
      std::cout << "Frame " << frame_count << "/" << source.total_frames()
                << " | Detections: " << detections.size()
                << " | Latency: " << latency_ms << "ms" << std::endl;
    }

    frame_count++;
  }

  unina::cuda_preprocess_destroy();
  unina::cuda_postprocess_destroy();

  std::cout << "\n===== Processing Complete =====" << std::endl;
  std::cout << "Frames processed: " << frame_count << std::endl;
  std::cout << "Total detections: " << total_detections << std::endl;
  if (frame_count > 0) {
    double avg_ms = total_latency_ms / frame_count;
    std::cout << "Avg latency: " << avg_ms << " ms" << std::endl;
    std::cout << "Avg FPS: " << (1000.0 / avg_ms) << std::endl;
  }
  if (!output_dir.empty()) {
    std::cout << "Results saved to: " << output_dir << std::endl;
  }

  return 0;
}
