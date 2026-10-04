/**
 * @file Perception3D.cpp
 * @brief Implementation of the 3D perception pipeline
 */

#include "Perception3D.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace unina {

// ============================================================================
// Static Helper
// ============================================================================

Perception3DConfig Perception3DConfig::from_file(const std::string &path) {
  Perception3DConfig cfg;
  Config file_config;

  if (file_config.load(path)) {
    cfg.conf_threshold = file_config.get_float("conf_threshold", 0.25f);
    cfg.score_threshold = file_config.get_float("score_threshold", 0.45f);
    cfg.nms_iou_threshold = file_config.get_float("nms_iou_threshold", 0.60f);
    cfg.max_detections = std::max(1, file_config.get_int("max_detections", 1000));
    cfg.use_nms = file_config.get_bool("use_nms", false);
    cfg.enable_3d = file_config.get_bool("zed_depth_enabled", false);
    cfg.input_w = file_config.get_int("input_width", 640);
    cfg.input_h = file_config.get_int("input_height", 640);
  }

  return cfg;
}

// ============================================================================
// Perception3D Implementation
// ============================================================================

Perception3D::~Perception3D() {
  if (initialized_) {
    cuda_preprocess_destroy();
    cuda_postprocess_destroy();
  }
}

bool Perception3D::init(const Perception3DConfig &config) {
  if (initialized_) {
    std::cerr << "Perception3D already initialized" << std::endl;
    return false;
  }

  config_ = config;

  // Initialize TensorRT engine
  std::cout << "Perception3D: Loading engine..." << std::endl;
  if (!runner_.init(config.engine_path)) {
    std::cerr << "Perception3D: Failed to load engine" << std::endl;
    return false;
  }

  // Initialize CUDA kernels
  // Buffer size: 4K resolution * 4 channels (BGRA)
  cuda_preprocess_init(4096 * 4096 * 4);
  cuda_postprocess_init();

  initialized_ = true;
  std::cout << "Perception3D: Initialized successfully" << std::endl;
  std::cout << "Perception3D: NMS "
            << (config_.use_nms ? "ENABLED" : "DISABLED (pure O2O decode)")
            << " | score_thres=" << config_.score_threshold
            << " | iou_thres=" << config_.nms_iou_threshold
            << " | max_det=" << config_.max_detections << std::endl;
  return true;
}

std::vector<Detection> Perception3D::detect_2d(const ImageFrame &frame) {
  if (!initialized_ || !frame.is_valid) {
    return {};
  }

  cudaStream_t stream = runner_.get_stream();
  void *trt_input = runner_.get_input_gpu_ptr();

  // Preprocess based on input format
  if (frame.channels == 4) {
    // BGRA (ZED format)
    if (frame.data_on_gpu) {
      if (runner_.input_is_fp16()) {
        // HWC4 format: DLA-native input, eliminates CHW transpose
        last_scale_ = cuda_preprocess_bgra_hwc4_fp16(
            frame.gpu_data, frame.width, frame.height, trt_input,
            config_.input_w, config_.input_h, stream);
      } else {
        last_scale_ =
            cuda_preprocess_bgra(frame.gpu_data, frame.width, frame.height,
                                 static_cast<float *>(trt_input),
                                 config_.input_w, config_.input_h, stream);
      }
    } else {
      // CPU BGRA - not directly supported, would need upload
      std::cerr << "Perception3D: CPU BGRA not supported" << std::endl;
      return {};
    }
  } else {
    // BGR (OpenCV format)
    if (runner_.input_is_fp16()) {
      last_scale_ = cuda_preprocess_fp16(
          frame.data(), frame.width, frame.height, trt_input, config_.input_w,
          config_.input_h, frame.data_on_gpu, stream);
    } else {
      last_scale_ =
          cuda_preprocess(frame.data(), frame.width, frame.height,
                          static_cast<float *>(trt_input), config_.input_w,
                          config_.input_h, frame.data_on_gpu, stream);
    }
  }

  // Inference
  runner_.inference();
  runner_.synchronize();

  // Postprocess
  auto output_buffers = runner_.get_output_gpu_ptrs();
  const auto &output_layouts = runner_.get_output_layouts();
  int num_scales = runner_.get_num_scales();

  // DYNAMIC BUFFER MATCHING: Prepare reg and cls pointers for decoder
  // We assume UninaPerception::init sorted regX first, then clsX in
  // all_out_names
  std::vector<const void *> reg_buffers(num_scales);
  std::vector<const void *> cls_buffers(num_scales);
  std::vector<OutputTensorLayout> reg_layouts(num_scales);
  std::vector<OutputTensorLayout> cls_layouts(num_scales);

  for (int i = 0; i < num_scales; ++i) {
    reg_buffers[i] = output_buffers[i];
    cls_buffers[i] = output_buffers[i + num_scales];
    reg_layouts[i] = output_layouts[i];
    cls_layouts[i] = output_layouts[i + num_scales];
  }

  auto detections =
      cuda_decode(cls_buffers.data(), reg_buffers.data(), cls_layouts.data(),
                  reg_layouts.data(), num_scales, runner_.get_strides().data(),
                  config_.input_w, config_.input_h, config_.conf_threshold,
                  runner_.get_num_classes(), config_.use_nms,
                  config_.score_threshold, config_.nms_iou_threshold,
                  config_.max_detections, runner_.output_is_fp16(), stream);

  // Adjust for letterbox
  int x_off, y_off;
  cuda_preprocess_get_offsets(x_off, y_off);
  adjust_detections_for_letterbox(detections, x_off, y_off, last_scale_,
                                  frame.width, frame.height);

  frame_counter_++;
  return detections;
}

FrameResult Perception3D::process_2d(const ImageFrame &frame) {
  FrameResult result;
  result.frame_id = frame.frame_id;
  result.timestamp_ns = frame.timestamp_ns;

  auto t0 = std::chrono::high_resolution_clock::now();

  auto detections = detect_2d(frame);

  auto t1 = std::chrono::high_resolution_clock::now();
  result.latency_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Convert detections to Cone3D (2D only)
  result.cones.reserve(detections.size());
  for (const auto &det : detections) {
    Cone3D cone;
    cone.color = static_cast<ConeColor>(det.class_id);
    cone.confidence = det.confidence;
    cone.bbox_x1 = static_cast<int>(det.x1);
    cone.bbox_y1 = static_cast<int>(det.y1);
    cone.bbox_x2 = static_cast<int>(det.x2);
    cone.bbox_y2 = static_cast<int>(det.y2);
    cone.is_valid = true; // 2D is always valid
    // x, y, z remain 0 (no 3D)
    result.cones.push_back(cone);
  }

  return result;
}

void Perception3D::get_offsets(int &x_offset, int &y_offset) const {
  cuda_preprocess_get_offsets(x_offset, y_offset);
}

} // namespace unina
