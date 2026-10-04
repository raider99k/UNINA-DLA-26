#pragma once

/**
 * @file Perception3D.hpp
 * @brief Main 3D perception pipeline orchestrator
 *
 * This class encapsulates the entire perception pipeline:
 * 1. Get image from source (ZED, file, etc.)
 * 2. Preprocess on GPU (CUDA)
 * 3. Infer on DLA (TensorRT)
 * 4. Decode 2D detections (CUDA)
 * 5. Optional: Fuse with depth for 3D (ZED SDK)
 *
 * Usage:
 *   unina::Perception3D perception;
 *   perception.init(config);
 *   while (running) {
 *     auto result = perception.process(image_source);
 *     // Use result.cones
 *   }
 */

#include "Cone3D.hpp"
#include "Config.hpp"
#include "CudaPostprocess.cuh"
#include "CudaPreprocess.cuh"
#include "IImageSource.hpp"
#include "UninaPerception.hpp"
#include "common.hpp"

#include <memory>
#include <string>
#include <vector>

namespace unina {

/**
 * @brief Configuration for Perception3D
 */
struct Perception3DConfig {
  // Engine
  std::string engine_path;

  // Model input dimensions (must match TensorRT engine)
  int input_w = 640; ///< From config.yaml
  int input_h = 640; ///< From config.yaml

  // Detection thresholds
  float conf_threshold = 0.25f;
  float score_threshold = 0.45f;
  float nms_iou_threshold = 0.60f;
  int max_detections = 1000;
  bool use_nms = false;

  // 3D options
  bool enable_3d = false; ///< If false, only 2D detection (no depth)

  // Load from config file
  static Perception3DConfig from_file(const std::string &path);
};

/**
 * @brief Main 3D perception pipeline
 *
 * This class is the core of the modular architecture.
 * It handles all GPU/DLA operations internally and exposes a clean API.
 */
class Perception3D {
public:
  Perception3D() = default;
  ~Perception3D();

  // Non-copyable
  Perception3D(const Perception3D &) = delete;
  Perception3D &operator=(const Perception3D &) = delete;

  /**
   * @brief Initialize the perception pipeline
   * @param config Pipeline configuration
   * @return true on success
   */
  bool init(const Perception3DConfig &config);

  /**
   * @brief Process a single frame (2D detection only)
   *
   * Use this when you don't have depth or don't need 3D.
   *
   * @param frame Input image frame
   * @return Frame result with 2D detections (Cone3D with x,y,z = 0)
   */
  FrameResult process_2d(const ImageFrame &frame);

  /**
   * @brief Get raw 2D detections (for ZED 3D fusion)
   *
   * Use this when you want to feed detections to ZED's ingestCustomBoxObjects.
   *
   * @param frame Input image frame
   * @return Vector of 2D detections
   */
  std::vector<Detection> detect_2d(const ImageFrame &frame);

  /**
   * @brief Get letterbox scale from last preprocessing
   */
  float get_scale() const { return last_scale_; }

  /**
   * @brief Get letterbox offsets from last preprocessing
   */
  void get_offsets(int &x_offset, int &y_offset) const;

  /**
   * @brief Check if initialized
   */
  bool is_initialized() const { return initialized_; }

  /**
   * @brief Get TensorRT runner (for advanced use)
   */
  UninaPerception &runner() { return runner_; }

private:
  bool initialized_ = false;
  Perception3DConfig config_;

  // TensorRT inference
  UninaPerception runner_;

  // Processing state
  float last_scale_ = 1.0f;
  int frame_counter_ = 0;
};

} // namespace unina
