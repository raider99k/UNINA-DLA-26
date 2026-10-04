#pragma once

/**
 * @file ZedImageSource.hpp
 * @brief ZED camera implementation of IImageSource
 *
 * This provides live camera input from ZED X for real-time perception.
 * Also handles the 3D fusion via ZED SDK's Custom Object Detection.
 */

#ifdef USE_ZED_SDK

#include "Cone3D.hpp"
#include "IImageSource.hpp"
#include "common.hpp"

#include <sl/Camera.hpp>
#include <vector>

namespace unina {

/**
 * @brief ZED camera configuration
 */
struct ZedConfig {
  std::string resolution = "HD720"; // HD720, HD1080, HD2K, VGA
  int fps = 30;
  bool enable_depth = true;
  std::string depth_mode = "PERFORMANCE"; // PERFORMANCE, QUALITY, ULTRA, NEURAL
};

/**
 * @brief ZED X camera image source with 3D fusion capability
 */
class ZedImageSource : public IImageSource {
public:
  ZedImageSource() = default;
  ~ZedImageSource() override { close(); }

  /**
   * @brief Initialize with configuration
   */
  bool init(const ZedConfig &config);

  // IImageSource interface
  bool init() override { return init(ZedConfig()); }
  bool grab(ImageFrame &frame) override;
  bool has_more() const override { return is_open_; }
  int width() const override { return width_; }
  int height() const override { return height_; }
  int channels() const override { return 4; } // BGRA
  bool provides_gpu_data() const override { return true; }
  void close() override;

  // === 3D Fusion (ZED-specific) ===

  /**
   * @brief Check if depth/3D is enabled
   */
  bool depth_enabled() const { return depth_enabled_; }

  /**
   * @brief Fuse 2D detections with depth to get 3D positions
   *
   * This uses ZED SDK's Custom Object Detection API.
   *
   * @param detections 2D bounding box detections from YOLO
   * @return Vector of 3D cone positions
   */
  std::vector<Cone3D> fuse_3d(const std::vector<Detection> &detections);

  /**
   * @brief Get raw ZED camera handle (for advanced use)
   */
  sl::Camera &camera() { return zed_; }

private:
  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Objects objects_;
  sl::ObjectDetectionRuntimeParameters od_runtime_;

  bool is_open_ = false;
  bool depth_enabled_ = false;
  int width_ = 0;
  int height_ = 0;
  int frame_id_ = 0;
};

} // namespace unina

#endif // USE_ZED_SDK
