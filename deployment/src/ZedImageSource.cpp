/**
 * @file ZedImageSource.cpp
 * @brief ZED camera image source implementation
 */

#ifdef USE_ZED_SDK

#include "ZedImageSource.hpp"
#include <iostream>

namespace unina {

bool ZedImageSource::init(const ZedConfig &config) {
  if (is_open_) {
    std::cerr << "ZedImageSource: Already open" << std::endl;
    return false;
  }

  sl::InitParameters init_params;

  // Resolution (ZED X supports HD1200, HD1080, SVGA)
  if (config.resolution == "HD1200")
    init_params.camera_resolution = sl::RESOLUTION::HD1200;
  else if (config.resolution == "HD1080")
    init_params.camera_resolution = sl::RESOLUTION::HD1080;
  else if (config.resolution == "SVGA")
    init_params.camera_resolution = sl::RESOLUTION::SVGA;
  else
    init_params.camera_resolution = sl::RESOLUTION::SVGA; // Default fallback

  init_params.camera_fps = config.fps;
  init_params.coordinate_units = sl::UNIT::MILLIMETER;
  // ROS Standard: X = Forward, Y = Left, Z = Up
  init_params.coordinate_system =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

  // Depth mode
  depth_enabled_ = config.enable_depth;
  if (depth_enabled_) {
    if (config.depth_mode == "ULTRA")
      init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    else if (config.depth_mode == "QUALITY")
      init_params.depth_mode = sl::DEPTH_MODE::QUALITY;
    else if (config.depth_mode == "NEURAL")
      init_params.depth_mode = sl::DEPTH_MODE::NEURAL;
    else
      init_params.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
  } else {
    init_params.depth_mode = sl::DEPTH_MODE::NONE;
  }

  init_params.sdk_verbose = 1;

  auto err = zed_.open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
    std::cerr << "ZedImageSource: Failed to open camera: " << sl::toString(err)
              << std::endl;
    return false;
  }

  auto info = zed_.getCameraInformation();
  width_ = info.camera_configuration.resolution.width;
  height_ = info.camera_configuration.resolution.height;

  // Initialize image container
  zed_image_ = sl::Mat(width_, height_, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);

  // Enable object detection if depth is on
  if (depth_enabled_) {
    // WARNING: Enabling Positional Tracking is required for robust Object
    // Tracking ID persistence over time, especially when the camera is moving
    // (SLAM). This consumes additional compute resources.
    auto pos_err = zed_.enablePositionalTracking();
    if (pos_err != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "ZedImageSource: Failed to enable positional tracking: "
                << sl::toString(pos_err) << std::endl;
      // We proceed even if it fails, as basic object detection might still
      // work, but tracking IDs will be unstable.
    }

    sl::ObjectDetectionParameters od_params;
    od_params.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    od_params.enable_tracking = true;
    od_params.enable_segmentation = false;

    err = zed_.enableObjectDetection(od_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "ZedImageSource: Failed to enable object detection: "
                << sl::toString(err) << std::endl;
      // Continue without 3D - not fatal
      depth_enabled_ = false;
    }
  }

  is_open_ = true;
  std::cout << "ZedImageSource: Opened " << width_ << "x" << height_ << " @ "
            << config.fps << "FPS"
            << " (depth=" << (depth_enabled_ ? "ON" : "OFF") << ")"
            << std::endl;
  return true;
}

bool ZedImageSource::grab(ImageFrame &frame) {
  if (!is_open_) {
    frame.is_valid = false;
    return false;
  }

  if (zed_.grab() != sl::ERROR_CODE::SUCCESS) {
    frame.is_valid = false;
    return false;
  }

  zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::GPU);

  frame.width = width_;
  frame.height = height_;
  frame.channels = 4;
  frame.gpu_data = zed_image_.getPtr<uint8_t>(sl::MEM::GPU);
  frame.cpu_data = nullptr;
  frame.data_on_gpu = true;
  frame.timestamp_ns =
      zed_.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
  frame.frame_id = frame_id_++;
  frame.is_valid = true;

  return true;
}

std::vector<Cone3D>
ZedImageSource::fuse_3d(const std::vector<Detection> &detections) {
  std::vector<Cone3D> cones;

  if (!depth_enabled_ || detections.empty()) {
    // Convert to Cone3D without 3D info
    cones.reserve(detections.size());
    for (const auto &det : detections) {
      Cone3D cone;
      cone.color = static_cast<ConeColor>(det.class_id);
      cone.confidence = det.confidence;
      cone.bbox_x1 = static_cast<int>(det.x1);
      cone.bbox_y1 = static_cast<int>(det.y1);
      cone.bbox_x2 = static_cast<int>(det.x2);
      cone.bbox_y2 = static_cast<int>(det.y2);
      cone.is_valid = false; // No 3D
      cones.push_back(cone);
    }
    return cones;
  }

  // Convert detections to ZED format
  std::vector<sl::CustomBoxObjectData> zed_boxes;
  zed_boxes.reserve(detections.size());

  for (const auto &det : detections) {
    sl::CustomBoxObjectData box;
    box.unique_object_id = sl::generate_unique_id();
    box.bounding_box_2d[0] = sl::uint2(static_cast<unsigned int>(det.x1),
                                       static_cast<unsigned int>(det.y1));
    box.bounding_box_2d[1] = sl::uint2(static_cast<unsigned int>(det.x2),
                                       static_cast<unsigned int>(det.y1));
    box.bounding_box_2d[2] = sl::uint2(static_cast<unsigned int>(det.x2),
                                       static_cast<unsigned int>(det.y2));
    box.bounding_box_2d[3] = sl::uint2(static_cast<unsigned int>(det.x1),
                                       static_cast<unsigned int>(det.y2));
    box.label = det.class_id;
    box.probability = det.confidence;
    box.is_grounded = true;
    zed_boxes.push_back(box);
  }

  // Ingest and retrieve
  zed_.ingestCustomBoxObjects(zed_boxes);
  zed_.retrieveObjects(objects_, od_runtime_);

  // Convert ZED objects to Cone3D
  cones.reserve(objects_.object_list.size());
  for (const auto &obj : objects_.object_list) {
    Cone3D cone;

    // Position (mm -> m)
    cone.x = obj.position.x / 1000.0f;
    cone.y = obj.position.y / 1000.0f;
    cone.z = obj.position.z / 1000.0f;

    cone.color = static_cast<ConeColor>(obj.label);
    cone.confidence = obj.confidence;
    cone.track_id = obj.id;

    // 2D bounding box
    if (obj.bounding_box_2d.size() >= 4) {
      cone.bbox_x1 = static_cast<int>(obj.bounding_box_2d[0].x);
      cone.bbox_y1 = static_cast<int>(obj.bounding_box_2d[0].y);
      cone.bbox_x2 = static_cast<int>(obj.bounding_box_2d[2].x);
      cone.bbox_y2 = static_cast<int>(obj.bounding_box_2d[2].y);
    }

    // Check if 3D is valid (not NaN or zero)
    cone.is_valid = (std::isfinite(cone.x) && std::isfinite(cone.y) &&
                     std::isfinite(cone.z) && cone.z > 0.1f);

    cones.push_back(cone);
  }

  return cones;
}

void ZedImageSource::close() {
  if (is_open_) {
    if (depth_enabled_) {
      zed_.disableObjectDetection();
    }
    zed_.close();
    is_open_ = false;
    std::cout << "ZedImageSource: Closed" << std::endl;
  }
}

} // namespace unina

#endif // USE_ZED_SDK
