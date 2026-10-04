/**
 * @file main_ros.cpp
 * @brief ROS 2 Wrapper for UNINA-DLA Perception (Humble)
 *
 * This node acts as a "Bridge":
 * 1. Runs the optimized pipeline (ZED + TensorRT/DLA)
 * 2. Converts internal data (unina::Cone3D) to SLAM team's ROS format
 *
 * Build with: colcon build --symlink-install --cmake-args -DUSE_ZED=ON
 * -DUSE_ROS=ON
 */

#ifdef USE_ROS

// === 1. ROS 2 INCLUDES ===
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// === 2. UNINA INCLUDES (Internal pipeline) ===
#include "Cone3D.hpp"
#include "Config.hpp"
#include "Perception3D.hpp"
#include "ZedImageSource.hpp"

#include <csignal>

// ============================================================================
// Signal Handling
// ============================================================================

volatile sig_atomic_t g_running = 1;

void signal_handler(int signum) {
  (void)signum;
  g_running = 0;
  rclcpp::shutdown();
}

// ============================================================================
// Timestamp Conversion (SLAM synchronization)
// ============================================================================

/**
 * @brief Convert ZED timestamp (nanoseconds) to rclcpp::Time
 *
 * Note: Using this->now() introduces 30-50ms jitter that degrades
 * SLAM odometry estimation. Always use the actual frame timestamp.
 */
rclcpp::Time zed_to_ros_time(uint64_t timestamp_ns) {
  return rclcpp::Time(static_cast<int64_t>(timestamp_ns));
}

// ============================================================================
// Adapter Functions
// ============================================================================

/**
 * @brief Convert internal ConeColor enum to ROS string
 */
std::string color_to_ros_string(unina::ConeColor c) {
  switch (c) {
  case unina::ConeColor::BLUE:
    return "blue";
  case unina::ConeColor::YELLOW:
    return "yellow";
  case unina::ConeColor::BIG_ORANGE:
    return "orange_big";
  case unina::ConeColor::SMALL_ORANGE:
    return "orange_small";
  default:
    return "unknown";
  }
}

/**
 * @brief Convert internal ConeColor enum to integer ID
 */
int color_to_ros_id(unina::ConeColor c) { return static_cast<int>(c); }

geometry_msgs::msg::PointStamped to_ros_message(const unina::Cone3D &cone,
                                                const std::string &frame_id,
                                                const rclcpp::Time &stamp) {
  geometry_msgs::msg::PointStamped msg;

  // Header - Use actual frame timestamp
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  // Position (meters, ROS coordinate system: X=forward, Y=left, Z=up)
  msg.point.x = cone.x;
  msg.point.y = cone.y;
  msg.point.z = cone.z;

  return msg;
}

visualization_msgs::msg::Marker to_marker(const unina::Cone3D &cone, int id,
                                          const std::string &frame_id,
                                          const rclcpp::Time &stamp) {
  visualization_msgs::msg::Marker marker;

  // Header - Use actual frame timestamp for SLAM consistency
  marker.header.stamp = stamp;
  marker.header.frame_id = frame_id;
  marker.ns = "cones";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Position
  marker.pose.position.x = cone.x;
  marker.pose.position.y = cone.y;
  marker.pose.position.z = cone.z / 2.0; // Cylinder base at ground

  marker.pose.orientation.w = 1.0;

  // Size (typical cone dimensions)
  marker.scale.x = 0.228; // Base diameter
  marker.scale.y = 0.228;
  marker.scale.z = 0.325; // Height

  // Color based on cone type
  marker.color.a = 0.9f;
  switch (cone.color) {
  case unina::ConeColor::BLUE:
    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 1.0f;
    break;
  case unina::ConeColor::YELLOW:
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    break;
  case unina::ConeColor::BIG_ORANGE:
  case unina::ConeColor::SMALL_ORANGE:
    marker.color.r = 1.0f;
    marker.color.g = 0.5f;
    marker.color.b = 0.0f;
    break;
  default:
    marker.color.r = 0.5f;
    marker.color.g = 0.5f;
    marker.color.b = 0.5f;
  }

  marker.lifetime = rclcpp::Duration::from_seconds(0.2);

  return marker;
}

// ============================================================================
// Main Node Class
// ============================================================================

class UninaPerceptionNode : public rclcpp::Node {
public:
  UninaPerceptionNode() : Node("unina_perception_dla") {
    // Parameters
    this->declare_parameter<std::string>("engine_path", "model.engine");
    this->declare_parameter<std::string>("config_path", "config.yaml");
    this->declare_parameter<std::string>("frame_id", "zed_left_camera_frame");
    this->declare_parameter<std::string>("zed_resolution", "HD720");
    this->declare_parameter<int>("zed_fps", 30);
    this->declare_parameter<std::string>("zed_depth_mode", "PERFORMANCE");
    this->declare_parameter<double>("conf_threshold", 0.25);

    std::string engine_path = this->get_parameter("engine_path").as_string();
    std::string config_path = this->get_parameter("config_path").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();

    // Publishers
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "perception/cones_viz", 10);

    // Initialize Pipeline
    RCLCPP_INFO(this->get_logger(), "Initializing UNINA-DLA Pipeline...");

    unina::Config config;
    if (!config.load(config_path)) {
      RCLCPP_WARN(this->get_logger(),
                  "Config file not found: %s. Using defaults.",
                  config_path.c_str());
    }

    unina::ZedConfig zed_config;
    zed_config.resolution = this->get_parameter("zed_resolution").as_string();
    zed_config.fps = static_cast<int>(this->get_parameter("zed_fps").as_int());
    zed_config.enable_depth = true;
    zed_config.depth_mode = this->get_parameter("zed_depth_mode").as_string();

    if (!camera_.init(zed_config)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open ZED X camera");
      throw std::runtime_error("Camera Init Failed");
    }

    unina::Perception3DConfig perc_config;
    perc_config.engine_path = engine_path;
    perc_config.conf_threshold =
        static_cast<float>(this->get_parameter("conf_threshold").as_double());
    perc_config.score_threshold = config.get_float("score_threshold", 0.45f);
    perc_config.nms_iou_threshold = config.get_float("nms_iou_threshold", 0.60f);
    perc_config.max_detections = config.get_int("max_detections", 1000);
    if (perc_config.max_detections < 1) {
      perc_config.max_detections = 1;
    }
    perc_config.use_nms = config.get_bool("use_nms", false);
    perc_config.enable_3d = true;

    // Propagate dynamic resolution from config to pipeline
    perc_config.input_w = config.get_int("input_width", 640);
    perc_config.input_h = config.get_int("input_height", 640);

    if (!perception_.init(perc_config)) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to initialize TensorRT/DLA engine: %s",
                   engine_path.c_str());
      throw std::runtime_error("Engine Init Failed");
    }

    // Hardware acceleration check
    if (perception_.runner().is_dla_active()) {
      RCLCPP_INFO(this->get_logger(),
                  "DLA Hardware Acceleration: [ACTIVE] on Core %d",
                  perception_.runner().runner().getDLACore());
    } else {
      RCLCPP_WARN(
          this->get_logger(),
          "DLA Hardware Acceleration: [INACTIVE]. Running on GPU fallback.");
      RCLCPP_WARN(this->get_logger(),
                  "Ensure the engine was built with --allowGPUFallback=false "
                  "for strict DLA.");
    }

    RCLCPP_INFO(this->get_logger(),
                "UNINA-DLA Node Started. Zero-Overhead Pipeline Active.");
    RCLCPP_INFO(this->get_logger(), "  Engine: %s", engine_path.c_str());
    RCLCPP_INFO(this->get_logger(), "  Frame ID: %s", frame_id_.c_str());

    // Using a Timer instead of while-loop for better ROS 2 integration
    // Run nicely at requested FPS (or faster) - 1ms interval to poll
    // aggressively
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1),
        std::bind(&UninaPerceptionNode::process_frame, this));
  }

  ~UninaPerceptionNode() {
    camera_.close();
    RCLCPP_INFO(this->get_logger(), "UNINA-DLA Node Shutdown Complete.");
  }

private:
  void process_frame() {
    if (!g_running)
      return;

    unina::ImageFrame frame;
    // ZED grab() is blocking but efficient
    if (!camera_.grab(frame)) {
      return;
    }

    // Extract frame timestamp for SLAM synchronization
    rclcpp::Time frame_time = zed_to_ros_time(frame.timestamp_ns);

    // Inference (DLA) + 3D Fusion (ZED SDK)
    auto detections = perception_.detect_2d(frame);
    auto cones_3d = camera_.fuse_3d(detections);

    // Publish Visualizations
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;

    for (const auto &cone : cones_3d) {
      if (cone.is_valid) {
        markers.markers.push_back(
            to_marker(cone, marker_id++, frame_id_, frame_time));
      }
    }

    if (!markers.markers.empty()) {
      marker_pub_->publish(markers);
    }

    // Logging throttle (every 30 frames)
    if (frame_count_++ % 30 == 0) {
      int valid_count = 0;
      for (const auto &c : cones_3d)
        if (c.is_valid)
          valid_count++;

      RCLCPP_INFO(this->get_logger(),
                  "Frame %d: %zu 2D detections, %d 3D cones", frame_count_,
                  detections.size(), valid_count);
    }
  }

  unina::Perception3D perception_;
  unina::ZedImageSource camera_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string frame_id_;
  int frame_count_ = 0;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    auto node = std::make_shared<UninaPerceptionNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }

  rclcpp::shutdown();
  return 0;
}

#else

#include <iostream>

int main() {
  std::cerr << "ROS support not enabled. Build with: colcon build --cmake-args "
               "-DUSE_ROS=ON"
            << std::endl;
  return -1;
}

#endif // USE_ROS
