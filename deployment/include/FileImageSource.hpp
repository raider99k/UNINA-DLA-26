#pragma once

/**
 * @file FileImageSource.hpp
 * @brief File-based implementation of IImageSource
 *
 * This provides offline image input for testing and debugging:
 * - Single image files
 * - Directory of images (for ROS2 bag extracted frames)
 * - Video files
 */

#include "IImageSource.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace unina {

/**
 * @brief File-based image source for offline testing
 *
 * Supports:
 * - Single image: FileImageSource("path/to/image.jpg")
 * - Directory: FileImageSource("path/to/frames/", true)
 * - Video: FileImageSource("path/to/video.mp4")
 */
class FileImageSource : public IImageSource {
public:
  /**
   * @brief Construct from file path
   * @param path Path to image, directory, or video
   * @param is_directory If true, treat path as directory of images
   */
  explicit FileImageSource(const std::string &path, bool is_directory = false);
  ~FileImageSource() override { close(); }

  // IImageSource interface
  bool init() override;
  bool grab(ImageFrame &frame) override;
  bool has_more() const override;
  int width() const override { return width_; }
  int height() const override { return height_; }
  int channels() const override { return 3; } // BGR
  bool provides_gpu_data() const override { return false; }
  void close() override;

  /**
   * @brief Get total number of frames (if known)
   */
  int total_frames() const { return static_cast<int>(file_list_.size()); }

  /**
   * @brief Get current frame index
   */
  int current_frame() const { return frame_index_; }

private:
  std::string path_;
  bool is_directory_;
  bool is_video_;

  std::vector<std::string> file_list_;
  cv::VideoCapture video_cap_;
  cv::Mat current_frame_;

  int frame_index_ = 0;
  int width_ = 0;
  int height_ = 0;
  bool initialized_ = false;

  void scan_directory();
};

} // namespace unina
