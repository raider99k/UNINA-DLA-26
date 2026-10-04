/**
 * @file FileImageSource.cpp
 * @brief File-based image source implementation
 */

#include "FileImageSource.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace unina {

FileImageSource::FileImageSource(const std::string &path, bool is_directory)
    : path_(path), is_directory_(is_directory), is_video_(false) {
  // Detect video files
  if (!is_directory) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    is_video_ = (ext == "mp4" || ext == "avi" || ext == "mkv" || ext == "mov" ||
                 ext == "webm");
  }
}

bool FileImageSource::init() {
  if (initialized_) {
    return true;
  }

  if (is_video_) {
    // Open video file
    video_cap_.open(path_);
    if (!video_cap_.isOpened()) {
      std::cerr << "FileImageSource: Failed to open video: " << path_
                << std::endl;
      return false;
    }
    width_ = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total = static_cast<int>(video_cap_.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "FileImageSource: Opened video " << width_ << "x" << height_
              << " (" << total << " frames)" << std::endl;
  } else if (is_directory_) {
    // Scan directory for images
    scan_directory();
    if (file_list_.empty()) {
      std::cerr << "FileImageSource: No images found in " << path_ << std::endl;
      return false;
    }
    // Get dimensions from first image
    cv::Mat first = cv::imread(file_list_[0]);
    if (first.empty()) {
      std::cerr << "FileImageSource: Failed to read " << file_list_[0]
                << std::endl;
      return false;
    }
    width_ = first.cols;
    height_ = first.rows;
    std::cout << "FileImageSource: Found " << file_list_.size() << " images, "
              << width_ << "x" << height_ << std::endl;
  } else {
    // Single image
    current_frame_ = cv::imread(path_);
    if (current_frame_.empty()) {
      std::cerr << "FileImageSource: Failed to load " << path_ << std::endl;
      return false;
    }
    width_ = current_frame_.cols;
    height_ = current_frame_.rows;
    file_list_.push_back(path_);
    std::cout << "FileImageSource: Loaded image " << width_ << "x" << height_
              << std::endl;
  }

  initialized_ = true;
  return true;
}

bool FileImageSource::grab(ImageFrame &frame) {
  if (!initialized_) {
    frame.is_valid = false;
    return false;
  }

  if (is_video_) {
    if (!video_cap_.read(current_frame_)) {
      frame.is_valid = false;
      return false;
    }
  } else {
    if (frame_index_ >= static_cast<int>(file_list_.size())) {
      frame.is_valid = false;
      return false;
    }
    current_frame_ = cv::imread(file_list_[frame_index_]);
    if (current_frame_.empty()) {
      frame.is_valid = false;
      return false;
    }
  }

  frame.width = current_frame_.cols;
  frame.height = current_frame_.rows;
  frame.channels = current_frame_.channels();
  frame.cpu_data = current_frame_.data;
  frame.gpu_data = nullptr;
  frame.data_on_gpu = false;
  frame.frame_id = frame_index_;
  frame.timestamp_ns = 0; // No timestamp for files
  frame.is_valid = true;

  frame_index_++;
  return true;
}

bool FileImageSource::has_more() const {
  if (!initialized_)
    return false;
  if (is_video_)
    return video_cap_.isOpened();
  return frame_index_ < static_cast<int>(file_list_.size());
}

void FileImageSource::close() {
  if (video_cap_.isOpened()) {
    video_cap_.release();
  }
  initialized_ = false;
}

void FileImageSource::scan_directory() {
  file_list_.clear();

  if (!fs::exists(path_) || !fs::is_directory(path_)) {
    return;
  }

  std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};

  for (const auto &entry : fs::directory_iterator(path_)) {
    if (!entry.is_regular_file())
      continue;

    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    for (const auto &valid_ext : extensions) {
      if (ext == valid_ext) {
        file_list_.push_back(entry.path().string());
        break;
      }
    }
  }

  // Sort alphabetically
  std::sort(file_list_.begin(), file_list_.end());
}

} // namespace unina
