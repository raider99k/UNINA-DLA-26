#pragma once
#include <string>
#include <vector>


namespace unina {

// Model Constants
constexpr int kNumClasses = 4; // Yellow, Blue, SmallOrange, BigOrange
constexpr int kRegMax = 16;

// Default image dimensions (fallback if not specified in config).
// The main CUDA path reads dimensions from config.yaml.
// PostProcess.cpp CPU fallback has been removed; full pipeline is CUDA/DLA resident.
constexpr int kInputH = 640;
constexpr int kInputW = 640;

enum class TensorLayout {
  kCHW,
  kCHW16,
};

struct OutputTensorLayout {
  int logical_channels = 0;
  int storage_channels = 0;
  int vector_width = 1;
  TensorLayout layout = TensorLayout::kCHW;
};

// Detection Output Structure
struct Detection {
  float x1, y1, x2, y2; // Absolute coordinates in image frame
  float confidence;
  int class_id;

  // Optional: for sorting
  bool operator>(const Detection &other) const {
    return confidence > other.confidence;
  }
};

} // namespace unina
