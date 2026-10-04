#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"

namespace unina {

class UninaPerception {
public:
  UninaPerception();
  ~UninaPerception();

  // Initialize TensorRT engine and allocate Zero-Copy memory
  bool init(const std::string &engine_path);

  // Run inference on the provided image buffer.
  // If input_is_fp16() is true, buffer must be half (2 bytes/elem).
  // If false, buffer must be float (4 bytes/elem).
  void inference();

  // Manually synchronize the CUDA stream. Call this before using outputs on
  // CPU.
  void synchronize();

  // Get pointers to the CPU-mapped memory for inputs and outputs
  void *get_input_ptr();
  std::vector<void *> get_output_ptrs();

  // Get pointers to the GPU device memory (for CUDA kernels)
  void *get_input_gpu_ptr();
  std::vector<void *> get_output_gpu_ptrs();
  const std::vector<OutputTensorLayout> &get_output_layouts() const {
    return output_layouts_;
  }

  // Type query helpers
  bool input_is_fp16() const { return input_is_fp16_; }
  bool output_is_fp16() const { return output_is_fp16_; }
  bool is_dla_active() const { return dla_active_; }
  int get_num_classes() const { return num_classes_; }
  int get_num_scales() const { return num_scales_; }
  const std::vector<int> &get_strides() const { return strides_; }

  // Get the internal CUDA stream for unified pipeline operations
  // All preprocessing and postprocessing operations must use this stream
  // to avoid race conditions with TensorRT inference.
  cudaStream_t get_stream() const { return stream_; }

private:
  // TensorRT objects
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;

  // Logger
  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
  } logger_;

  // Memory pointers
  void *input_cpu_ptr_ = nullptr;
  void *input_gpu_ptr_ = nullptr;

  // Track dynamic outputs (regX, clsX)
  std::vector<std::string> reg_names_;
  std::vector<std::string> cls_names_;
  std::vector<int> strides_;
  int num_scales_ = 0;

  std::vector<void *> output_cpu_ptrs_;
  std::vector<void *> output_gpu_ptrs_;
  std::vector<OutputTensorLayout> output_layouts_;

  // Helper to allocate zero-copy memory
  bool allocate_zero_copy_memory(size_t size, void **cpu_ptr, void **gpu_ptr);

  // internal state
  size_t input_size_bytes_ = 0;
  std::vector<size_t> output_sizes_bytes_;
  bool input_is_fp16_ = false;
  bool output_is_fp16_ = false;
  bool dla_active_ = false;
  int num_classes_ = 4; // Default safe fallback
  int input_h_ = 640;   // Dynamic input height tracking
};

} // namespace unina
