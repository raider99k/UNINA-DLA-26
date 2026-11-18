#include "UninaPerception.hpp"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

namespace unina {

namespace {

int round_up(int value, int multiple) {
  if (multiple <= 1) {
    return value;
  }
  return ((value + multiple - 1) / multiple) * multiple;
}

OutputTensorLayout get_output_layout(const nvinfer1::ICudaEngine &engine,
                                     const char *name, int logical_channels) {
  OutputTensorLayout layout;
  layout.logical_channels = logical_channels;
  layout.storage_channels = logical_channels;

  const auto format = engine.getTensorFormat(name);
  const int vectorized_dim = engine.getTensorVectorizedDim(name);
  const int components = engine.getTensorComponentsPerElement(name);

  if (format == nvinfer1::TensorFormat::kCHW16) {
    layout.layout = TensorLayout::kCHW16;
    layout.vector_width = 16;
  } else if (vectorized_dim == 1 && components > 1) {
    layout.layout = TensorLayout::kCHW16;
    layout.vector_width = components;
  }

  layout.storage_channels =
      round_up(layout.logical_channels, layout.vector_width);
  return layout;
}

size_t get_storage_volume(const nvinfer1::Dims &dims,
                          const OutputTensorLayout &layout) {
  size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    const int dim = (i == 1) ? layout.storage_channels : dims.d[i];
    volume *= static_cast<size_t>(std::max(1, dim));
  }
  return volume;
}

} // namespace

void UninaPerception::Logger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cout << "[TRT] " << msg << std::endl;
  }
}

UninaPerception::UninaPerception() : stream_(nullptr) {}

UninaPerception::~UninaPerception() {
  if (stream_)
    cudaStreamDestroy(stream_);

  // Free Zero-Copy Memory
  if (input_cpu_ptr_)
    cudaFreeHost(input_cpu_ptr_);
  for (void *ptr : output_cpu_ptrs_) {
    if (ptr)
      cudaFreeHost(ptr);
  }
}

bool UninaPerception::allocate_zero_copy_memory(size_t size, void **cpu_ptr,
                                                void **gpu_ptr) {
  if (size == 0)
    return false;
  unsigned int flags = cudaHostAllocMapped;

  cudaError_t err = cudaHostAlloc(cpu_ptr, size, flags);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  err = cudaHostGetDevicePointer(gpu_ptr, *cpu_ptr, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostGetDevicePointer failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool UninaPerception::init(const std::string &engine_path) {
  // 1. Load Engine File
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "Error reading engine file: " << engine_path << std::endl;
    return false;
  }
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();

  // 2. Create Runtime and Engine
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_)
    return false;

  engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
  if (!engine_)
    return false;

  context_.reset(engine_->createExecutionContext());
  if (!context_)
    return false;

  // 2.5 Hardware Acceleration Diagnostics
  int dla_core = engine_->getDLACore();
  if (dla_core != -1) {
    std::cout << "[TRT] Info: Engine targets DLA Core " << dla_core
              << std::endl;
    context_->setDLACore(dla_core);
    dla_active_ = true;

    // Check for GPU fallback: DLA engines should report themselves as such
    // Note: TensorRT 8.6+ has better introspection, here we log intended target
    std::cout << "[TRT] Pipeline: DLA-Accelerated mode active." << std::endl;
  } else {
    std::cout << "[TRT] Warning: Engine built for GPU. DLA ignored."
              << std::endl;
  }

  // 3. Allocate Memory
  int nbBindings = engine_->getNbIOTensors();

  // DYNAMIC BINDING DISCOVERY: Scan engine for regX and clsX pairs
  std::vector<std::string> reg_candidates;
  std::vector<std::string> cls_candidates;

  for (int i = 0; i < nbBindings; ++i) {
    std::string name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name.c_str());
    if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (name.find("reg") == 0)
        reg_candidates.push_back(name);
      else if (name.find("cls") == 0)
        cls_candidates.push_back(name);
    }
  }

  // Sort and match to ensure correct scale ordering (P3 -> P4 -> P5 etc)
  // We assume names follow reg3, reg4... or similar pattern that sorts
  // logically.
  std::sort(reg_candidates.begin(), reg_candidates.end());
  std::sort(cls_candidates.begin(), cls_candidates.end());

  if (reg_candidates.size() != cls_candidates.size() ||
      reg_candidates.empty()) {
    std::cerr << "Engine output mismatch: Found " << reg_candidates.size()
              << " reg and " << cls_candidates.size() << " cls outputs."
              << std::endl;
    return false;
  }

  num_scales_ = reg_candidates.size();
  reg_names_ = reg_candidates;
  cls_names_ = cls_candidates;

  // Input
  const char *input_name = "images";
  auto dims = engine_->getTensorShape(input_name);
  if (dims.nbDims == -1) {
    std::cerr << "Could not find input binding: " << input_name << std::endl;
    return false;
  }

  size_t vol = 1;
  for (int i = 0; i < dims.nbDims; ++i)
    vol *= std::max(1, dims.d[i]);

  // Capture input dimensions for dynamic preprocessing & stride calculation
  if (dims.nbDims >= 4) {
    // dims: [Batch, Channels, H, W]
    input_h_ = dims.d[2];
    input_w_ = dims.d[3];
    std::cout << "[TRT] Auto-detected Input Size: " << input_w_ << "x"
              << input_h_ << std::endl;
  } else if (dims.nbDims >= 3) {
    input_h_ = dims.d[2];
    std::cout << "[TRT] Auto-detected Input Height: " << input_h_ << std::endl;
  }

  auto type = engine_->getTensorDataType(input_name);
  input_is_fp16_ = (type == nvinfer1::DataType::kHALF);

  size_t element_size = input_is_fp16_ ? 2 : 4;
  input_size_bytes_ = vol * element_size;
  std::cout << "[TRT] Input '" << input_name
            << "' Type=" << (input_is_fp16_ ? "FP16" : "FP32")
            << " Size=" << input_size_bytes_ << " bytes" << std::endl;

  if (!allocate_zero_copy_memory(input_size_bytes_, &input_cpu_ptr_,
                                 &input_gpu_ptr_))
    return false;
  context_->setTensorAddress(input_name, input_gpu_ptr_);

  // Outputs
  std::vector<std::string> all_out_names;
  all_out_names.insert(all_out_names.end(), reg_names_.begin(),
                       reg_names_.end());
  all_out_names.insert(all_out_names.end(), cls_names_.begin(),
                       cls_names_.end());

  // Check first output type to set global output flag
  auto otype = engine_->getTensorDataType(reg_names_[0].c_str());
  output_is_fp16_ = (otype == nvinfer1::DataType::kHALF);

  strides_.clear();
  output_layouts_.clear();
  for (const auto &name : all_out_names) {
    dims = engine_->getTensorShape(name.c_str());
    if (dims.nbDims == -1) {
      std::cerr << "Could not find output binding: " << name << std::endl;
      return false;
    }

    const auto layout = get_output_layout(*engine_, name.c_str(), dims.d[1]);
    vol = get_storage_volume(dims, layout);

    // Infer strides and detect classes
    if (name.find("cls") == 0) {
      // dims: [Batch, Classes, H, W]
      if (dims.nbDims >= 2 && strides_.empty()) {
        num_classes_ = dims.d[1];
        std::cout << "[TRT] Auto-detected Num Classes: " << num_classes_
                  << std::endl;
      }
      // Infer stride: InputH / GridH
      int stride = input_h_ / dims.d[2]; // Dynamic stride calculation
      strides_.push_back(stride);
    }

    type = engine_->getTensorDataType(name.c_str());
    element_size = (type == nvinfer1::DataType::kHALF) ? 2 : 4;

    size_t bytes = vol * element_size;
    output_sizes_bytes_.push_back(bytes);

    void *cpu_ptr = nullptr;
    void *gpu_ptr = nullptr;

    if (!allocate_zero_copy_memory(bytes, &cpu_ptr, &gpu_ptr))
      return false;

    output_cpu_ptrs_.push_back(cpu_ptr);
    output_gpu_ptrs_.push_back(gpu_ptr);
    output_layouts_.push_back(layout);

    context_->setTensorAddress(name.c_str(), gpu_ptr);

    if (layout.vector_width > 1) {
      std::cout << "[TRT] Output '" << name << "' uses native CHW"
                << layout.vector_width << " storage"
                << " (logical C=" << layout.logical_channels
                << ", storage C=" << layout.storage_channels << ")"
                << std::endl;
    }
  }

  std::cout << "[TRT] Detected " << num_scales_ << " scales with strides: ";
  for (int s : strides_)
    std::cout << s << " ";
  std::cout << std::endl;

  std::cout << "[TRT] Outputs initialized. Precision="
            << (output_is_fp16_ ? "FP16" : "FP32") << std::endl;

  cudaStreamCreate(&stream_);
  return true;
}

void UninaPerception::inference() {
  if (!context_->enqueueV3(stream_)) {
    std::cerr << "enqueueV3 failed." << std::endl;
  }
}

void UninaPerception::synchronize() { cudaStreamSynchronize(stream_); }

void *UninaPerception::get_input_ptr() { return input_cpu_ptr_; }

std::vector<void *> UninaPerception::get_output_ptrs() {
  return output_cpu_ptrs_;
}

void *UninaPerception::get_input_gpu_ptr() { return input_gpu_ptr_; }

std::vector<void *> UninaPerception::get_output_gpu_ptrs() {
  return output_gpu_ptrs_;
}

} // namespace unina
