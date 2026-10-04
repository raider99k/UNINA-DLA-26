#include "CudaPostprocess.cuh"
#include <algorithm>
#include <cstdio>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace unina {

// ============================================================================
// Error Handling Macro
// ============================================================================
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error in %s:%d: %s\n", __FILE__, __LINE__,  \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

// ============================================================================
// Module State
// ============================================================================

// Device memory for detections
static Detection *d_detections = nullptr;
static Detection *d_filtered_detections = nullptr;
static int *d_detection_count = nullptr;
static int *d_filtered_count = nullptr;
static unsigned char *d_keep_flags = nullptr;

// Host memory for results
static Detection *h_detections = nullptr;

// Strides are now passed dynamically from the Perception engine
// static const int strides[3] = {8, 16, 32};

// ============================================================================
// Initialization / Cleanup
// ============================================================================

cudaError_t cuda_postprocess_init() {
  cudaError_t err;

  // Allocate device memory for detections
  err = cudaMalloc(&d_detections, kMaxDetections * sizeof(Detection));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  // Allocate secondary buffer for compacted (NMS) detections
  err = cudaMalloc(&d_filtered_detections, kMaxDetections * sizeof(Detection));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  // Allocate device memory for atomic counter
  err = cudaMalloc(&d_detection_count, sizeof(int));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  err = cudaMalloc(&d_filtered_count, sizeof(int));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  // Allocate keep-mask buffer for NMS decisions
  err = cudaMalloc(&d_keep_flags, kMaxDetections * sizeof(unsigned char));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  // Allocate pinned host memory for fast transfer
  err = cudaMallocHost(&h_detections, kMaxDetections * sizeof(Detection));
  if (err != cudaSuccess) {
    cuda_postprocess_destroy();
    return err;
  }

  return cudaSuccess;
}

void cuda_postprocess_destroy() {
  if (d_detections) {
    cudaFree(d_detections);
    d_detections = nullptr;
  }
  if (d_filtered_detections) {
    cudaFree(d_filtered_detections);
    d_filtered_detections = nullptr;
  }
  if (d_detection_count) {
    cudaFree(d_detection_count);
    d_detection_count = nullptr;
  }
  if (d_filtered_count) {
    cudaFree(d_filtered_count);
    d_filtered_count = nullptr;
  }
  if (d_keep_flags) {
    cudaFree(d_keep_flags);
    d_keep_flags = nullptr;
  }
  if (h_detections) {
    cudaFreeHost(h_detections);
    h_detections = nullptr;
  }
}

/**
 * @brief Get value from buffer using native CUDA intrinsics for speed
 */
__device__ __forceinline__ float get_val_device(const void *ptr, int idx,
                                                bool is_fp16) {
  if (is_fp16) {
    return __half2float(reinterpret_cast<const __half *>(ptr)[idx]);
  } else {
    return reinterpret_cast<const float *>(ptr)[idx];
  }
}

__device__ __forceinline__ int logical_to_storage_idx(int channel,
                                                      int spatial_idx,
                                                      int grid_size,
                                                      OutputTensorLayout layout) {
  if (layout.vector_width <= 1) {
    return channel * grid_size + spatial_idx;
  }

  return ((channel / layout.vector_width) * grid_size + spatial_idx) *
             layout.vector_width +
         (channel % layout.vector_width);
}

__device__ __forceinline__ float get_channel_val_device(
    const void *ptr, int channel, int spatial_idx, int grid_size,
    OutputTensorLayout layout, bool is_fp16) {
  const int idx =
      logical_to_storage_idx(channel, spatial_idx, grid_size, layout);
  return get_val_device(ptr, idx, is_fp16);
}

struct DetectionConfidenceDesc {
  __host__ __device__ bool operator()(const Detection &lhs,
                                      const Detection &rhs) const {
    if (lhs.confidence == rhs.confidence) {
      return lhs.class_id < rhs.class_id;
    }
    return lhs.confidence > rhs.confidence;
  }
};

static inline void sort_detections_by_confidence(Detection *d_buffer, int count,
                                                 cudaStream_t stream) {
  if (count <= 1 || !d_buffer) {
    return;
  }
  thrust::device_ptr<Detection> begin(d_buffer);
  thrust::device_ptr<Detection> end = begin + count;
  if (stream) {
    thrust::sort(thrust::cuda::par.on(stream), begin, end,
                 DetectionConfidenceDesc{});
  } else {
    thrust::sort(begin, end, DetectionConfidenceDesc{});
  }
}

/**
 * @brief Parallel decode kernel for a single scale
 *
 * One thread per grid cell. Finds best class, decodes box if above threshold,
 * and atomically appends to output array.
 */
__global__ void decode_kernel(const void *__restrict__ cls_ptr,
                              const void *__restrict__ reg_ptr,
                              Detection *__restrict__ detections,
                              int *__restrict__ detection_count,
                              float conf_thres, int num_classes, int grid_h,
                              int grid_w, int stride,
                              OutputTensorLayout cls_layout,
                              OutputTensorLayout reg_layout, bool is_fp16) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= grid_w || y >= grid_h)
    return;

  int spatial_idx = y * grid_w + x;
  int grid_size = grid_h * grid_w;

  // Find best class
  int best_class = -1;
  float best_conf = conf_thres;

  for (int c = 0; c < num_classes; ++c) {
    float conf = get_channel_val_device(cls_ptr, c, spatial_idx, grid_size,
                                        cls_layout, is_fp16);

    if (conf > best_conf) {
      best_conf = conf;
      best_class = c;
    }
  }

  // If no class above threshold, skip
  if (best_class < 0)
    return;

  // Decode box (LTRB format)
  float l = get_channel_val_device(reg_ptr, 0, spatial_idx, grid_size,
                                   reg_layout, is_fp16);
  float t = get_channel_val_device(reg_ptr, 1, spatial_idx, grid_size,
                                   reg_layout, is_fp16);
  float r = get_channel_val_device(reg_ptr, 2, spatial_idx, grid_size,
                                   reg_layout, is_fp16);
  float b = get_channel_val_device(reg_ptr, 3, spatial_idx, grid_size,
                                   reg_layout, is_fp16);

  // Anchor point (grid center)
  float ax = x + 0.5f;
  float ay = y + 0.5f;

  // Decode to absolute coordinates
  float x1 = (ax - l) * stride;
  float y1 = (ay - t) * stride;
  float x2 = (ax + r) * stride;
  float y2 = (ay + b) * stride;

  // Atomic append to output buffer
  int idx = atomicAdd(detection_count, 1);
  if (idx < kMaxDetections) {
    detections[idx].x1 = x1;
    detections[idx].y1 = y1;
    detections[idx].x2 = x2;
    detections[idx].y2 = y2;
    detections[idx].confidence = best_conf;
    detections[idx].class_id = best_class;
  }
}

__global__ void init_keep_mask_kernel(const Detection *__restrict__ detections,
                                      unsigned char *__restrict__ keep,
                                      int count, float score_threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  keep[idx] = detections[idx].confidence >= score_threshold ? 1 : 0;
}

__device__ __forceinline__ float iou_device(const Detection &a,
                                            const Detection &b) {
  const float xx1 = max(a.x1, b.x1);
  const float yy1 = max(a.y1, b.y1);
  const float xx2 = min(a.x2, b.x2);
  const float yy2 = min(a.y2, b.y2);
  const float w = max(0.0f, xx2 - xx1);
  const float h = max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = max(0.0f, a.x2 - a.x1) * max(0.0f, a.y2 - a.y1);
  const float area_b = max(0.0f, b.x2 - b.x1) * max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  if (uni <= 1e-6f) {
    return 0.0f;
  }
  return inter / uni;
}

// One-block deterministic NMS over confidence-sorted detections.
// For kMaxDetections<=1000 this is fast and avoids host-side suppression logic.
__global__ void nms_sorted_single_block_kernel(
    const Detection *__restrict__ detections, unsigned char *__restrict__ keep,
    int count, float iou_threshold) {
  if (blockIdx.x != 0) {
    return;
  }

  __shared__ int active_ref;

  for (int i = 0; i < count; ++i) {
    if (threadIdx.x == 0) {
      active_ref = keep[i] ? 1 : 0;
    }
    __syncthreads();

    if (!active_ref) {
      __syncthreads();
      continue;
    }

    const Detection ref = detections[i];
    for (int j = i + 1 + threadIdx.x; j < count; j += blockDim.x) {
      if (!keep[j]) {
        continue;
      }
      if (detections[j].class_id != ref.class_id) {
        continue;
      }
      if (iou_device(ref, detections[j]) > iou_threshold) {
        keep[j] = 0;
      }
    }
    __syncthreads();
  }
}

__global__ void pack_kept_detections_kernel(
    const Detection *__restrict__ input, const unsigned char *__restrict__ keep,
    int count, Detection *__restrict__ output, int *__restrict__ output_count,
    int max_detections) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int out_idx = 0;
  for (int i = 0; i < count; ++i) {
    if (!keep[i]) {
      continue;
    }
    output[out_idx++] = input[i];
    if (out_idx >= max_detections) {
      break;
    }
  }
  *output_count = out_idx;
}

std::vector<Detection> cuda_decode(const void *const *cls_buffers,
                                   const void *const *reg_buffers,
                                   const OutputTensorLayout *cls_layouts,
                                   const OutputTensorLayout *reg_layouts,
                                   int num_scales, const int *strides,
                                   int input_w, int input_h,
                                   float conf_threshold, int num_classes,
                                   bool use_nms, float score_threshold,
                                   float nms_iou_threshold, int max_detections,
                                   bool is_fp16, cudaStream_t stream) {
  if (!d_detections || !d_filtered_detections || !d_detection_count ||
      !d_filtered_count || !d_keep_flags || !h_detections) {
    fprintf(stderr, "CUDA Postprocess: Not initialized. Call "
                    "cuda_postprocess_init() first.\n");
    return {};
  }

  conf_threshold = std::clamp(conf_threshold, 0.0f, 1.0f);
  score_threshold = std::clamp(score_threshold, 0.0f, 1.0f);
  nms_iou_threshold = std::clamp(nms_iou_threshold, 0.0f, 1.0f);
  max_detections = std::max(1, std::min(max_detections, kMaxDetections));

  // Reset detection counter
  int zero = 0;
  if (stream) {
    cudaMemcpyAsync(d_detection_count, &zero, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
  } else {
    cudaMemcpy(d_detection_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
  }

  // Process each scale
  for (int scale = 0; scale < num_scales; ++scale) {
    int grid_h = input_h / strides[scale];
    int grid_w = input_w / strides[scale];

    dim3 block(16, 16);
    dim3 grid((grid_w + block.x - 1) / block.x,
              (grid_h + block.y - 1) / block.y);

    decode_kernel<<<grid, block, 0, stream>>>(
        cls_buffers[scale], reg_buffers[scale], d_detections, d_detection_count,
        conf_threshold, num_classes, grid_h, grid_w, strides[scale],
        cls_layouts[scale], reg_layouts[scale], is_fp16);
  }

  // Check for kernel launch errors
  CUDA_CHECK_KERNEL();

  // Get detection count
  int h_count = 0;
  if (stream) {
    cudaMemcpyAsync(&h_count, d_detection_count, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  } else {
    cudaMemcpy(&h_count, d_detection_count, sizeof(int),
               cudaMemcpyDeviceToHost);
  }

  // Clamp to max
  h_count = std::min(h_count, kMaxDetections);

  if (h_count == 0) {
    return {};
  }

  Detection *copy_source = d_detections;
  int copy_count = h_count;

  if (use_nms) {
    sort_detections_by_confidence(d_detections, h_count, stream);

    constexpr int kNmsThreads = 256;
    const int nms_blocks = (h_count + kNmsThreads - 1) / kNmsThreads;
    init_keep_mask_kernel<<<nms_blocks, kNmsThreads, 0, stream>>>(
        d_detections, d_keep_flags, h_count, score_threshold);
    CUDA_CHECK_KERNEL();

    nms_sorted_single_block_kernel<<<1, kNmsThreads, 0, stream>>>(
        d_detections, d_keep_flags, h_count, nms_iou_threshold);
    CUDA_CHECK_KERNEL();

    if (stream) {
      cudaMemcpyAsync(d_filtered_count, &zero, sizeof(int),
                      cudaMemcpyHostToDevice, stream);
    } else {
      cudaMemcpy(d_filtered_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }

    pack_kept_detections_kernel<<<1, 1, 0, stream>>>(
        d_detections, d_keep_flags, h_count, d_filtered_detections,
        d_filtered_count, max_detections);
    CUDA_CHECK_KERNEL();

    if (stream) {
      cudaMemcpyAsync(&copy_count, d_filtered_count, sizeof(int),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    } else {
      cudaMemcpy(&copy_count, d_filtered_count, sizeof(int),
                 cudaMemcpyDeviceToHost);
    }
    copy_count = std::max(0, std::min(copy_count, max_detections));
    copy_source = d_filtered_detections;
  } else if (h_count > max_detections) {
    // Keep highest-confidence detections when output cap is active.
    sort_detections_by_confidence(d_detections, h_count, stream);
    copy_count = max_detections;
  }

  if (copy_count == 0) {
    return {};
  }

  // Copy detections to host
  if (stream) {
    cudaMemcpyAsync(h_detections, copy_source, copy_count * sizeof(Detection),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  } else {
    cudaMemcpy(h_detections, copy_source, copy_count * sizeof(Detection),
               cudaMemcpyDeviceToHost);
  }

  // Return as vector
  return std::vector<Detection>(h_detections, h_detections + copy_count);
}

void adjust_detections_for_letterbox(std::vector<Detection> &detections,
                                     int x_offset, int y_offset, float scale,
                                     int orig_w, int orig_h) {
  for (auto &det : detections) {
    // Remove letterbox offset and scale to original size
    det.x1 = (det.x1 - x_offset) / scale;
    det.y1 = (det.y1 - y_offset) / scale;
    det.x2 = (det.x2 - x_offset) / scale;
    det.y2 = (det.y2 - y_offset) / scale;

    // Clip to image bounds
    det.x1 = std::max(0.0f, std::min(det.x1, (float)orig_w));
    det.y1 = std::max(0.0f, std::min(det.y1, (float)orig_h));
    det.x2 = std::max(0.0f, std::min(det.x2, (float)orig_w));
    det.y2 = std::max(0.0f, std::min(det.y2, (float)orig_h));
  }
}

} // namespace unina
