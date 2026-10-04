#include "CudaPreprocess.cuh"
#include <cstdio>
#include <cuda_fp16.h>

namespace unina {

// ============================================================================
// Error Handling Macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return err;                                                              \
    }                                                                          \
  } while (0)

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

// Device memory for source image (when copying from CPU)
static uint8_t *d_src_buffer = nullptr;
static size_t d_src_buffer_size = 0;

// Last letterbox parameters (for coordinate adjustment)
static int g_x_offset = 0;
static int g_y_offset = 0;

// ============================================================================
// Initialization / Cleanup
// ============================================================================

cudaError_t cuda_preprocess_init(size_t max_src_size) {
  if (d_src_buffer != nullptr) {
    cudaFree(d_src_buffer);
  }
  d_src_buffer_size = max_src_size;
  return cudaMalloc(&d_src_buffer, max_src_size);
}

void cuda_preprocess_destroy() {
  if (d_src_buffer != nullptr) {
    cudaFree(d_src_buffer);
    d_src_buffer = nullptr;
    d_src_buffer_size = 0;
  }
}

void cuda_preprocess_get_offsets(int &x_offset, int &y_offset) {
  x_offset = g_x_offset;
  y_offset = g_y_offset;
}

// ============================================================================
// Bilinear Sampling Helpers
// ============================================================================

/**
 * @brief Bilinear interpolation for BGR (3-channel) images
 */
__device__ __forceinline__ float bilinear_sample_bgr(const uint8_t *src,
                                                     int src_w, int src_h,
                                                     float x, float y,
                                                     int channel) {
  // Clamp coordinates
  x = fmaxf(0.0f, fminf(x, (float)(src_w - 1)));
  y = fmaxf(0.0f, fminf(y, (float)(src_h - 1)));

  int x0 = (int)floorf(x);
  int y0 = (int)floorf(y);
  int x1 = min(x0 + 1, src_w - 1);
  int y1 = min(y0 + 1, src_h - 1);

  float dx = x - x0;
  float dy = y - y0;

  // HWC format (3 channels): index = (y * w + x) * 3 + channel
  float v00 = src[(y0 * src_w + x0) * 3 + channel];
  float v01 = src[(y0 * src_w + x1) * 3 + channel];
  float v10 = src[(y1 * src_w + x0) * 3 + channel];
  float v11 = src[(y1 * src_w + x1) * 3 + channel];

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;

  return v0 * (1.0f - dy) + v1 * dy;
}

/**
 * @brief Bilinear interpolation for RGBA/BGRA (4-channel) images (ZED X)
 */
__device__ __forceinline__ float bilinear_sample_rgba(const uint8_t *src,
                                                      int src_w, int src_h,
                                                      float x, float y,
                                                      int channel) {
  // Clamp coordinates
  x = fmaxf(0.0f, fminf(x, (float)(src_w - 1)));
  y = fmaxf(0.0f, fminf(y, (float)(src_h - 1)));

  int x0 = (int)floorf(x);
  int y0 = (int)floorf(y);
  int x1 = min(x0 + 1, src_w - 1);
  int y1 = min(y0 + 1, src_h - 1);

  float dx = x - x0;
  float dy = y - y0;

  // HWC format (4 channels): index = (y * w + x) * 4 + channel
  float v00 = src[(y0 * src_w + x0) * 4 + channel];
  float v01 = src[(y0 * src_w + x1) * 4 + channel];
  float v10 = src[(y1 * src_w + x0) * 4 + channel];
  float v11 = src[(y1 * src_w + x1) * 4 + channel];

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;

  return v0 * (1.0f - dy) + v1 * dy;
}

// ============================================================================
// Preprocessing Kernels
// ============================================================================

/**
 * @brief Fused preprocessing kernel for BGR images (OpenCV default)
 *
 * Performs in a single pass:
 * 1. Letterbox resize with bilinear interpolation
 * 2. BGR → RGB color conversion
 * 3. HWC → CHW layout transformation
 * 4. Normalize to [0, 1]
 */
__global__ void preprocess_kernel_bgr(const uint8_t *__restrict__ src,
                                      float *__restrict__ dst, int src_w,
                                      int src_h, int dst_w, int dst_h,
                                      int x_offset, int y_offset, float scale,
                                      float pad_value) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_w || dst_y >= dst_h)
    return;

  int plane_size = dst_w * dst_h;
  int dst_idx = dst_y * dst_w + dst_x;

  // Check if this pixel is in the letterbox padding region
  int scaled_w = (int)(src_w * scale);
  int scaled_h = (int)(src_h * scale);
  bool in_padding = (dst_x < x_offset || dst_x >= x_offset + scaled_w ||
                     dst_y < y_offset || dst_y >= y_offset + scaled_h);

  if (in_padding) {
    // Padding: fill with pad_value (114/255 ≈ 0.447)
    dst[0 * plane_size + dst_idx] = pad_value; // R
    dst[1 * plane_size + dst_idx] = pad_value; // G
    dst[2 * plane_size + dst_idx] = pad_value; // B
  } else {
    // Map destination coordinate back to source coordinate
    float src_x = (dst_x - x_offset) / scale;
    float src_y = (dst_y - y_offset) / scale;

    // Sample BGR and convert to RGB + normalize
    float b = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 0) / 255.0f;
    float g = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 1) / 255.0f;
    float r = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 2) / 255.0f;

    // Write CHW format (R, G, B planes)
    dst[0 * plane_size + dst_idx] = r;
    dst[1 * plane_size + dst_idx] = g;
    dst[2 * plane_size + dst_idx] = b;
  }
}

/**
 * @brief Fused preprocessing kernel for BGRA images (ZED X camera)
 *
 * ZED SDK outputs BGRA format in GPU memory. This kernel handles 4-channel
 * input and ignores the alpha channel.
 */
__global__ void preprocess_kernel_bgra(const uint8_t *__restrict__ src,
                                       float *__restrict__ dst, int src_w,
                                       int src_h, int dst_w, int dst_h,
                                       int x_offset, int y_offset, float scale,
                                       float pad_value) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_w || dst_y >= dst_h)
    return;

  int plane_size = dst_w * dst_h;
  int dst_idx = dst_y * dst_w + dst_x;

  // Check if this pixel is in the letterbox padding region
  int scaled_w = (int)(src_w * scale);
  int scaled_h = (int)(src_h * scale);
  bool in_padding = (dst_x < x_offset || dst_x >= x_offset + scaled_w ||
                     dst_y < y_offset || dst_y >= y_offset + scaled_h);

  if (in_padding) {
    dst[0 * plane_size + dst_idx] = pad_value; // R
    dst[1 * plane_size + dst_idx] = pad_value; // G
    dst[2 * plane_size + dst_idx] = pad_value; // B
  } else {
    float src_x = (dst_x - x_offset) / scale;
    float src_y = (dst_y - y_offset) / scale;

    // Sample BGRA (4 channels): B=0, G=1, R=2, A=3 (ignore A)
    float b = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 0) / 255.0f;
    float g = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 1) / 255.0f;
    float r = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 2) / 255.0f;

    dst[0 * plane_size + dst_idx] = r;
    dst[1 * plane_size + dst_idx] = g;
    dst[2 * plane_size + dst_idx] = b;
  }
}

// ============================================================================
// FP16 Preprocessing Kernels (for DLA input)
// ============================================================================

/**
 * @brief FP16 preprocessing kernel for BGR images
 *
 * Same as preprocess_kernel_bgr but outputs __half (FP16) for DLA input.
 */
__global__ void preprocess_kernel_bgr_fp16(const uint8_t *__restrict__ src,
                                           __half *__restrict__ dst, int src_w,
                                           int src_h, int dst_w, int dst_h,
                                           int x_offset, int y_offset,
                                           float scale, float pad_value) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_w || dst_y >= dst_h)
    return;

  int plane_size = dst_w * dst_h;
  int dst_idx = dst_y * dst_w + dst_x;

  int scaled_w = (int)(src_w * scale);
  int scaled_h = (int)(src_h * scale);
  bool in_padding = (dst_x < x_offset || dst_x >= x_offset + scaled_w ||
                     dst_y < y_offset || dst_y >= y_offset + scaled_h);

  if (in_padding) {
    __half pad_half = __float2half(pad_value);
    dst[0 * plane_size + dst_idx] = pad_half;
    dst[1 * plane_size + dst_idx] = pad_half;
    dst[2 * plane_size + dst_idx] = pad_half;
  } else {
    float src_x = (dst_x - x_offset) / scale;
    float src_y = (dst_y - y_offset) / scale;

    float b = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 0) / 255.0f;
    float g = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 1) / 255.0f;
    float r = bilinear_sample_bgr(src, src_w, src_h, src_x, src_y, 2) / 255.0f;

    dst[0 * plane_size + dst_idx] = __float2half(r);
    dst[1 * plane_size + dst_idx] = __float2half(g);
    dst[2 * plane_size + dst_idx] = __float2half(b);
  }
}

/**
 * @brief FP16 preprocessing kernel for BGRA images (ZED X)
 */
__global__ void preprocess_kernel_bgra_fp16(const uint8_t *__restrict__ src,
                                            __half *__restrict__ dst, int src_w,
                                            int src_h, int dst_w, int dst_h,
                                            int x_offset, int y_offset,
                                            float scale, float pad_value) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_w || dst_y >= dst_h)
    return;

  int plane_size = dst_w * dst_h;
  int dst_idx = dst_y * dst_w + dst_x;

  int scaled_w = (int)(src_w * scale);
  int scaled_h = (int)(src_h * scale);
  bool in_padding = (dst_x < x_offset || dst_x >= x_offset + scaled_w ||
                     dst_y < y_offset || dst_y >= y_offset + scaled_h);

  if (in_padding) {
    __half pad_half = __float2half(pad_value);
    dst[0 * plane_size + dst_idx] = pad_half;
    dst[1 * plane_size + dst_idx] = pad_half;
    dst[2 * plane_size + dst_idx] = pad_half;
  } else {
    float src_x = (dst_x - x_offset) / scale;
    float src_y = (dst_y - y_offset) / scale;

    float b = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 0) / 255.0f;
    float g = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 1) / 255.0f;
    float r = bilinear_sample_rgba(src, src_w, src_h, src_x, src_y, 2) / 255.0f;

    dst[0 * plane_size + dst_idx] = __float2half(r);
    dst[1 * plane_size + dst_idx] = __float2half(g);
    dst[2 * plane_size + dst_idx] = __float2half(b);
  }
}

// ============================================================================
// Public API
// ============================================================================

float cuda_preprocess(const uint8_t *src, int src_w, int src_h, float *dst,
                      int dst_w, int dst_h, bool src_on_gpu,
                      cudaStream_t stream) {
  // Calculate letterbox parameters
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int x_offset = (dst_w - new_w) / 2;
  int y_offset = (dst_h - new_h) / 2;

  // Store for later retrieval
  g_x_offset = x_offset;
  g_y_offset = y_offset;

  // Get source pointer on GPU
  const uint8_t *d_src;
  if (src_on_gpu) {
    d_src = src;
  } else {
    // Copy from CPU to GPU (assumes BGR, 3 channels)
    size_t src_size = src_w * src_h * 3 * sizeof(uint8_t);
    if (src_size > d_src_buffer_size) {
      fprintf(stderr,
              "CUDA Preprocess: Source image too large (%zu > %zu). "
              "Call cuda_preprocess_init() with larger size.\n",
              src_size, d_src_buffer_size);
      return 0.0f;
    }
    if (stream) {
      cudaMemcpyAsync(d_src_buffer, src, src_size, cudaMemcpyHostToDevice,
                      stream);
    } else {
      cudaMemcpy(d_src_buffer, src, src_size, cudaMemcpyHostToDevice);
    }
    d_src = d_src_buffer;
  }

  // Launch preprocessing kernel
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  float pad_value = 114.0f / 255.0f; // Standard YOLO padding

  preprocess_kernel_bgr<<<grid, block, 0, stream>>>(d_src, dst, src_w, src_h,
                                                    dst_w, dst_h, x_offset,
                                                    y_offset, scale, pad_value);

  // Check for kernel launch errors
  CUDA_CHECK_KERNEL();

  return scale;
}

/**
 * @brief Preprocess BGRA image (ZED X camera) for DLA inference
 *
 * Same as cuda_preprocess but handles 4-channel BGRA input.
 * Source MUST be on GPU (ZED SDK provides GPU memory directly).
 */
float cuda_preprocess_bgra(const uint8_t *src, int src_w, int src_h, float *dst,
                           int dst_w, int dst_h, cudaStream_t stream) {
  // Calculate letterbox parameters
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int x_offset = (dst_w - new_w) / 2;
  int y_offset = (dst_h - new_h) / 2;

  // Store for later retrieval
  g_x_offset = x_offset;
  g_y_offset = y_offset;

  // Launch preprocessing kernel
  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  float pad_value = 114.0f / 255.0f;

  preprocess_kernel_bgra<<<grid, block, 0, stream>>>(
      src, dst, src_w, src_h, dst_w, dst_h, x_offset, y_offset, scale,
      pad_value);

  // Check for kernel launch errors
  CUDA_CHECK_KERNEL();

  return scale;
}

// ============================================================================
// FP16 Public API (for DLA input)
// ============================================================================

/**
 * @brief Preprocess BGR image to FP16 output (for DLA input)
 */
float cuda_preprocess_fp16(const uint8_t *src, int src_w, int src_h, void *dst,
                           int dst_w, int dst_h, bool src_on_gpu,
                           cudaStream_t stream) {
  // Calculate letterbox parameters
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int x_offset = (dst_w - new_w) / 2;
  int y_offset = (dst_h - new_h) / 2;

  g_x_offset = x_offset;
  g_y_offset = y_offset;

  const uint8_t *d_src;
  if (src_on_gpu) {
    d_src = src;
  } else {
    size_t src_size = src_w * src_h * 3 * sizeof(uint8_t);
    if (src_size > d_src_buffer_size) {
      fprintf(stderr, "CUDA Preprocess FP16: Source too large.\n");
      return 0.0f;
    }
    if (stream) {
      cudaMemcpyAsync(d_src_buffer, src, src_size, cudaMemcpyHostToDevice,
                      stream);
    } else {
      cudaMemcpy(d_src_buffer, src, src_size, cudaMemcpyHostToDevice);
    }
    d_src = d_src_buffer;
  }

  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  float pad_value = 114.0f / 255.0f;

  preprocess_kernel_bgr_fp16<<<grid, block, 0, stream>>>(
      d_src, (__half *)dst, src_w, src_h, dst_w, dst_h, x_offset, y_offset,
      scale, pad_value);

  CUDA_CHECK_KERNEL();
  return scale;
}

/**
 * @brief Preprocess BGRA image (ZED X) to FP16 output (for DLA input)
 */
float cuda_preprocess_bgra_fp16(const uint8_t *src, int src_w, int src_h,
                                void *dst, int dst_w, int dst_h,
                                cudaStream_t stream) {
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int x_offset = (dst_w - new_w) / 2;
  int y_offset = (dst_h - new_h) / 2;

  g_x_offset = x_offset;
  g_y_offset = y_offset;

  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  float pad_value = 114.0f / 255.0f;

  preprocess_kernel_bgra_fp16<<<grid, block, 0, stream>>>(
      src, (__half *)dst, src_w, src_h, dst_w, dst_h, x_offset, y_offset, scale,
      pad_value);

  CUDA_CHECK_KERNEL();
  return scale;
}

// ============================================================================
// HWC4 Preprocessing (DLA-native input format)
// ============================================================================

/**
 * @brief Bilinear interpolation for BGRA with RGBA HWC4 output
 *
 * Direct BGRA→RGBA conversion with channel swap during sampling.
 * Returns float value for specified output channel (R=0, G=1, B=2, A=3).
 */
__device__ __forceinline__ float
bilinear_sample_bgra_to_rgba(const uint8_t *src, int src_w, int src_h, float x,
                             float y, int out_channel) {
  // Clamp coordinates
  x = fmaxf(0.0f, fminf(x, (float)(src_w - 1)));
  y = fmaxf(0.0f, fminf(y, (float)(src_h - 1)));

  int x0 = (int)floorf(x);
  int y0 = (int)floorf(y);
  int x1 = min(x0 + 1, src_w - 1);
  int y1 = min(y0 + 1, src_h - 1);

  float dx = x - x0;
  float dy = y - y0;

  // BGRA→RGBA channel mapping: out_channel 0(R)→src 2, 1(G)→src 1, 2(B)→src 0,
  // 3(A)→src 3
  int src_channel = (out_channel == 0)   ? 2
                    : (out_channel == 2) ? 0
                                         : out_channel;

  // HWC format (4 channels): index = (y * w + x) * 4 + channel
  float v00 = src[(y0 * src_w + x0) * 4 + src_channel];
  float v01 = src[(y0 * src_w + x1) * 4 + src_channel];
  float v10 = src[(y1 * src_w + x0) * 4 + src_channel];
  float v11 = src[(y1 * src_w + x1) * 4 + src_channel];

  float v0 = v00 * (1.0f - dx) + v01 * dx;
  float v1 = v10 * (1.0f - dx) + v11 * dx;

  return v0 * (1.0f - dy) + v1 * dy;
}

/**
 * @brief Fused preprocessing kernel for BGRA→HWC4 FP16 output (DLA native)
 *
 * ZED X outputs BGRA in HWC format. This kernel:
 * 1. Letterbox resize with bilinear interpolation
 * 2. BGRA → RGBA color conversion (DLA expects RGB order)
 * 3. Keeps HWC layout (channel-interleaved, 4 channels)
 * 4. Normalize to [0, 1] and output FP16
 *
 * Output memory layout: For each pixel at (y, x), 4 consecutive FP16 values (R,
 * G, B, A) DLA HWC4 requires W stride alignment to 64 bytes on Orin = 32 FP16
 * elements = 8 pixels. For 640 width, 640 % 8 = 0, so no padding needed.
 */
__global__ void preprocess_kernel_bgra_to_hwc4_fp16(
    const uint8_t *__restrict__ src, __half *__restrict__ dst, int src_w,
    int src_h, int dst_w, int dst_h, int x_offset, int y_offset, float scale,
    float pad_value) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_w || dst_y >= dst_h)
    return;

  // HWC4 Orin Requirement: Row stride must be 64-byte aligned.
  // FP16 (2 bytes) * 4 channels = 8 bytes per pixel.
  // 64 bytes / 8 bytes/pixel = 8 pixels alignment.
  int line_stride_pixels = (dst_w + 7) & ~7;
  int base_idx = (dst_y * line_stride_pixels + dst_x) * 4;

  // Check if this pixel is in the letterbox padding region
  int scaled_w = (int)(src_w * scale);
  int scaled_h = (int)(src_h * scale);
  bool in_padding = (dst_x < x_offset || dst_x >= x_offset + scaled_w ||
                     dst_y < y_offset || dst_y >= y_offset + scaled_h);

  if (in_padding) {
    __half pad_half = __float2half(pad_value);
    __half zero_half = __float2half(0.0f);
    dst[base_idx + 0] = pad_half;  // R
    dst[base_idx + 1] = pad_half;  // G
    dst[base_idx + 2] = pad_half;  // B
    dst[base_idx + 3] = zero_half; // A (padding channel)
  } else {
    float src_x = (dst_x - x_offset) / scale;
    float src_y = (dst_y - y_offset) / scale;

    // Sample and convert BGRA→RGBA, normalize to [0,1]
    float r = bilinear_sample_bgra_to_rgba(src, src_w, src_h, src_x, src_y, 0) /
              255.0f;
    float g = bilinear_sample_bgra_to_rgba(src, src_w, src_h, src_x, src_y, 1) /
              255.0f;
    float b = bilinear_sample_bgra_to_rgba(src, src_w, src_h, src_x, src_y, 2) /
              255.0f;

    dst[base_idx + 0] = __float2half(r);
    dst[base_idx + 1] = __float2half(g);
    dst[base_idx + 2] = __float2half(b);
    dst[base_idx + 3] = __float2half(0.0f); // A channel = 0 (DLA ignores it)
  }
}

/**
 * @brief Preprocess BGRA image (ZED X) to HWC4 FP16 output (DLA native format)
 *
 * This is the optimal preprocessing path for DLA: ZED's BGRA HWC output is
 * converted directly to RGBA HWC4 FP16, eliminating the CHW transpose.
 *
 * @param src Source BGRA image on GPU (from ZED SDK sl::Mat::getPtr)
 * @param src_w Source image width
 * @param src_h Source image height
 * @param dst Destination buffer - FP16 HWC4 format (__half*), size = dst_w *
 * dst_h * 4 * 2 bytes
 * @param dst_w Target width (e.g., 768)
 * @param dst_h Target height (e.g., 480)
 * @param stream CUDA stream for async execution
 * @return Letterbox scale factor
 */
float cuda_preprocess_bgra_hwc4_fp16(const uint8_t *src, int src_w, int src_h,
                                     void *dst, int dst_w, int dst_h,
                                     cudaStream_t stream) {
  float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
  int new_w = (int)(src_w * scale);
  int new_h = (int)(src_h * scale);
  int x_offset = (dst_w - new_w) / 2;
  int y_offset = (dst_h - new_h) / 2;

  g_x_offset = x_offset;
  g_y_offset = y_offset;

  dim3 block(32, 32);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  float pad_value = 114.0f / 255.0f;

  preprocess_kernel_bgra_to_hwc4_fp16<<<grid, block, 0, stream>>>(
      src, (__half *)dst, src_w, src_h, dst_w, dst_h, x_offset, y_offset, scale,
      pad_value);

  CUDA_CHECK_KERNEL();
  return scale;
}

} // namespace unina
