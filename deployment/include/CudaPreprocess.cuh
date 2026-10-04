#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace unina {

/**
 * @brief CUDA Preprocessing for Zero-Overhead Inference
 *
 * Performs fused letterbox resize + BGR→RGB + HWC→CHW + normalize.
 * Input can be either CPU or GPU memory; output is always GPU memory
 * ready for TensorRT inference.
 */

/**
 * @brief Initialize preprocessing resources (allocates device memory)
 * @param max_src_size Maximum expected source image size (width * height * 3)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_preprocess_init(size_t max_src_size);

/**
 * @brief Release preprocessing resources
 */
void cuda_preprocess_destroy();

/**
 * @brief Preprocess image for DLA inference (main entry point)
 *
 * @param src Source image data (HWC, BGR, uint8). If on CPU, will be copied to
 * GPU.
 * @param src_w Source image width
 * @param src_h Source image height
 * @param dst Destination buffer (CHW, RGB, float [0,1]). Must be GPU memory.
 * @param dst_w Target width (typically 640)
 * @param dst_h Target height (typically 640)
 * @param src_on_gpu True if src pointer is already in GPU memory (e.g., from
 * ZED SDK)
 * @param stream CUDA stream for async execution (nullptr for default stream)
 * @return Letterbox scale factor (needed for post-processing coordinate
 * adjustment)
 */
float cuda_preprocess(const uint8_t *src, int src_w, int src_h, float *dst,
                      int dst_w, int dst_h, bool src_on_gpu = false,
                      cudaStream_t stream = nullptr);

/**
 * @brief Preprocess BGRA image (ZED X camera) for DLA inference
 *
 * Same as cuda_preprocess but handles 4-channel BGRA input from ZED SDK.
 * Source MUST be on GPU (ZED SDK provides GPU memory directly).
 *
 * @param src Source BGRA image on GPU (from ZED SDK sl::Mat::getPtr)
 * @param src_w Source image width
 * @param src_h Source image height
 * @param dst Destination buffer (CHW, RGB, float [0,1]). Must be GPU memory.
 * @param dst_w Target width (typically 640)
 * @param dst_h Target height (typically 640)
 * @param stream CUDA stream for async execution
 * @return Letterbox scale factor
 */
float cuda_preprocess_bgra(const uint8_t *src, int src_w, int src_h, float *dst,
                           int dst_w, int dst_h, cudaStream_t stream = nullptr);

// ============================================================================
// FP16 Preprocessing (for DLA input)
// ============================================================================

/**
 * @brief Preprocess BGR image with FP16 output (for DLA)
 *
 * Same as cuda_preprocess but outputs FP16 (__half) for DLA input.
 * Use this when your TensorRT engine expects FP16 input.
 *
 * @param dst Destination buffer - must be FP16 (__half*) cast to void*
 */
float cuda_preprocess_fp16(const uint8_t *src, int src_w, int src_h,
                           void *dst, // __half* cast to void*
                           int dst_w, int dst_h, bool src_on_gpu = false,
                           cudaStream_t stream = nullptr);

/**
 * @brief Preprocess BGRA image (ZED X) with FP16 output (for DLA)
 *
 * @param dst Destination buffer - must be FP16 (__half*) cast to void*
 */
float cuda_preprocess_bgra_fp16(const uint8_t *src, int src_w, int src_h,
                                void *dst, // __half* cast to void*
                                int dst_w, int dst_h,
                                cudaStream_t stream = nullptr);

// ============================================================================
// HWC4 Preprocessing (DLA-native input format)
// ============================================================================

/**
 * @brief Preprocess BGRA image (ZED X) to HWC4 FP16 output (DLA native format)
 *
 * This is the optimal preprocessing path for DLA: ZED's BGRA HWC output is
 * converted directly to RGBA HWC4 FP16, eliminating the CHW transpose.
 * Use with --inputIOFormats=fp16:dla_hwc4 when building TensorRT engine.
 *
 * @param dst Destination buffer - FP16 HWC4 format (__half*), size = dst_w *
 * dst_h * 4 * 2 bytes
 */
float cuda_preprocess_bgra_hwc4_fp16(const uint8_t *src, int src_w, int src_h,
                                     void *dst, // __half* cast to void*
                                     int dst_w, int dst_h,
                                     cudaStream_t stream = nullptr);

/**
 * @brief Get letterbox offsets from last preprocessing call
 * @param x_offset Output: horizontal offset in pixels
 * @param y_offset Output: vertical offset in pixels
 */
void cuda_preprocess_get_offsets(int &x_offset, int &y_offset);

} // namespace unina
