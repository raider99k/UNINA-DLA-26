#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace unina {

/**
 * @brief CUDA Preprocessing for Zero-Overhead Inference
 *
 * Performs fused letterbox resize + BGR/BGRA→RGB + HWC→CHW/HWC4 + normalize.
 * Input can be either CPU or GPU memory; output is always GPU memory
 * ready for TensorRT inference.
 */

cudaError_t cuda_preprocess_init(size_t max_src_size);
void cuda_preprocess_destroy();

/**
 * @brief Preprocess BGR image for DLA/GPU inference
 *
 * @param src Source image data (HWC, BGR, uint8). If on CPU, will be copied to GPU.
 * @param src_w Source image width
 * @param src_h Source image height
 * @param dst Destination buffer (CHW, RGB, float [0,1]). Must be GPU memory.
 * @param dst_w Target width
 * @param dst_h Target height
 * @param src_on_gpu True if src pointer is already in GPU memory
 * @param stream CUDA stream for async execution (nullptr for default stream)
 * @return Letterbox scale factor (needed for post-processing coordinate adjustment)
 */
float cuda_preprocess(const uint8_t *src, int src_w, int src_h, float *dst,
                      int dst_w, int dst_h, bool src_on_gpu = false,
                      cudaStream_t stream = nullptr);

/**
 * @brief Preprocess 4-channel BGRA GPU image for DLA/GPU inference
 *
 * Handles 4-channel BGRA input on GPU memory (alpha channel ignored).
 */
float cuda_preprocess_bgra(const uint8_t *src, int src_w, int src_h, float *dst,
                           int dst_w, int dst_h, cudaStream_t stream = nullptr);

// FP16 Preprocessing (for DLA input)

/**
 * @brief Preprocess BGR image with FP16 output
 * @param dst Destination buffer - must be FP16 (__half*) cast to void*
 */
float cuda_preprocess_fp16(const uint8_t *src, int src_w, int src_h,
                           void *dst,
                           int dst_w, int dst_h, bool src_on_gpu = false,
                           cudaStream_t stream = nullptr);

/**
 * @brief Preprocess 4-channel BGRA GPU image with FP16 output
 * @param dst Destination buffer - must be FP16 (__half*) cast to void*
 */
float cuda_preprocess_bgra_fp16(const uint8_t *src, int src_w, int src_h,
                                void *dst,
                                int dst_w, int dst_h,
                                cudaStream_t stream = nullptr);

// HWC4 Preprocessing (DLA-native input format)

/**
 * @brief Preprocess 4-channel BGRA GPU image to HWC4 FP16 output (DLA native format)
 *
 * Converts BGRA HWC input directly to RGBA HWC4 FP16, eliminating the CHW transpose.
 * Use with --inputIOFormats=fp16:dla_hwc4 when building TensorRT engine.
 *
 * @param dst Destination buffer - FP16 HWC4 format (__half*), size = dst_w * dst_h * 4 * 2 bytes
 */
float cuda_preprocess_bgra_hwc4_fp16(const uint8_t *src, int src_w, int src_h,
                                     void *dst,
                                     int dst_w, int dst_h,
                                     cudaStream_t stream = nullptr);

void cuda_preprocess_get_offsets(int &x_offset, int &y_offset);

} // namespace unina
