#pragma once

#include "common.hpp"
#include <cuda_runtime.h>

namespace unina {

/**
 * @brief CUDA Postprocessing for Zero-Overhead Inference
 *
 * Performs parallel decode of DLA outputs on GPU, avoiding CPU iteration
 * over 25,200+ grid cells. Uses atomic operations for variable-length output.
 */

// Maximum detections per inference (prevents buffer overflow)
constexpr int kMaxDetections = 1000;

/**
 * @brief Initialize postprocessing resources (allocates device memory)
 * @return cudaError_t CUDA error code
 */
cudaError_t cuda_postprocess_init();

/**
 * @brief Release postprocessing resources
 */
void cuda_postprocess_destroy();

/**
 * @brief Decode DLA outputs on GPU (main entry point)
 *
 * @param cls_buffers Array of 3 class probability buffers [C,H,W] per scale
 * @param reg_buffers Array of 3 regression buffers [4,H,W] per scale (LTRB)
 * @param num_scales Number of detection scales (typically 3)
 * @param strides Array of stride values per scale
 * @param input_w Model input width (e.g., 1024)
 * @param input_h Model input height (e.g., 640)
 * @param conf_threshold Confidence threshold for filtering
 * @param num_classes Number of object classes
 * @param use_nms Enable GPU class-wise NMS after decode
 * @param score_threshold Score threshold used during NMS candidate filtering
 * @param nms_iou_threshold IoU threshold for class-wise suppression
 * @param max_detections Maximum detections returned
 * @param is_fp16 True if buffers contain FP16 data
 * @param stream CUDA stream for async execution
 * @return Vector of detections (copied to CPU)
 */
std::vector<Detection>
cuda_decode(const void *const *cls_buffers, const void *const *reg_buffers,
            const OutputTensorLayout *cls_layouts,
            const OutputTensorLayout *reg_layouts, int num_scales,
            const int *strides, int input_w, int input_h,
            float conf_threshold, int num_classes, bool use_nms = false,
            float score_threshold = 0.45f, float nms_iou_threshold = 0.60f,
            int max_detections = kMaxDetections, bool is_fp16 = false,
            cudaStream_t stream = nullptr);

/**
 * @brief Adjust detection coordinates for letterbox preprocessing
 *
 * @param detections Vector of detections to adjust (modified in-place)
 * @param x_offset Letterbox horizontal offset
 * @param y_offset Letterbox vertical offset
 * @param scale Letterbox scale factor
 * @param orig_w Original image width
 * @param orig_h Original image height
 */
void adjust_detections_for_letterbox(std::vector<Detection> &detections,
                                     int x_offset, int y_offset, float scale,
                                     int orig_w, int orig_h);

} // namespace unina
