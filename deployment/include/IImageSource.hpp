#pragma once

/**
 * @file IImageSource.hpp
 * @brief Abstract interface for image sources
 *
 * This interface allows the perception pipeline to work with:
 * - Pre-recorded image directories
 * - Video files
 * - Live camera streams or synthetic test data
 */

#include <cstdint>
#include <memory>
#include <string>

namespace unina {

/**
 * @brief Image data container
 *
 * Holds image data that can be either on CPU or GPU.
 */
struct ImageFrame {
  // Image dimensions
  int width = 0;
  int height = 0;
  int channels = 0; // 3=BGR, 4=BGRA

  // Data pointers (one or both may be valid)
  const uint8_t *cpu_data = nullptr; ///< CPU pointer (if available)
  const uint8_t *gpu_data = nullptr; ///< GPU pointer (if available)

  // Timestamp
  uint64_t timestamp_ns = 0;
  int frame_id = 0;

  // Flags
  bool is_valid = false;
  bool data_on_gpu = false; ///< True if primary data is on GPU

  /**
   * @brief Get the appropriate data pointer
   */
  const uint8_t *data() const { return data_on_gpu ? gpu_data : cpu_data; }
};

/**
 * @brief Abstract interface for image sources
 *
 * Implement this interface to provide images from any source.
 */
class IImageSource {
public:
  virtual ~IImageSource() = default;

  /**
   * @brief Initialize the image source
   * @return true on success
   */
  virtual bool init() = 0;

  /**
   * @brief Get the next frame
   * @param frame Output frame (filled on success)
   * @return true if a frame was retrieved
   */
  virtual bool grab(ImageFrame &frame) = 0;

  /**
   * @brief Check if more frames are available
   * @return true if not at end of stream
   */
  virtual bool has_more() const = 0;

  /**
   * @brief Get image width
   */
  virtual int width() const = 0;

  /**
   * @brief Get image height
   */
  virtual int height() const = 0;

  /**
   * @brief Get number of channels (3 or 4)
   */
  virtual int channels() const = 0;

  /**
   * @brief Check if source provides GPU data directly
   */
  virtual bool provides_gpu_data() const = 0;

  /**
   * @brief Release resources
   */
  virtual void close() = 0;
};

} // namespace unina
