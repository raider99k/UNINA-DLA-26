# VM Build Instructions (Non-Jetson) - ROS 2 Humble & Standalone

> [!WARNING]
> **Hardware Testing Notice**: The deployment code has not been tested on physical embedded hardware due to a lack of target hardware access.

If you are building this package on a standard Ubuntu workstation or VM (x86_64) with an NVIDIA GPU (e.g., T4 / RTX 30xx/40xx / A100), follow these instructions to set up the build environment.

## 1. Prerequisites
- **CUDA Toolkit** (11.8+ / 12.x recommended)
- **TensorRT** (8.6+ recommended)
- **OpenCV** (4.x)
- **CMake** (3.18+)
- **ROS 2 Humble** (Optional, required only for ROS 2 node)
- **ZED SDK 4.x** (Optional, required only when building with live ZED camera support `-DUSE_ZED=ON`)

---

## 2. CUDA Architecture Configuration
`deployment/CMakeLists.txt` defaults to `sm_87` (Jetson Orin) if unspecified. When compiling on a non-Orin GPU, supply `-DCMAKE_CUDA_ARCHITECTURES=native` (or your specific compute capability, e.g. `75` for T4, `86` for RTX 30xx, `89` for RTX 40xx). No manual edits to `CMakeLists.txt` are required.

---

## 3. Option A: Standalone C++ Build (No ROS, Zero-Copy)

For benchmarking, engine verification, and offline dataset testing without ROS 2:

```bash
cd deployment
mkdir -p build && cd build

# Build standalone executables (unina_perception and unina_perception_offline)
cmake -DUSE_ROS=OFF -DUSE_ZED=OFF -DCMAKE_CUDA_ARCHITECTURES=native ..
make -j$(nproc)

# Test single image
./unina_perception /path/to/engine.engine /path/to/test.jpg

# Test directory of images
./unina_perception_offline /path/to/engine.engine /path/to/images/ --dir
```

---

## 4. Option B: ROS 2 Humble Colcon Build

Assuming a standard colcon workspace at `~/ros2_ws`:

```bash
# 1. Link deployment package into workspace
mkdir -p ~/ros2_ws/src
ln -s /path/to/UNINA-DLA-26/deployment ~/ros2_ws/src/unina_perception_dla

# 2. Build deployment package
cd ~/ros2_ws

# If ZED SDK 4.x is installed (builds live ROS 2 node unina_perception_ros):
colcon build --symlink-install --packages-select unina_perception_dla \
  --cmake-args -DUSE_ROS=ON -DUSE_ZED=ON -DCMAKE_CUDA_ARCHITECTURES=native

# If ZED SDK is not installed (builds standalone and offline tools under ROS 2):
colcon build --symlink-install --packages-select unina_perception_dla \
  --cmake-args -DUSE_ROS=ON -DUSE_ZED=OFF -DCMAKE_CUDA_ARCHITECTURES=native

# 3. Source environment
source install/setup.bash
```

---

## 5. Troubleshooting
- **Missing `nvinfer` / `nvparsers`**: Ensure TensorRT libraries are in `LD_LIBRARY_PATH` (e.g., `export LD_LIBRARY_PATH=/usr/local/TensorRT/lib:$LD_LIBRARY_PATH`).
- **CUDA Architecture Mismatch**: Pass `-DCMAKE_CUDA_ARCHITECTURES=native` so nvcc targets your host GPU directly.
- **ZED Header Missing**: If you do not have ZED SDK installed, pass `-DUSE_ZED=OFF`.
