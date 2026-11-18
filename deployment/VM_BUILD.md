# Workstation / VM Build Instructions (Non-Jetson)

> [!WARNING]
> **Hardware Testing Notice**: The deployment code has not been tested on physical embedded hardware due to a lack of target hardware access.

If you are building this package on a standard Linux workstation, VM, or cloud GPU environment (e.g., T4 / RTX 30xx/40xx / A100), follow these instructions to set up the build environment.

## 1. Prerequisites
- **CUDA Toolkit** (11.8+ / 12.x recommended)
- **TensorRT** (8.6+ or 10.x)
- **OpenCV** (4.x)
- **CMake** (3.18+)

---

## 2. CUDA Architecture Configuration
`deployment/CMakeLists.txt` defaults to `sm_87` (Jetson Orin) if unspecified. When compiling on a non-Orin GPU, supply `-DCMAKE_CUDA_ARCHITECTURES=native` (or your specific compute capability, e.g. `75` for T4, `86` for RTX 30xx, `89` for RTX 40xx). No manual edits to `CMakeLists.txt` are required.

---

## 3. Standalone C++ Build

For benchmarking, engine verification, and offline image/video testing:

```bash
cd deployment
mkdir -p build && cd build

# Build standalone executables (unina_perception and unina_perception_offline)
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
make -j$(nproc)

# Test single image
./unina_perception /path/to/engine.engine /path/to/test.jpg

# Test directory of images or video
./unina_perception_offline /path/to/engine.engine /path/to/images/ --dir --output ./results/
```

---

## 4. Troubleshooting
- **Missing `nvinfer`**: Ensure TensorRT libraries are in `LD_LIBRARY_PATH` (e.g., `export LD_LIBRARY_PATH=/usr/local/TensorRT/lib:$LD_LIBRARY_PATH`).
- **CUDA Architecture Mismatch**: Pass `-DCMAKE_CUDA_ARCHITECTURES=native` so `nvcc` targets your host GPU directly.
