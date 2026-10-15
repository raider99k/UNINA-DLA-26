# UNINA-DLA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-brightgreen.svg)](https://docs.ros.org/en/humble/)
[![Target: Jetson Orin](https://img.shields.io/badge/Hardware-Jetson_Orin-76B900.svg)](https://developer.nvidia.com/embedded/jetson-orin)

UNINA-DLA is a real-time object detection framework designed for complete execution on the NVIDIA Jetson Orin Deep Learning Accelerator (NVDLA 2.0). 

By using structurally re-parameterized convolutions (RepVGG), a memory-bounded path aggregation network (Rep-PAN), and an anchor-free One-to-One (O2O) direct regression head, UNINA-DLA offloads 2D object detection entirely to DLA hardware. This eliminates GPU fallback layers, providing low, deterministic latency while freeing the GPU for SLAM, point-cloud processing, and control tasks.

---

## Architecture Overview

- **Backbone**: RepVGG-A1 (adapted). Employs a multi-branch topology during training that mathematically folds into single-path $3 \times 3$ convolutions and ReLUs for inference. Follows the RepVGG-A1 `[2, 4, 14, 1]` layout with Stage 4 at 512 channels. In **Gen 2.6**, standard un-grouped convolutions (`groups=1`) are utilized throughout, providing 100% ImageNet pretraining coverage across all 4 stages while requiring only 3 out of 16 DLA CBUF SRAM banks. (Gen 2.5 legacy used `groups=2`).
- **Activation Functions**: Uniform ReLU. Executes in 1 cycle on DLA Single Data Point (SDP) hardware, avoiding FP16 precision drops.
- **Neck**: Rep-PAN with Small-Object Detection (SOD) priority channel sizing (`[256, 192, 128]`). Uses standard convolutions (`groups=1` in Gen 2.6) with deconvolutional upsampling (`DLAUpsample`).
- **Detection Head**: Decoupled dual-head during training (O2O target head + scheduled auxiliary O2M head). Direct coordinate regression outputs 4 linear box offsets, removing Non-Maximum Suppression (NMS), Softmax, and Distribution Focal Loss (DFL).
- **Deployment Output**: Exports a 6-tensor output tuple `(reg3, reg4, reg5, cls3, cls4, cls5)` with class Sigmoids applied, directly consumable by zero-copy C++ TensorRT decoders.
- **Model Compression**: 2:4 Structured Sparsity and Quantization-Aware Training (INT8 QAT) with sensitive prediction heads preserved in FP16.

For full architectural derivations, hardware proofs, and loss formulations, see [**`docs/ARCHITECTURE.md`**](docs/ARCHITECTURE.md).

---

## Benchmark Highlights

Evaluated on the Formula Student Objects in Context (FSOCO) validation split (1,157 images, input resolution $480 \times 768$, batch size 1):

| Model | Target / Precision | Size (MB) | mAP50 | Latency (RTX 6000) | Throughput (RTX 6000) | Latency (Tesla T4) | Throughput (Tesla T4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT (Gen 2.6 Champion)** | TensorRT INT8 | **25.1** | **84.3%** | **0.46 ms** | **1,378.0 FPS** | - | - |
| **UNINA-Sparse (Gen 2.6 Champion)** | TensorRT FP16 | **38.0** | **88.2%** | **0.55 ms** | **1,223.9 FPS** | - | - |
| **UNINA-QAT (Gen 2.5 Pretrained)** | TensorRT INT8 | **22.1** | **83.1%** | **0.51 ms** | **1,294.3 FPS** | **2.77 ms** | **266.2 FPS** |
| **UNINA-Sparse (Gen 2.5 Pretrained)** | TensorRT FP16 | 36.4 | 81.6% | 0.56 ms | 1,225.7 FPS | 4.18 ms | 193.6 FPS |
| **UNINA-QAT (Gen 2.5 Scratch)** | TensorRT INT8 | 22.1 | 80.7% | 0.51 ms | 1,294.3 FPS | 2.77 ms | 266.2 FPS |
| **UNINA-QAT (Gen 1 Prototype)** | TensorRT INT8 | 19.8 | 80.7% | 0.47 ms | 1,359.8 FPS | - | - |
| **YOLO26N-Engine** | TensorRT FP16 | 11.8 | 87.0% | 0.89 ms | 954.2 FPS | 3.35 ms | 247.1 FPS |
| **YOLO26S-Engine** | TensorRT FP16 | 38.9 | 90.8% | 1.32 ms | 681.1 FPS | 8.08 ms | 113.9 FPS |
| **YOLO26X-Engine** | TensorRT FP16 | 215.5 | 93.6% | 4.50 ms | 214.9 FPS | 48.07 ms | 20.5 FPS |

> [!NOTE]
> **Gen 2.6 Production Champion (150-Epoch Schedule)**: Combines 100% RepVGG-A1 pretraining across Stages 0–4 (`groups=1`) with transposed convolution upsampling, achieving **84.3% mAP50** (and **89.3% mAP50** at operational deployment threshold `conf=0.25`) at an ultra-fast **0.46 ms** inference latency (**1,378.0 FPS**), running 1.93x faster than YOLO26N-Engine while operating within only 3/16 DLA CBUF SRAM banks.

For complete benchmark tables, per-class metrics, and execution profiles, see [**`docs/BENCHMARK_REPORT.md`**](docs/BENCHMARK_REPORT.md).

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/raider99k/UNINA-DLA-26.git
cd UNINA-DLA-26

# Install dependencies and package
pip install --extra-index-url https://pypi.nvidia.com -r requirements.txt
pip install -e .
```

### 2. Training Pipeline

UNINA-DLA uses a 3-phase optimization pipeline:

```bash
# Phase 1: Standalone Student Training (Generation 2.5 default, with optional RepVGG-A1 pretrained backbone)
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --pretrained_backbone repvgg_a1 \
  --epochs 100 \
  --batch 16 \
  --lr 0.01

# Phase 1.5: 2:4 Structured Sparsity Fine-Tuning
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --resume runs/exp/best.pth \
  --skip_distillation \
  --enable_sparsity \
  --epochs_sparsity 20 \
  --batch 16

# Phase 2: Quantization-Aware Training (INT8 QAT)
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --resume runs/exp/best_sparse.pth \
  --skip_distillation --skip_sparsity \
  --enable_qat \
  --epochs_qat 10 \
  --batch 16

# Phase 3: Export DLA-Compliant ONNX
python scripts/export_onnx.py \
  --checkpoint runs/exp/best_qat.pth \
  --output unina_dla_qat.onnx \
  --imgsz 640 \
  --qat-clean-deploy

# Verify DLA 2.0 Compliance (0 forbidden ops, 0 GPU fallbacks)
python -m unina_dla.utils.verify_dla_onnx unina_dla_qat.onnx
```

> **Note on Distillation**: Distillation from a YOLO26 teacher model is available as an experimental option (`--enable_distillation`). Standalone training with dynamic auxiliary loss decay is recommended as it yields sharper single-positive localization and higher final mAP. Detailed training analysis is documented in [**`docs/TRAINING_REPORT.md`**](docs/TRAINING_REPORT.md).

---

## Deployment

> [!WARNING]
> **Hardware Testing Notice**: The deployment software (C++ standalone and ROS 2 Humble node) and actual DLA hardware residency have not been empirically tested on physical NVIDIA Jetson Orin hardware due to a lack of target hardware access.

### 1. TensorRT Engine Compilation (on Jetson Orin)

```bash
# Automated DLA compilation and verification:
./scripts/verify_dla.sh unina_dla_qat.onnx 640 640 1

# Or with trtexec:
trtexec --onnx=unina_dla_qat.onnx \
        --saveEngine=unina_dla_qat_dla0.engine \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --inputIOFormats=fp16:chw \
        --outputIOFormats=fp16:chw \
        --shapes=images:1x3x640x640 \
        --exportLayerInfo=layer_info.json
```

### 2. Standalone C++ Execution

```bash
cd deployment
mkdir -p build && cd build
cmake -DUSE_ROS=OFF -DUSE_ZED=OFF .. && make -j$(nproc)
./unina_perception ../unina_dla_qat_dla0.engine sample_cone.jpg
```

### 3. ROS 2 Humble Integration

```bash
cd ~/ros2_ws/src
ln -s /path/to/UNINA-DLA-26/deployment unina_perception_dla
cd ~/ros2_ws
colcon build --symlink-install --packages-select unina_perception_dla --cmake-args -DUSE_ZED=ON -DUSE_ROS=ON
source install/setup.bash

ros2 run unina_perception_dla unina_perception_ros \
  --ros-args -p engine_path:=/path/to/unina_dla_qat_dla0.engine \
             -p config_path:=/path/to/config.yaml
```

For full setup instructions, see [**`docs/DEPLOYMENT_GUIDE.md`**](docs/DEPLOYMENT_GUIDE.md).

---

## Python Inference via Ultralytics API

An Ultralytics-compatible inference wrapper is provided for quick validation and testing:

```bash
# Convert checkpoint to .pt format
python scripts/convert_to_ultralytics_pt.py \
  --checkpoint runs/exp/best_qat.pth \
  --output runs/exp/unina_dla.pt \
  --num-classes 4 \
  --names unina_dla/config/unina_dla_data.yaml

# Run inference
python scripts/infer_ultralytics.py \
  --model runs/exp/unina_dla.pt \
  --source path/to/image_or_video \
  --device 0
```

For full wrapper options, see [**`docs/ULTRALYTICS_INFERENCE.md`**](docs/ULTRALYTICS_INFERENCE.md).

---

## Repository Structure

```
UNINA-DLA-26/
├── configs/                      # Dataset and benchmark configurations
│   ├── benchmark_fsd.yaml        # Benchmark suite configuration
│   └── formula_student.yaml      # Formula Student reference configuration
├── deployment/                   # C++ TensorRT deployment (Standalone & ROS 2)
│   ├── CMakeLists.txt            # Modular CMake build
│   ├── VM_BUILD.md               # Workstation / VM compilation guide
│   ├── config.yaml               # Deployment configuration
│   ├── include/                  # C++ headers (Perception3D, Cone3D, CUDA kernels)
│   └── src/                      # C++ implementation files
├── docs/                         # Technical documentation
│   ├── ARCHITECTURE.md           # Architecture specifications, math proofs, DLA constraints
│   ├── BENCHMARK_REPORT.md       # Multi-platform benchmark suite results
│   ├── DEPLOYMENT_GUIDE.md       # Jetson Orin deployment & ROS 2 setup
│   ├── TRAINING_REPORT.md        # Multi-phase training dynamics and generation comparison
│   └── ULTRALYTICS_INFERENCE.md  # Ultralytics Python inference wrapper guide
├── scripts/                      # Workflow and utility scripts
│   ├── benchmark.py              # Unified benchmark suite
│   ├── build_engine.py           # TensorRT engine builder with metadata embedding
│   ├── convert_to_ultralytics_pt.py # Convert training checkpoints to Ultralytics .pt
│   ├── dla_utils.py              # DLA CLI and engine utilities
│   ├── embed_engine_metadata.py  # Embed metadata into TensorRT engines
│   ├── export_onnx.py            # DLA-compliant ONNX graph exporter
│   ├── infer_ultralytics.py      # Ultralytics-style inference runner
│   ├── train_student.py          # Student training pipeline (Phase 1, 1.5, 2)
│   ├── train_teacher.py          # Optional teacher model pretraining
│   ├── val.py                    # Model validation script
│   └── verify_dla.sh             # DLA compilation and layer verification script
└── unina_dla/                    # Core Python package
    ├── benchmark/                # Benchmark adapters, metrics, and reporting
    ├── config/                   # Default dataset configuration YAMLs
    ├── engines/                  # Training engines (Phase 1, Sparsity, QAT, Export)
    ├── model/                    # RepVGG backbone, Rep-PAN neck, YOLO26 head, losses
    ├── ultralytics_wrapper.py    # Ultralytics DetectionPredictor wrapper
    └── utils/                    # Box operations, metrics, optimizers, DLA verifier
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
