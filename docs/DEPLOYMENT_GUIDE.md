# Deployment Guide (NVIDIA Jetson Orin / ROS 2 Humble)

> [!WARNING]
> **Hardware Testing Notice**: The deployment software (standalone C++ inference pipeline and ROS 2 Humble node) and actual DLA hardware residency have not been empirically tested on physical NVIDIA Jetson Orin hardware due to a lack of target hardware access.

Step-by-step instructions for exporting, compiling, and deploying UNINA-DLA on the NVIDIA Jetson Orin SoC (AGX, NX, Nano) targeting DLA Core 0 with zero GPU fallback.

---

## 1. Pipeline Overview

The deployment workflow consists of four stages:

1. **Train & Export (PC / Workstation)**: Train the student model (standalone or distillation), fine-tune with 2:4 structured sparsity and QAT, then export a DLA-compliant ONNX graph (`.onnx`).
2. **Transfer**: Copy the exported `.onnx` and configuration to the Jetson Orin workspace.
3. **Compile (Jetson Orin)**: Compile the ONNX model into a serialized TensorRT engine targeting DLA Core 0 with INT8 precision (`.engine`).
4. **Execute**: Run perception either as a real-time ROS 2 Humble node or as a standalone zero-copy C++ executable.

---

## 2. Step-by-Step Instructions

### Step 1: Train & Export (Workstation)

Execute the training and optimization pipeline:

```bash
# 1. Phase 1: Standalone Student Training (Gen 2.5 default)
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --epochs 100 \
  --batch 16 \
  --lr 0.01

# Note: Knowledge Distillation is optionally available (experimental):
# python scripts/train_student.py --data configs/formula_student.yaml \
#   --teacher path/to/teacher.pt --enable_distillation --epochs 100 --batch 16

# 2. Phase 1.5: 2:4 Structured Sparsity Fine-Tuning
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --resume runs/exp/best.pth \
  --skip_distillation \
  --enable_sparsity \
  --epochs_sparsity 20 \
  --batch 16

# 3. Phase 2: Quantization-Aware Training (QAT)
python scripts/train_student.py \
  --data configs/formula_student.yaml \
  --resume runs/exp/best_sparse.pth \
  --skip_distillation --skip_sparsity \
  --enable_qat \
  --epochs_qat 10 \
  --batch 16

# 4. Phase 3: Export DLA-Native ONNX model
python scripts/export_onnx.py \
  --checkpoint runs/exp/best_qat.pth \
  --output unina_dla_qat.onnx \
  --imgsz 640 \
  --qat-clean-deploy

# 5. Verify DLA Operation Compliance
python -m unina_dla.utils.verify_dla_onnx unina_dla_qat.onnx
```

**Artifacts Produced:**
- `unina_dla_qat.onnx`: Clean ONNX graph with 6-tensor output tuple `(reg3, reg4, reg5, cls3, cls4, cls5)`.
- `unina_dla_qat_manifest.json`: Architecture metadata and precision manifest.

---

### Step 2: Transfer to Jetson Orin

Copy the exported ONNX model to the Jetson Orin:

```bash
scp unina_dla_qat.onnx user@jetson_ip:~/ros2_ws/src/unina_perception_dla/
```

---

### Step 3: Compile to TensorRT DLA Engine (Jetson Orin)

On the target Jetson Orin, compile the ONNX graph into a TensorRT engine targeting DLA Core 0:

```bash
# Automated verification and build:
./scripts/verify_dla.sh unina_dla_qat.onnx 640 640 1

# Or invoke trtexec directly:
trtexec --onnx=unina_dla_qat.onnx \
        --saveEngine=unina_dla_qat_dla0.engine \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --inputIOFormats=fp16:chw \
        --outputIOFormats=fp16:chw \
        --shapes=images:1x3x640x640 \
        --profilingVerbosity=detailed \
        --noDataTransfers \
        --exportLayerInfo=layer_info.json
```

**Verification Check:** Review `layer_info.json` to verify that all convolution layers map to DLA with zero GPU fallback layers.

---

### Step 4: Standalone C++ Execution (No ROS)

For minimal latency and standalone verification:

```bash
cd deployment
mkdir -p build && cd build

# Standalone build (no ROS 2, no ZED SDK required)
cmake -DUSE_ROS=OFF -DUSE_ZED=OFF .. && make -j$(nproc)

# Test single image
./unina_perception ../unina_dla_qat_dla0.engine sample_cone.jpg

# Test offline dataset directory
./unina_perception_offline ../unina_dla_qat_dla0.engine /path/to/dataset_frames/ --dir

# Standalone with live ZED camera (requires ZED SDK)
cmake -DUSE_ROS=OFF -DUSE_ZED=ON .. && make -j$(nproc)
./unina_perception_zed ../unina_dla_qat_dla0.engine ../config.yaml
```

---

### Step 5: ROS 2 Humble Integration

Deploy as a real-time node within a ROS 2 workspace:

```bash
# 1. Symlink or copy deployment package into your ROS 2 workspace
cd ~/ros2_ws/src
ln -s /path/to/UNINA-DLA-26/deployment unina_perception_dla

# 2. Build with colcon
cd ~/ros2_ws
colcon build --symlink-install --packages-select unina_perception_dla \
  --cmake-args -DUSE_ZED=ON -DUSE_ROS=ON
source install/setup.bash

# 3. Launch the perception node
ros2 run unina_perception_dla unina_perception_ros \
  --ros-args -p engine_path:=~/ros2_ws/src/unina_perception_dla/unina_dla_qat_dla0.engine \
             -p config_path:=~/ros2_ws/src/unina_perception_dla/config.yaml
```

**Published Topics:**
- `/perception/cones_viz` (`visualization_msgs/msg/MarkerArray`): 3D bounding box markers for RViz2 / Foxglove.
- `/perception/cones`: 3D cone coordinates for SLAM and state estimation.
