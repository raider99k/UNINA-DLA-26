# Deployment Guide (NVIDIA Jetson Orin & Standalone TensorRT)

> [!WARNING]
> **Hardware Testing Notice**: The standalone C++ inference pipeline and actual DLA hardware residency have not been empirically tested on physical NVIDIA Jetson Orin hardware due to a lack of target hardware access.

Step-by-step instructions for exporting, compiling, and deploying UNINA-DLA on the NVIDIA Jetson Orin SoC (AGX, NX, Nano) targeting DLA Core 0 with zero GPU fallback, or on a standard Linux GPU host.

---

## 1. Pipeline Overview

The deployment workflow consists of four stages:

1. **Train & Export (PC / Workstation)**: Train the student model (standalone or distillation), fine-tune with 2:4 structured sparsity and QAT, then export a DLA-compliant ONNX graph (`.onnx`).
2. **Transfer**: Copy the exported `.onnx` (and `*_int8_scales.json` if using `qat-clean-deploy`) to the target machine.
3. **Compile (Jetson Orin / Target GPU)**: Compile the ONNX model into a serialized TensorRT engine targeting DLA Core 0 or GPU (`.engine`).
4. **Execute**: Run standalone zero-copy C++/CUDA inference (`unina_perception` or `unina_perception_offline`).

---

## 2. Step-by-Step Instructions

### Step 1: Train & Export (Workstation)

Execute the training and optimization pipeline (replace `--data` with your own YOLO dataset YAML if not using the FSOCO/Formula Student showcase):

```bash
# 1. Phase 1: Standalone Student Training (Gen 3.2 default)
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
- `unina_dla_qat_int8_scales.json`: INT8 QAT scale factors (used automatically by `scripts/build_engine.py`).
- `unina_dla_qat_manifest.json`: Architecture metadata and precision manifest.

---

### Step 2: Transfer to Target

Copy the exported ONNX model (and `_int8_scales.json` if present) to the target device:

```bash
scp unina_dla_qat.onnx unina_dla_qat_int8_scales.json user@jetson_ip:~/UNINA-DLA-26/models/
```

---

### Step 3: Compile to TensorRT DLA Engine (Jetson Orin)

On the target Jetson Orin, compile the ONNX graph into a TensorRT engine targeting DLA Core 0:

```bash
# Automated build (picks up *_int8_scales.json automatically):
python scripts/build_engine.py --onnx models/unina_dla_qat.onnx --target dla

# Or automated verification and build via trtexec:
./scripts/verify_dla.sh models/unina_dla_qat.onnx 640 640 1
```

**Verification Check:** Review `layer_info.json` to verify that all convolution layers map to DLA with zero GPU fallback layers.

---

### Step 4: Standalone C++ Execution

For minimal-latency standalone C++/CUDA inference (on non-Orin GPUs, pass `-DCMAKE_CUDA_ARCHITECTURES=native` per [deployment/VM_BUILD.md](../deployment/VM_BUILD.md)):

```bash
cd deployment
mkdir -p build && cd build

cmake .. && make -j$(nproc)

# Test single image
./unina_perception ../../models/unina_dla_qat_dla0.engine sample_image.jpg

# Test offline image directory or video
./unina_perception_offline ../../models/unina_dla_qat_dla0.engine /path/to/dataset_frames/ --dir
```
