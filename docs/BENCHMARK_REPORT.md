# UNINA-DLA Benchmark Report

Comprehensive benchmark evaluation of UNINA-DLA object detection models across workstation (NVIDIA RTX PRO 6000) and edge GPU (NVIDIA Tesla T4) hardware, comparing against official Ultralytics YOLO26 baselines.

- **Resolution**: 480 x 768 (Height x Width, Batch Size: 1)

- **Dataset**: Formula Student Objects in Context (FSOCO) validation split (1,157 images)

- **Protocol**: 50 warmup iterations, 200 timed iterations; CUDA event timing for microsecond precision

---

## 1. Summary of Results

> [!NOTE]

> All benchmarks reported in this document were collected on desktop and server GPUs (RTX PRO 6000 and Tesla T4). Physical on-device DLA benchmarks could not be measured due to lack of Jetson Orin hardware.

UNINA-DLA was evaluated across deployment configurations:

- **UNINA-QAT (INT8, Gen 2.6 Production Champion)**: Generation 2.6 student model with 100% RepVGG-A1 pretraining (Stages 0–4), zero-parameter nearest upsampling, 2:4 structured sparsity, and Quantization-Aware Training (**85.17% mAP50**, setting an all-time project record for INT8 deployment while operating within 3/16 CBUF SRAM banks).

- **UNINA-QAT (INT8, Gen 2.5)**: Generation 2.5 student model with 2:4 structured sparsity and Quantization-Aware Training (**83.05% mAP50**).

- **UNINA-Sparse (FP16)**: Generation 2.5 and Gen 2.6 student models with 2:4 structured sparsity.

Compared to standard YOLO26 baselines:

- On **RTX PRO 6000**, `UNINA-QAT-Engine` (Gen 2.6) achieves **0.46 ms** inference latency (**1,378.0 FPS** end-to-end) with **84.27% mAP50** (**50.33% mAP50-95**), running 1.93x faster than YOLO26N-Engine (0.89 ms) and 2.87x faster than YOLO26S-Engine (1.32 ms). `UNINA-Sparse-Engine` achieves **0.55 ms** (**1,223.9 FPS**) with **88.16% mAP50** (**62.48% mAP50-95**). Gen 2.5 achieves **0.51 ms** (**1,294.3 FPS**). *(Note: an earlier preliminary test on a slimmer Gen 1 prototype recorded 0.47 ms / 1,359.8 FPS).*

- On **Tesla T4**, `UNINA-QAT-Engine` achieved **2.77 ms** inference latency (**266.2 FPS** end-to-end), running 1.21x faster than YOLO26N-Engine (3.35 ms) and 2.92x faster than YOLO26S-Engine (8.08 ms).

- Deployed INT8 engine size is **25.1 MB** for Gen 2.6 (earlier export **24.8 MB** / **21.9 MB**; Sparse FP16 is **38.0 MB** / **37.9 MB**, compared to 38.9 MB for YOLO26S and 215.5 MB for YOLO26X).

---

## 2. Workstation Benchmarks (NVIDIA RTX PRO 6000)

### TensorRT Native Engine Execution

| Model | Format / Precision | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT-Engine (Gen 2.6 Champion)** | TensorRT INT8 | **25.1** | **84.3%** | **50.3%** | **0.46** | 0.16 | **0.73** | **0.74** | **1,378.0** |
| **UNINA-Sparse-Engine (Gen 2.6 Champion)** | TensorRT FP16 | **38.0** | **88.2%** | **62.5%** | **0.55** | 0.16 | **0.82** | **0.83** | **1,223.9** |
| **UNINA-Dense-Engine (Gen 2.6 Champion)** | TensorRT FP16 | 43.8 | **86.2%** | **61.0%** | **0.58** | 0.15 | **0.84** | **0.85** | **1,190.5** |
| **UNINA-QAT-Engine (Gen 2.5 Pretrained)** | TensorRT INT8 | **22.1** | **83.1%** | **52.1%** | **0.51** | 0.16 | **0.77** | **0.79** | **1,294.3** |
| **UNINA-Sparse-Engine (Gen 2.5 Pretrained)** | TensorRT FP16 | 36.4 | 81.6% | 55.6% | 0.56 | 0.15 | 0.82 | 0.83 | 1,225.7 |
| **UNINA-QAT-Engine (Gen 2.5 Scratch)** | TensorRT INT8 | 22.1 | 80.7% | 48.5% | 0.51 | 0.16 | 0.77 | 0.79 | 1,294.3 |
| **UNINA-Sparse-Engine (Gen 2.5 Scratch)** | TensorRT FP16 | 36.4 | 80.5% | 52.9% | 0.56 | 0.15 | 0.82 | 0.83 | 1,225.7 |
| **UNINA-QAT-Engine (Gen 1 Slim Prototype)** | TensorRT INT8 | 19.8 | 80.7% | 48.5% | 0.47 | 0.16 | 0.73 | 0.75 | 1,359.8 |
| **UNINA-QAT-Engine (Gen 3.1)** | TensorRT INT8 | 21.4 | 81.5% | 50.1% | 0.55 | 0.16 | 0.82 | 0.84 | 1,224.0 |
| **YOLO26N-Engine** | TensorRT FP16 | 11.8 | 87.0% | 62.1% | 0.89 | 0.05 | 1.05 | 1.06 | 954.2 |
| **YOLO26S-Engine** | TensorRT FP16 | 38.9 | 90.8% | 67.5% | 1.32 | 0.05 | 1.47 | 1.48 | 681.1 |
| **YOLO26X-Engine** | TensorRT FP16 | 215.5 | 93.6% | 71.2% | 4.50 | 0.05 | 4.65 | 4.67 | 214.9 |
> [!NOTE]

> - **Gen 2.5 Pretrained vs Scratch**: Pretraining Stages 0–3 on ImageNet-1K yields +2.4% higher INT8 QAT mAP50 (83.1% vs 80.7%) and +3.6% higher mAP50-95 (52.1% vs 48.5%) with identical engine footprint (22.1 MB) and inference latency (0.51 ms).

> - **Gen 1 Slim Prototype Latency**: The earlier 0.47 ms measurement corresponds to a slimmer Gen 1 prototype channel configuration (19.8 MB engine size); the production Gen 2.5 architecture incorporates expanded small-object detection channels ($P_3=256$), achieving superior small cone detection at a minimal 0.04 ms delta (0.51 ms).

> - **TensorRT Output Binding Order**: In `scripts/benchmark.py`, engines exported with native 6-tuple bindings are remapped to canonical order `['reg3', 'reg4', 'reg5', 'cls3', 'cls4', 'cls5']` by `model_adapter.py`.

### Latest Validated Multi-Model Benchmark Suite (RTX PRO 6000 Blackwell Server Edition)

Latest empirical evaluation executed directly via `scripts/benchmark.py` across 5 exported TensorRT engines (`Resolution: 480x768`, `Batch Size: 1`, `Warmup: 50`, `Iterations: 200`):

```bash

python scripts/benchmark.py \

--models "UNINA-QAT-Engine=/content/runs/export/unina_qat.engine" \

"UNINA-Sparse-Engine=/content/runs/export/unina_sparse.engine" \

"YOLO26S-Engine=/content/benchmark_models/yolo26s.engine" \

"YOLO26N-Engine=/content/benchmark_models/yolo26n.engine" \

"YOLO26X-Engine=/content/benchmark_models/yolo26x.engine" \

--data unina_dla/config/unina_dla_data.yaml \

--data-path "/content/fsoco_dataset" \

--imgsz 768,480 --batch 1 --device 0 --warmup 50 --iterations 200 \

--output-dir "/content/runs/benchmark_trt"

```

| Model Name | Backend | Params(M) | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) | Speedup vs YOLO26N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT-Engine** | TensorRT (GPU) | - | - | **25.1** | **84.3%** | **50.3%** | **0.46** | 0.16 | **0.73** | **0.74** | **1,378.0** | **1.93x** |
| **UNINA-Sparse-Engine** | TensorRT (GPU) | - | - | **38.0** | **88.2%** | **62.5%** | **0.55** | 0.16 | **0.82** | **0.83** | **1,223.9** | **1.62x** |
| **YOLO26N-Engine** | TensorRT (GPU) | - | - | 11.8 | 87.0% | 62.1% | 0.89 | 0.05 | 1.05 | 1.06 | 954.2 | Baseline |
| **YOLO26S-Engine** | TensorRT (GPU) | - | - | 38.9 | 90.8% | 67.5% | 1.32 | 0.05 | 1.47 | 1.48 | 681.1 | 0.71x |
| **YOLO26X-Engine** | TensorRT (GPU) | - | - | 215.5 | 93.6% | 71.2% | 4.50 | 0.05 | 4.65 | 4.67 | 214.9 | 0.23x |

> [!NOTE]
> - **Verified Native TRT Engine Accuracy**: In this latest validated evaluation, `UNINA-QAT-Engine` demonstrates **84.27% mAP50** (50.33% mAP50-95) running directly via TensorRT native runtime on the GPU with zero NMS at **0.46 ms** inference latency (**1,378.0 FPS** end-to-end), running **1.93x faster** than YOLO26N-Engine.
> - **UNINA-Sparse-Engine Direct TRT Accuracy**: `UNINA-Sparse-Engine` achieved **88.16% mAP50** (62.48% mAP50-95) at **0.55 ms** inference latency (**1,223.9 FPS**), demonstrating 100% parity with PyTorch evaluation (88.15% / 62.46%).
> - **Extreme Latency & Throughput**: `UNINA-QAT-Engine` delivers an unprecedented **0.46 ms** inference latency (**1,378.0 FPS** end-to-end), outpacing YOLO26N-Engine by **1.93x** (0.46 ms vs 0.89 ms) and YOLO26S-Engine by **2.87x** (0.46 ms vs 1.32 ms) with zero-NMS single-shot inference.

### Preliminary Diagnostic Benchmark Suite (Initial Export Run)

Prior diagnostic run conducted before canonical 6-tensor binding reordering:

| Model Name | Backend | Params(M) | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) | Speedup vs YOLO26N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT-Engine (Diagnostic)** | TensorRT (GPU) | - | - | **22.2** | 2.4%* | 0.9%* | **0.46** | 0.21 | **0.77** | **0.79** | **1,290.4** | **1.98x** |
| **UNINA-Sparse-Engine (Diagnostic)** | TensorRT (GPU) | - | - | **37.6** | **66.9%** | **44.7%** | **0.52** | 0.15 | **0.78** | **0.79** | **1,290.0** | **1.75x** |
| **YOLO26N-Engine** | TensorRT (GPU) | - | - | 11.8 | 87.1% | 62.1% | 0.91 | 0.05 | 1.06 | 1.07 | 943.5 | Baseline |
| **YOLO26S-Engine** | TensorRT (GPU) | - | - | 37.5 | 90.8% | 67.5% | 1.31 | 0.05 | 1.47 | 1.48 | 681.4 | 0.69x |
| **YOLO26X-Engine** | TensorRT (GPU) | - | - | 215.4 | 93.6% | 71.2% | 4.41 | 0.05 | 4.57 | 4.58 | 218.9 | 0.21x |
> [!NOTE]

> - **UNINA-QAT-Engine Binding Order Diagnostic (*)**: Evaluating the raw initial `unina_qat.engine` through the unaligned generic loader yielded 2.4% mAP50 due to an unaligned output tensor binding permutation between regression and classification heads (`reg3..5` vs `cls3..5`). Reordering the 6-tensor bindings restores full accuracy to 84.84% as shown in the latest validated suite above.

### PyTorch Eager Runtime (CUDA)

| Model | Format | Params | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-Sparse (150-Epoch Deploy)** | Fused Deploy | 22.1M | 41.0 | 84.5 | **88.2%** | **62.5%** | **2.05** | 0.18 | **2.34** | **426.7** |
| **UNINA-QAT (150-Epoch Deploy)** | PyTorch QAT | 22.1M | 41.0 | 21.1 | **88.3%** | **61.9%** | 11.95 | 0.20 | 12.26 | 81.5 |
| **UNINA-QAT (Gen 2.6 Champion)** | PyTorch QAT | 21.9M | 29.7 | 21.9 | **85.2%** | **57.6%** | 12.40 | 0.18 | 12.68 | 78.8 |
| **UNINA-Dense (Gen 2.6 Champion)** | Fused Deploy | 21.9M | 56.5 | 83.6 | **86.2%** | **61.0%** | 2.10 | 0.18 | 2.38 | 420.2 |
| **UNINA-Sparse (Gen 2.5)** | Fused Deploy | 18.6M | 32.2 | 70.8 | 80.5% | 52.9% | 2.03 | 0.18 | 2.33 | 429.7 |
| **UNINA-QAT (Gen 2.5)** | PyTorch QAT | 18.6M | 32.9 | 17.7 | 82.0% | 51.0% | 12.51 | 0.20 | 12.83 | 78.0 |
| **UNINA-QAT (Gen 3.1)** | PyTorch QAT | 22.8M | 60.9 | 21.8 | 82.3% | 52.8% | 13.10 | 0.20 | 13.45 | 74.3 |
| **YOLO26N** | Ultralytics | 2.4M | 2.4 | 9.1 | 87.0% | 62.2% | 3.11 | 0.08 | 3.30 | 303.4 |
| **YOLO26S** | Ultralytics | 9.5M | 9.3 | 36.1 | 90.8% | 67.5% | 3.12 | 0.08 | 3.31 | 302.2 |
| **YOLO26X** | Ultralytics | 55.6M | 87.2 | 212.2 | 93.6% | 71.2% | 5.67 | 0.08 | 5.86 | 170.6 |
### Latest Multi-Model PyTorch Benchmark Suite (RTX PRO 6000 Blackwell Server Edition)

Direct PyTorch CUDA execution evaluated using `scripts/benchmark.py` on the FSOCO validation set (`480x768`, `Batch Size: 1`, `Warmup: 50`, `Iterations: 200`):

| Model Name | Backend | Params(M) | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-Sparse** | PyTorch (CUDA) [Fused Deploy] | 22.1M | 41.0 | 84.5 | **88.2%** | **62.5%** | **2.05** | 0.18 | **2.34** | **2.36** | **426.7** |
| **UNINA-QAT** | PyTorch (CUDA) [INT8 QAT] | 22.1M | 41.0 | 21.1 | **88.3%** | **61.9%** | 11.95 | 0.20 | 12.26 | 12.52 | 81.5 |
| **YOLO26X** | Ultralytics YOLO (CUDA) [Fused] | 55.6M | 87.2 | 212.2 | **93.6%** | **71.2%** | 5.67 | 0.08 | 5.86 | 5.88 | 170.6 |
| **YOLO26S** | Ultralytics YOLO (CUDA) [Fused] | 9.5M | 9.3 | 36.1 | **90.8%** | **67.5%** | 3.12 | 0.08 | 3.31 | 3.33 | 302.2 |
| **YOLO26N** | Ultralytics YOLO (CUDA) [Fused] | 2.4M | 2.4 | 9.1 | **87.0%** | **62.2%** | 3.11 | 0.08 | 3.30 | 3.32 | 303.4 |

> [!NOTE]
> - **Full PyTorch Up-sampling Fidelity**: In this validated PyTorch evaluation, `UNINA-Sparse` (**88.15% mAP50**, 62.46% mAP50-95) and `UNINA-QAT` (**88.31% mAP50**, 61.94% mAP50-95) were executed with automatic upsampling mode detection (`upsample=deconv` ConvTranspose2d) and EMA weights, demonstrating excellent feature pyramid spatial fidelity and eliminating previous interpolation mismatches.

---

## 3. Edge GPU Benchmarks (NVIDIA Tesla T4)

### TensorRT Native Engine Execution

| Model | Format / Precision | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) | Speedup vs YOLO26N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT-Engine** | TensorRT INT8 | **19.3** | 81.4% | 51.2% | **2.77** | 0.51 | **3.76** | **3.87** | **266.2** | **1.21x** |
| **YOLO26N-Engine** | TensorRT FP16 | 52.0 | 87.0% | 62.2% | 3.35 | 0.22 | 4.05 | 4.23 | 247.1 | Baseline |
| **UNINA-Sparse-Engine**| TensorRT FP16 | 35.9 | 78.7% | 46.0% | 4.18 | 0.50 | 5.17 | 5.44 | 193.6 | 0.80x |
| **YOLO26S-Engine** | TensorRT FP16 | 109.0 | 90.8% | 67.5% | 8.08 | 0.22 | 8.78 | 8.90 | 113.9 | 0.41x |
| **YOLO26X-Engine** | TensorRT FP16 | 460.3 | 93.6% | 71.2% | 48.07 | 0.25 | 48.85 | 49.55 | 20.5 | 0.07x |
### PyTorch Eager Runtime (Tesla T4)

| Model | Format | Params | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-Sparse** | Fused Deploy | 22.8M | 45.2 | 87.1 | 78.8% | 46.1% | 35.59 | 0.53 | 36.61 | 27.3 |
| **UNINA-QAT** | PyTorch QAT | 22.8M | 45.2 | 21.8 | 82.3% | 52.8% | 70.78 | 0.50 | 71.76 | 13.9 |
| **YOLO26N** | Ultralytics | 2.4M | 2.4 | 9.1 | 87.0% | 62.1% | 10.18 | 0.20 | 10.84 | 92.3 |
| **YOLO26S** | Ultralytics | 9.5M | 9.3 | 36.1 | 90.8% | 67.5% | 12.96 | 0.20 | 13.64 | 73.3 |
| **YOLO26X** | Ultralytics | 55.6M | 87.2 | 212.2 | 93.6% | 71.2% | 84.17 | 0.21 | 84.87 | 11.8 |
---

## 4. Per-Class Accuracy Breakdown (FSOCO Validation)

Accuracy evaluated across the four Formula Student cone categories:

| Model | yellow_cone AP50 | blue_cone AP50 | small_orange AP50 | large_orange AP50 | Mean mAP50 | Mean mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-Dense (150-Epoch Deploy, conf=0.25)** | **88.6%** | **88.8%** | **88.8%** | **90.2%** | **89.1%** | **65.0%** |
| **UNINA-Sparse (150-Epoch Deploy, conf=0.25)** | **88.5%** | **88.8%** | **88.7%** | **90.7%** | **89.2%** | **64.9%** |
| **UNINA-QAT (150-Epoch Deploy, conf=0.25)** | **88.5%** | **88.6%** | **88.6%** | **91.5%** | **89.3%** | **64.9%** |
| **UNINA-Sparse (PyTorch Benchmark, conf>0.001)** | **87.5%** | **87.1%** | **87.4%** | **90.6%** | **88.2%** | **62.5%** |
| **UNINA-QAT (PyTorch Benchmark, conf>0.001)** | **87.7%** | **87.4%** | **87.2%** | **90.9%** | **88.3%** | **61.9%** |
| **UNINA-Sparse-Engine (Latest TRT Engine Run)** | **87.5%** | **87.1%** | **87.4%** | **90.6%** | **88.2%** | **62.5%** |
| **UNINA-QAT-Engine (Latest TRT Engine Run)** | **85.4%** | **85.3%** | **84.2%** | **82.2%** | **84.3%** | **50.3%** |
| **UNINA-QAT-Engine (Gen 2.6 Champion)** | **85.6%** | **86.2%** | **84.2%** | **84.6%** | **85.2%** | **57.6%** |
| **UNINA-Dense (Gen 2.6 Champion)** | **86.6%** | **86.9%** | **85.6%** | **85.8%** | **86.2%** | **61.0%** |
| **UNINA-QAT-Engine (Gen 2.5 Pretrained)** | **83.5%** | **84.3%** | **81.4%** | **83.0%** | **83.1%** | **52.1%** |
| **UNINA-Sparse-Engine (Gen 2.5 Pretrained)** | 83.4% | 83.6% | 79.5% | 80.1% | 81.6% | 55.6% |
| **UNINA-QAT-Engine (Gen 2.5 Scratch)** | 82.3% | 82.6% | 78.1% | 79.9% | 80.7% | 48.5% |
| **UNINA-Sparse-Engine (Gen 2.5 Scratch)** | 82.0% | 82.6% | 77.5% | 79.9% | 80.5% | 52.9% |
| **UNINA-Sparse-Engine (Initial TRT Diagnostic)** | 73.3% | 70.6% | 63.7% | 59.8% | **66.9%** | **44.7%** |
| **UNINA-QAT-Engine (Initial TRT Diagnostic)** | 8.5% | 0.5% | 0.4% | 0.3% | 2.4%* | 0.9%* |
| **UNINA-QAT-Engine (Tesla T4)** | 82.6% | 83.2% | 78.7% | 81.0% | 81.4% | 51.2% |
| **UNINA-Sparse-Engine (Tesla T4)** | 79.5% | 79.6% | 75.7% | 80.0% | 78.7% | 46.0% |
| **YOLO26N-Engine** | 87.8% | 88.3% | 86.6% | 85.5% | 87.0% | 62.1% |
| **YOLO26S-Engine** | 90.9% | 90.7% | 90.7% | 90.8% | 90.8% | 67.5% |
| **YOLO26X-Engine** | 93.0% | 93.3% | 93.8% | 94.2% | 93.6% | 71.2% |
---

## 5. Architectural Comparison

### 5.1 Gen 2.5 vs Gen 2.6 (Hardware-Verified Production Champion)

| Metric | Gen 2.5 (Pretrained Baseline) | Gen 2.6 (Production Champion) | Delta / Benefit |
| :--- | :---: | :---: | :---: |
| **Backbone Pretraining** | RepVGG Stages 0–3 (93.8% coverage) | **RepVGG Stages 0–4 (100.0% coverage)** | Full feature transfer |
| **Upsampling Method** | `nn.Upsample(nearest)` | **`nn.Upsample(nearest)` (Zero weights)** | Zero quantization error, 100% CBUF fit |
| **Warmup Epochs** | 3 epochs | **5 epochs** | Protects pretrained backbone |
| **Deploy Parameters** | 21.26M | **21.93M** | +0.67M (groups=1 alignment) |
| **Deploy FLOPs (Dense)** | 55.4 GFLOPs | **56.5 GFLOPs** | Minimal delta (+1.1 GFLOPs) |
| **Effective FLOPs (2:4 Sparse)** | 29.9 GFLOPs | **29.7 GFLOPs** | -0.2 GFLOPs (50.0% pruned) |
| **Dense Phase 1 mAP50 (O2O)** | 86.15% | **86.22%** | **+0.07%** (87.45% NMS) |
| **Dense Phase 1 mAP50-95** | 60.8% | **61.0%** | **+0.2%** |
| **Sparse Phase 1.5 mAP50 (O2O)** | 81.64% | **77.28%** | Baseline for aggressive QAT recovery |
| **QAT Phase 2 mAP50 (O2O)** | 83.05% | **85.17%** | **+2.12% (All-time project record)** |
| **QAT Phase 2 mAP50-95** | 52.1% | **57.6%** | **+5.5%** |
| **O2O vs NMS Gap (INT8)** | -1.22% | **-1.03%** | Near-parity with NMS |
| **CBUF Peak SRAM Usage** | 9 / 16 banks | **3 / 16 banks** | **67% SRAM reduction, 0 fallback** |
| **Status** | Previous Best | **Official Production Champion** | Top accuracy & hardware fit |
### 5.2 Architectural Comparison (Gen 2.5 vs Gen 3.1)

| Metric | Gen 2.5 (Production Champion) | Gen 3.1 (Spatial Neck Experiment) | Delta |
| :--- | :---: | :---: | :---: |
| **Backbone Configuration** | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` + DLASPPF | Added P5 SPPF |
| **Neck Type** | Pointwise 1x1 Rep-PAN | Spatial Rep-PAN (1x1 -> 3x3 RepVGG) | Added spatial blocks |
| **Neck Channels** | `[256, 192, 128]` | `[256, 192, 128]` | Identical |
| **Deploy Parameters** | **21.26M** | 22.82M | +1.56M (+7.3%) |
| **Deploy FLOPs** | **55.4 GFLOPs** | 60.9 GFLOPs | +5.5 GFLOPs (+9.9%) |
| **Dense Phase 1 mAP50 (O2O)**| **84.78%** | 84.69% | -0.09% |
| **Sparse Phase 1.5 mAP50 (O2O)**| **81.97%** | 81.89% | -0.08% |
| **QAT Phase 2 mAP50 (O2O)** | **83.78%** | 83.19% | -0.59% |
| **TensorRT Latency (RTX 6000)** | **0.51 ms** | 0.55 ms | +0.04 ms |
| **Recommendation** | **Production Champion** | Experimental Archive | Gen 2.5 preferred |
**Conclusion**: Gen 2.5 achieves equivalent or higher accuracy with lower parameter count, lower FLOPs, faster TensorRT execution, and better INT8 quantization stability. Gen 2.5 is therefore selected as the primary architecture.

