# UNINA-DLA Benchmark Report

Comprehensive benchmark evaluation of UNINA-DLA object detection models across workstation (NVIDIA RTX PRO 6000) and edge GPU (NVIDIA Tesla T4) hardware, comparing against official Ultralytics YOLO26 baselines.

- **Resolution**: 480 x 768 (Height x Width, Batch Size: 1)
- **Dataset**: Formula Student Objects in Context (FSOCO) validation split (1,157 images)
- **Protocol**: 50 warmup iterations, 200 timed iterations; CUDA event timing for microsecond precision

---

## 1. Summary of Results

> [!NOTE]
> All benchmarks reported in this document were collected on desktop and server GPUs (RTX PRO 6000 and Tesla T4). Physical on-device DLA benchmarks could not be measured due to lack of Jetson Orin hardware.

UNINA-DLA was evaluated across two deployment configurations:
- **UNINA-QAT (INT8)**: Generation 2.5 student model with 2:4 structured sparsity and Quantization-Aware Training.
- **UNINA-Sparse (FP16)**: Generation 2.5 student model with 2:4 structured sparsity.

Compared to standard YOLO26 baselines:
- On **RTX PRO 6000**, `UNINA-QAT-Engine` achieved **0.47 ms** inference latency (**1,359.8 FPS** end-to-end), running 1.94x faster than YOLO26N-Engine (0.91 ms) and 2.83x faster than YOLO26S-Engine (1.33 ms).
- On **Tesla T4**, `UNINA-QAT-Engine` achieved **2.77 ms** inference latency (**266.2 FPS** end-to-end), running 1.21x faster than YOLO26N-Engine (3.35 ms) and 2.92x faster than YOLO26S-Engine (8.08 ms).
- Deployed INT8 engine size is **19.8 MB** (compared to 38.9 MB for YOLO26S and 215.4 MB for YOLO26X).

---

## 2. Workstation Benchmarks (NVIDIA RTX PRO 6000)

### TensorRT Native Engine Execution

| Model | Format / Precision | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | P95 (ms) | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-QAT-Engine (Gen 2.5)** | TensorRT INT8 | **19.8** | **80.7%** | **48.5%** | **0.47** | 0.16 | **0.74** | **0.75** | **1,359.8** |
| **UNINA-Sparse-Engine (Gen 2.5)** | TensorRT FP16 | 31.8 | 80.5% | 52.9% | 0.54 | 0.15 | 0.80 | 0.81 | 1,252.2 |
| **UNINA-QAT-Engine (Gen 3.1)** | TensorRT INT8 | 21.4 | 81.5% | 50.1% | 0.55 | 0.16 | 0.82 | 0.84 | 1,224.0 |
| **YOLO26N-Engine** | TensorRT FP16 | 11.9 | 87.1% | 62.1% | 0.91 | 0.05 | 1.06 | 1.07 | 940.8 |
| **YOLO26S-Engine** | TensorRT FP16 | 38.9 | 90.8% | 67.5% | 1.33 | 0.05 | 1.48 | 1.50 | 674.0 |
| **YOLO26X-Engine** | TensorRT FP16 | 215.4 | 93.6% | 71.2% | 4.62 | 0.05 | 4.77 | 4.79 | 209.4 |

### PyTorch Eager Runtime (CUDA)

| Model | Format | Params | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Infer (ms) | Post (ms) | E2E (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNINA-Sparse (Gen 2.5)** | Fused Deploy | 18.6M | 32.2 | 70.8 | 80.5% | 52.9% | 2.03 | 0.18 | 2.33 | 429.7 |
| **UNINA-QAT (Gen 2.5)** | PyTorch QAT | 18.6M | 32.9 | 17.7 | 82.0% | 51.0% | 12.51 | 0.20 | 12.83 | 78.0 |
| **UNINA-QAT (Gen 3.1)** | PyTorch QAT | 22.8M | 60.9 | 21.8 | 82.3% | 52.8% | 13.10 | 0.20 | 13.45 | 74.3 |
| **YOLO26N** | Ultralytics | 2.4M | 2.4 | 9.1 | 87.0% | 62.2% | 3.06 | 0.08 | 3.25 | 307.4 |
| **YOLO26S** | Ultralytics | 9.5M | 9.3 | 36.1 | 90.8% | 67.5% | 3.10 | 0.08 | 3.29 | 304.2 |
| **YOLO26X** | Ultralytics | 55.6M | 87.2 | 212.2 | 93.6% | 71.2% | 6.10 | 0.08 | 6.30 | 158.8 |

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
| **UNINA-QAT-Engine (RTX 6000)** | 82.3% | 82.6% | 78.1% | 79.9% | **80.7%** | 48.5% |
| **UNINA-Sparse-Engine (RTX 6000)**| 82.0% | 82.6% | 77.5% | 79.9% | **80.5%** | 52.9% |
| **UNINA-QAT-Engine (Tesla T4)** | 82.6% | 83.2% | 78.7% | 81.0% | **81.4%** | 51.2% |
| **UNINA-Sparse-Engine (Tesla T4)** | 79.5% | 79.6% | 75.7% | 80.0% | 78.7% | 46.0% |
| **YOLO26N-Engine** | 87.8% | 88.3% | 86.6% | 85.5% | 87.0% | 62.2% |
| **YOLO26S-Engine** | 90.9% | 90.7% | 90.7% | 90.8% | 90.8% | 67.5% |
| **YOLO26X-Engine** | 93.0% | 93.3% | 93.8% | 94.2% | 93.6% | 71.2% |

---

## 5. Architectural Comparison (Gen 2.5 vs Gen 3.1)

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
| **TensorRT Latency (RTX 6000)** | **0.47 ms** | 0.55 ms | +0.08 ms |
| **Recommendation** | **Production Champion** | Experimental Archive | Gen 2.5 preferred |

**Conclusion**: Gen 2.5 achieves equivalent or higher accuracy with lower parameter count, lower FLOPs, faster TensorRT execution, and better INT8 quantization stability. Gen 2.5 is therefore selected as the primary architecture.
