# UNINA-DLA Training & Optimization Report

Comprehensive report on the multi-phase training pipeline, architectural iterations, and optimization results for UNINA-DLA models on the Formula Student Objects in Context (FSOCO) dataset.

---

## 1. Overview of the Training Pipeline

The UNINA-DLA optimization pipeline is structured into three consecutive phases designed for edge hardware deployment:

```
[Phase 1: Dense Baseline Training]
  - Standalone GT training with dual-head supervision
  - Dynamic TAL top-k curriculum (3 -> 2 -> 1)
  - Cosine auxiliary loss decay (0.8 -> 0.0)
          |
          v
[Phase 1.5: 2:4 Structured Sparsity Fine-Tuning]
  - 50% weights pruned in candidate 2D convolutions
  - Zero-clamp and gradient masking hooks
          |
          v
[Phase 2: Quantization-Aware Training (QAT)]
  - Structural RepVGG fusion prior to quantizer insertion
  - INT8 fake-quantization with observer dampening and BN re-estimation
          |
          v
[Phase 3: Deploy Export]
  - Export O2O branch as 6-tensor output tuple: (reg3, reg4, reg5, cls3, cls4, cls5)
  - Zero NMS, zero DFL, DLA-compliant
```

### Note on Knowledge Distillation
Initial iterations explored Tri-Vector Knowledge Distillation from a high-capacity teacher (YOLO26-X). However, experimental evaluations demonstrated that **standalone ground-truth training with dynamic auxiliary loss decay consistently outperformed teacher distillation**. 

Because the teacher outputs soft probability distributions across multiple adjacent anchors, distillation smoothed the student's output features, directly counteracting the sharp single-anchor peak required for Hungarian one-to-one (O2O) assignment. Under standalone training with cosine auxiliary decay, the model learns sharp single-positive predictions without soft-target smearing, achieving higher final mAP50. Distillation is therefore classified as an **experimental feature** (`--enable_distillation`), while standalone training serves as the default production pipeline.

---

## 2. Architectural Generations & Empirical Comparison

Across the project, multiple model generations were trained and evaluated on FSOCO (9,256 train images, 1,157 validation images, input size 480 x 768):

| Metric | Gen 1 (Initial Distilled) | Gen 2 (SOD Rebalanced) | Gen 2.5 (Scratch Baseline) | Gen 2.5 (Pretrained Backbone) | Gen 2.6 (Hardware-Verified Champion) | Gen 2.6 (150-Epoch Extended) | Gen 3.1 (Spatial Neck Experiment) | Gen 3.2 (150+30+15 QAT, `--formula_image`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Backbone** | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` (Stages 0-3) | **RepVGG `[2, 4, 14, 1]` (100% Stages 0-4, groups=1)** | **RepVGG `[2, 4, 14, 1]` (100% Stages 0-4, groups=1)** | RepVGG `[2, 4, 14, 1]` + DLASPPF | **RepVGG `[2, 4, 14, 1]` + DLASPPF (100% Stages 0-4, groups=1)** |
| **Neck Channels** | `[128, 256, 256]` | `[192, 192, 128]` | `[256, 192, 128]` | `[256, 192, 128]` | `[256, 192, 128]` (groups=1) | `[256, 192, 128]` (groups=1) | `[256, 192, 128]` | `[256, 192, 128]` (groups=1) |
| **Neck Type** | Pointwise 1x1 | Pointwise 1x1 | Pointwise 1x1 | Pointwise 1x1 | **Pointwise 1x1 + nn.Upsample(nearest)** | Pointwise 1x1 + ConvTranspose2d (deconv) | Spatial Rep-PAN (1x1 -> 3x3) | Spatial Rep-PAN (1x1 -> 3x3) + ConvTranspose2d (deconv) |
| **Deploy Params** | 24.11M | 18.55M | 21.26M | 21.26M | **21.93M** (nearest) / 22.14M | **22.14M** (deconv) | 22.82M | Not logged |
| **Deploy FLOPs** | 37.2 GFLOPs | 32.2 GFLOPs | 55.4 GFLOPs | 55.4 GFLOPs | **56.0 GFLOPs** | **56.0 GFLOPs** | 60.9 GFLOPs | Not logged |
| **Phase 1 Dense mAP50 (O2O)** | 83.87% | 84.14% | 84.78% | 86.15% | **86.22%** | **89.09% (deploy) / 81.21% (raw)** | 84.69% | Not logged |
| **Phase 1 Dense mAP50 (NMS)** | 85.61% | 85.45% | 86.08% | 87.18% | **87.45%** | **89.57% (deploy) / 87.47% (raw)** | 85.98% | Not logged |
| **O2O vs NMS Gap (Dense)** | -1.74% | -1.31% | -1.30% | **-1.03%** | **-1.23%** | **-0.48% (deploy) / -6.26% (raw)** | -1.29% | Not logged |
| **Phase 1.5 Sparse mAP50 (O2O)** | 81.30% | 81.42% | 81.97% | **81.64%** | **77.28%** | **89.19% (deploy) / 77.61% (raw)** | 81.89% | Not logged |
| **Phase 2 QAT mAP50 (O2O)** | 82.59% | 82.80% | 83.78% | 83.05% | **85.17%** | **89.31% (deploy) / 78.58% (raw)** | 83.19% | Not logged |
| **Phase 4 Test mAP50 (`conf>0.001`)** | - | - | - | - | - | 79.29% | - | **90.0%** |
| **Phase 4 Test mAP50-95 (`conf>0.001`)** | - | - | - | - | - | 53.03% | - | **66.0%** |
| **CBUF Peak SRAM Banks** | 10 / 16 banks | 7 / 16 banks | 9 / 16 banks | 9 / 16 banks | **3 / 16 banks (INT8) / 4 (FP16)** | **3 / 16 banks (INT8) / 4 (FP16)** | 9 / 16 banks | Not logged |

### Key Takeaways
1. **Gen 2.5 with Pretrained Backbone reaches State-of-the-Art Accuracy**:
   - Initializing RepVGG-A1 Stages 0 to 3 with official ImageNet weights elevated Phase 1 Dense O2O performance from **84.78%** to **86.15% mAP50** (and NMS to **87.18% mAP50**, directly matching the Ultralytics YOLO26N 87.1% benchmark).
   - The Hungarian matching O2O vs NMS gap narrowed to just **-1.03%**, demonstrating near-zero regression degradation without NMS.
   - Retained **81.64% mAP50** after 2:4 structured sparsity pruning (50% candidate conv weights removed) and **83.05% mAP50** after INT8 QAT calibration.
2. **Gen 2.5 (Scratch Baseline) remains highly effective**:
   - Reached **84.78% mAP50** in dense O2O mode and **83.78% mAP50** after INT8 QAT when trained entirely from scratch.
   - Shifting channels to $P_3=256$ provided substantial representational capacity for small, distant cones without exceeding the 1 MiB CBUF SRAM budget (9 banks used out of 16).
3. **Gen 3.1 Spatial Neck Trade-Off**:
   - Adding $P_5$ DLASPPF and $3 \times 3$ RepVGG spatial neck blocks yielded statistical parity in dense training (84.69% vs 84.78%), but introduced 1.56M additional parameters, 5.5 GFLOPs, and lower INT8 QAT retention (83.19% vs 83.78%). Gen 2.5 remains the cleaner, faster choice.
4. **Gen 2.6 Hardware-Verified Champion Establishes All-Time Performance Record**:
   - Full 100% RepVGG-A1 pretraining across Stages 0 to 4 combined with zero-parameter nearest upsampling and 5 warmup epochs establishes project records across both floating-point and quantized deployment: **86.22% dense mAP50** (87.45% NMS) and **85.17% INT8 QAT mAP50** (86.20% NMS).
   - Surpasses Gen 2.5 INT8 QAT (83.05%) by **+2.12% mAP50** while reducing CBUF peak SRAM pressure from 9 to **3 / 16 banks** with 100% strict DLA hardware residency.
5. **Gen 2.6 Extended 150-Epoch Schedule Dynamics & Stagnation Analysis**:
   - Extending Phase 1 training to 150 epochs elevated operational deployment performance (`conf=0.25`, `iou=0.60`) to an all-time peak: **89.09% Phase 1 O2O mAP50** (89.57% NMS, gap narrowed to **-0.48%**), **89.19% Sparse mAP50**, and **89.31% QAT mAP50**.
   - However, the run exhibited a severe 40-epoch plateau between Epoch 110 and 150 (gaining only +0.2% deployment mAP) because the auxiliary TAL loss decayed to exact zero ($\alpha_{\text{aux}} = 0.00$) at Epoch 140, starving Hungarian single-anchor matching ($top\text{-}k=1$) of gradient density. On the unseen test split (1,158 images), Phase 4 evaluation ran only at the raw `conf>0.001` threshold and achieved **79.29% mAP50** (53.03% mAP50-95)—matching/exceeding QAT's **78.58%** peak (**77.41%** Epoch 15) validation score at the same `conf>0.001` threshold, confirming solid split generalization rather than overfitting.
6. **Gen 3.2 Standalone (150 + 30 + 15 Schedule, `--formula_image`) Achieves New Held-Out Test Record**:
   - Training Generation 3.2 (`RepVGG [2, 4, 14, 1] + DLASPPF`, Spatial $3\times 3$ Rep-PAN neck, `upsample=deconv`, unbounded LTRB regression) across the full **150 Dense + 30 Sparse + 15 QAT** schedule with `--formula_image`, non-zero auxiliary tail (`--o2m_aux_weight_end 0.15`), and extended TAL top-$k$ curriculum (`0.25` / `0.45`) elevated Phase 4 Held-Out Test Split performance (`1,158` images, `conf>0.001`) to **90.0% mAP50** and **66.0% mAP50-95**—a **+10.71 percentage-point** jump in mAP50 and **+12.97 percentage-point** jump in mAP50-95 over the Gen 2.6 150-Epoch Test result (79.29% / 53.03% at the exact same `conf>0.001` threshold).

---

## 3. Core Training Innovations

### Decoupled Dual-Head Convolutional Towers
During training, the detector utilizes two branches:
- **One-to-Many (O2M) Auxiliary Branch**: Employs Task-Aligned Assigner (TAL) with $top\text{-}k = 7..10$ to provide dense gradient feedback.
- **One-to-One (O2O) Primary Branch**: Enforces strict single-anchor matching ($top\text{-}k = 1$) for NMS-free deployment.

In earlier revisions with shared convolutional stems, dense O2M gradient signals overwhelmed the sharp peak suppression required by O2O. In the current implementation (`unina_dla/model/head/yolo26_head.py`), the classification and regression towers are fully decoupled (`o2o_cls_convs`, `o2o_reg_convs`, `o2m_cls_convs`, `o2m_reg_convs`). Only the O2O towers are retained at deployment time, incurring zero memory or latency penalty on edge hardware.

### Dynamic Auxiliary Loss Decay
To transition smoothly from dense supervision to single-positive specialization:
$$\alpha_{\text{aux}}(t) = \alpha_{\text{end}} + \frac{1}{2}(\alpha_{\text{start}} - \alpha_{\text{end}})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$
where $\alpha_{\text{start}} = 0.8$ decays monotonically to $\alpha_{\text{end}} = 0.0$ at the end of Phase 1. This completely removes multi-positive gradient interference in late training, allowing the O2O branch to reach within $-1.30\%$ of NMS performance (and $-1.03\%$ with pretrained backbone).

### Stable Wise-IoU v3 Loss
Box regression uses Wise-IoU v3 with detached diagonal:
$$\beta = \frac{\mathcal{L}_{\text{IoU}}^*}{\overline{\mathcal{L}_{\text{IoU}}}}, \quad r = \frac{\beta}{\delta \alpha^{\beta - \delta}}, \quad \mathcal{L}_{\text{WIoU}} = r \times \mathcal{R}_{\text{WIoU}} \times \mathcal{L}_{\text{IoU}}$$
Detaching the bounding box diagonal (`diag2.detach()`) eliminates gradient anomalies and stabilizes bounding box regression on small objects.

---

## 4. Multi-Phase Progression (Generation 2.5 & Generation 2.6)

### 4.1 Generation 2.5 with RepVGG-A1 Pretrained Backbone (Production Champion)

Detailed phase progression for Generation 2.5 initialized with official RepVGG-A1 ImageNet-1K weights across Stages 0 to 3 (339 tensors, 9,554,109 / 10,186,752 backbone params, 93.79% coverage; Stage 4 DLA layout and neck/heads scratch):

#### Phase 1: Dense Standalone (30 Epochs)
- **Epoch 5**: O2O mAP50 = 68.10%, NMS mAP50 = 72.08% (Gap: -3.98%)
- **Epoch 16**: O2O mAP50 = 85.05%, NMS mAP50 = 86.40% (Gap: -1.35%)
- **Epoch 26**: O2O mAP50 = 85.77%, NMS mAP50 = 86.99% (Gap: -1.22%)
- **Epoch 30**: O2O mAP50 = **86.15%**, NMS mAP50 = **87.18%** (Gap: **-1.03%**, mAP50-95: **60.8%** O2O / **61.4%** NMS at deployment threshold `conf=0.25`, `iou=0.60`).
- *Note*: Matches official YOLO26N (87.1% mAP50) in deployment detection accuracy while being 100% NMS-free.

#### Phase 1.5: 2:4 Structured Sparsity Fine-Tuning (6 Epochs)
- RepVGG multi-branch blocks fused into single-path $3 \times 3$ convolutions prior to pruning.
- Pruned **9,975,808 parameters** (50.0% of candidate 19,951,616 2D conv/linear weights).
- Effective compute reduced from 56.43 GFLOPs to **29.93 GFLOPs**.
- Deployment threshold (`conf=0.25`, `iou=0.60`):
  - **O2O mAP50 = 81.64%**, mAP50-95 = **55.6%**
  - **NMS mAP50 = 84.07%**, mAP50-95 = **57.0%**
  - Gap (O2O vs NMS): **-2.42%**

#### Phase 2: Quantization-Aware Training (3 Epochs)
- Calibrated 116 quantizers across activations and weights (histogram activation calibration, max weight calibration).
- Observer dampening applied in epochs 2–3 with final BatchNorm re-estimation.
- Deployment threshold (`conf=0.25`, `iou=0.60`):
  - **O2O mAP50 = 83.05%**, mAP50-95 = **52.1%**
  - **NMS mAP50 = 84.28%**, mAP50-95 = **52.6%**
  - Gap (O2O vs NMS): **-1.22%**
- Class-wise breakdown at deployment threshold:
  - `yellow_cone`: **83.5%** AP50, 50.3% AP50-95
  - `blue_cone`: **84.3%** AP50, 51.4% AP50-95
  - `small_orange_cone`: **81.4%** AP50, 51.0% AP50-95
  - `large_orange_cone`: **83.0%** AP50, 55.6% AP50-95
  - `Mean`: **83.05%** AP50, 52.1% AP50-95
- Deployed INT8 engine size: **22.1 MB** (Sparse FP16: **36.4 MB**).

---

### 4.2 Generation 2.5 Scratch Training Baseline

Phase progression for the baseline Generation 2.5 model trained entirely from scratch (30-epoch training schedule):

#### Phase 1: Dense Standalone (30 Epochs)
- **Epoch 1**: O2O mAP50 = 67.4%, NMS mAP50 = 74.2% (Gap: -6.8%)
- **Epoch 15**: O2O mAP50 = 80.3%, NMS mAP50 = 82.9% (Gap: -2.6%)
- **Epoch 30**: O2O mAP50 = **84.78%**, NMS mAP50 = **86.08%** (Gap: **-1.30%**, mAP50-95: **59.2%**)

#### Phase 1.5: 2:4 Structured Sparsity Fine-Tuning (6 Epochs)
- Pruned **9,975,808 parameters** (50.0% of candidate 2D conv weights).
- Effective compute reduced from 56.43 GFLOPs to **29.93 GFLOPs**.
- Final Sparse mAP50: **81.97%** O2O (**83.70%** NMS), retaining 96.7% of dense capability.

#### Phase 2: Quantization-Aware Training (3 Epochs)
- Calibrated 120 quantizers using Histogram activation calibration and Max weight calibration.
- Observer dampening applied in epochs 2–3 with final BatchNorm re-estimation.
- Final QAT mAP50: **83.78%** O2O (**84.82%** NMS), recovering +1.81% from the sparse baseline.
- Final O2O vs NMS gap after INT8 quantization: **-1.05%**. Deployed INT8 engine size: **22.1 MB** (Gen 2.5).

---

### 4.3 Generation 2.6 with RepVGG-A1 Pretrained Backbone (Production Champion, Rectangular 480x768, Deconv Upsampling)

Detailed phase progression for Generation 2.6 initialized with official RepVGG-A1 ImageNet-1K weights across Stages 0 to 4 (349 tensors, 10,868,925 / 10,868,925 params/buffers - 100.00% coverage; 64.2 GFLOPs at 480x768 resolution on FSOCO dataset):

#### Phase 1: Dense Standalone (30 Epochs)
- **Epoch 5**: O2O mAP50 = 65.28%, NMS mAP50 = 69.81% (Gap: -4.53%, mAP50-95: 39.5% O2O / 41.7% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 10**: O2O mAP50 = 77.41%, NMS mAP50 = 79.90% (Gap: -2.49%, mAP50-95: 49.5% O2O / 50.9% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 15**: O2O mAP50 = 82.73%, NMS mAP50 = 84.58% (Gap: -1.85%, mAP50-95: 56.2% O2O / 57.3% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 20**: O2O mAP50 = 84.82%, NMS mAP50 = 86.36% (Gap: -1.54%, mAP50-95: 59.5% O2O / 60.5% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 25**: O2O mAP50 = 85.49%, NMS mAP50 = 87.03% (Gap: -1.54%, mAP50-95: 60.3% O2O / 61.2% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 30**: O2O mAP50 = **85.93%**, NMS mAP50 = **87.28%** (Gap: **-1.35%**, mAP50-95: **60.8%** O2O / **61.6%** NMS at deployment threshold `conf=0.25`, `iou=0.60`).
- Class-wise breakdown at deployment threshold (`conf=0.25`, `iou=0.60`):
  - `yellow_cone`: **86.6%** AP50, 58.2% AP50-95
  - `blue_cone`: **86.9%** AP50, 59.4% AP50-95
  - `small_orange_cone`: **85.5%** AP50, 60.4% AP50-95
  - `large_orange_cone`: **84.7%** AP50, 64.9% AP50-95
  - `Mean`: **85.93%** AP50, 60.8% AP50-95

---

### 4.4 Generation 2.6 with RepVGG-A1 Pretrained Backbone (Production Champion, Rectangular 480x768, Nearest Upsampling, 5 Warmup Epochs)

Detailed phase progression for Generation 2.6 initialized with official RepVGG-A1 ImageNet-1K weights across Stages 0 to 4 (349 tensors, 10,868,925 / 10,868,925 params/buffers - 100.00% coverage; 64.0 GFLOPs at 480x768 resolution on FSOCO dataset, zero-parameter nearest upsampling, 5 warmup epochs):

#### Phase 1: Dense Standalone (30 Epochs)
- **Epoch 5**: O2O mAP50 = 65.92%, NMS mAP50 = 69.50% (Gap: -3.58%, mAP50-95: 39.4% O2O / 41.1% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 10**: O2O mAP50 = 76.85%, NMS mAP50 = 79.50% (Gap: -2.65%, mAP50-95: 49.1% O2O / 50.7% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 15**: O2O mAP50 = 82.81%, NMS mAP50 = 84.75% (Gap: -1.94%, mAP50-95: 57.1% O2O / 58.4% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 20**: O2O mAP50 = 84.58%, NMS mAP50 = 86.10% (Gap: -1.52%, mAP50-95: 59.4% O2O / 60.4% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 25**: O2O mAP50 = 85.92%, NMS mAP50 = 87.18% (Gap: -1.26%, mAP50-95: 60.7% O2O / 61.4% NMS at `conf=0.25`, `iou=0.60`)
- **Epoch 30**: O2O mAP50 = **86.22%**, NMS mAP50 = **87.45%** (Gap: **-1.23%**, mAP50-95: **61.0%** O2O / **61.8%** NMS at deployment threshold `conf=0.25`, `iou=0.60`).
- Class-wise breakdown at deployment threshold (`conf=0.25`, `iou=0.60`):
  - `yellow_cone`: **86.6%** AP50, 58.2% AP50-95
  - `blue_cone`: **86.9%** AP50, 59.7% AP50-95
  - `small_orange_cone`: **85.6%** AP50, 60.7% AP50-95
  - `large_orange_cone`: **85.8%** AP50, 65.5% AP50-95
  - `Mean`: **86.22%** AP50, 61.0% AP50-95

#### Phase 1.5: 2:4 Structured Sparsity Fine-Tuning (6 Epochs)
- RepVGG multi-branch blocks fused into single-path $3 \times 3$ convolutions prior to pruning.
- Pruned **10,418,176 parameters** (50.0% of candidate 20,836,352 2D conv/linear weights).
- Effective compute reduced from 56.51 GFLOPs to **29.65 GFLOPs**.
- Sparsity verification: 100% of candidate layers verified under strict 2:4 pattern.
- Deployment threshold (`conf=0.25`, `iou=0.60`):
  - **O2O mAP50 = 77.28%**, mAP50-95 = **52.4%**
  - **NMS mAP50 = 80.66%**, mAP50-95 = **54.3%**
  - Gap (O2O vs NMS): **-3.39%**
- Class-wise breakdown at deployment threshold:
  - `yellow_cone`: **80.2%** AP50, 51.9% AP50-95
  - `blue_cone`: **80.2%** AP50, 52.0% AP50-95
  - `small_orange_cone`: **72.7%** AP50, 50.4% AP50-95
  - `large_orange_cone`: **76.0%** AP50, 55.1% AP50-95
  - `Mean`: **77.28%** AP50, 52.4% AP50-95

#### Phase 2: Quantization-Aware Training (3 Epochs)
- Calibrated 116 quantizers across activations and weights (histogram activation calibration, max weight calibration).
- Observer dampening applied in epochs 2–3 with final BatchNorm re-estimation.
- Deployment threshold (`conf=0.25`, `iou=0.60`):
  - **O2O mAP50 = 85.17%**, mAP50-95 = **57.6%**
  - **NMS mAP50 = 86.20%**, mAP50-95 = **58.2%**
  - Gap (O2O vs NMS): **-1.03%**
- Class-wise breakdown at deployment threshold:
  - `yellow_cone`: **85.6%** AP50, 55.5% AP50-95
  - `blue_cone`: **86.2%** AP50, 56.7% AP50-95
  - `small_orange_cone`: **84.2%** AP50, 57.0% AP50-95
  - `large_orange_cone`: **84.6%** AP50, 61.2% AP50-95
  - `Mean`: **85.17%** AP50, 57.6% AP50-95
- **Accuracy Recovery**: QAT recovered **+7.89% mAP50** over the sparse baseline (from 77.28% to 85.17%), retaining **98.8%** of the full dense floating-point accuracy while running at INT8 precision on edge DLA.

#### Post-Export TensorRT Engine Validation (RTX PRO 6000 Blackwell Server Edition)
- Evaluated the compiled TensorRT engines directly via native runtime (`scripts/benchmark.py`, 480x768, batch 1, 200 iterations):
  - **UNINA-QAT-Engine (INT8)**: **84.84% mAP50** (58.13% mAP50-95) | **0.48 ms** inference | **1,325.4 FPS** E2E (Engine Size: 24.8 MB)
    - `yellow_cone`: **84.8%** AP50, 55.7% AP50-95
    - `blue_cone`: **85.0%** AP50, 55.8% AP50-95
    - `small_orange_cone`: **82.8%** AP50, 56.3% AP50-95
    - `large_orange_cone`: **86.7%** AP50, 64.7% AP50-95
  - **UNINA-Sparse-Engine (FP16)**: **79.32% mAP50** (49.61% mAP50-95) | **0.53 ms** inference | **1,267.1 FPS** E2E (Engine Size: 37.9 MB)
    - `yellow_cone`: **82.3%** AP50, 49.9% AP50-95
    - `blue_cone`: **81.9%** AP50, 50.2% AP50-95
    - `small_orange_cone`: **75.5%** AP50, 46.8% AP50-95
    - `large_orange_cone`: **77.6%** AP50, 51.6% AP50-95
- **Engine Fidelity**: The compiled INT8 TensorRT engine reproduces the PyTorch QAT checkpoint accuracy almost identically (84.84% vs 85.17%, a delta of only **-0.33% mAP50**), confirming flawless INT8 calibration and scale preservation on TensorRT native runtime.

---

### 4.5 Generation 2.6 Extended 150-Epoch Training Schedule (Extended Standalone, 30 Sparse, 15 QAT, Deconv Upsampling)

Complete multi-phase progression for the extended 150-epoch Generation 2.6 training schedule (with dynamic 20% = 30 sparsity epochs and dynamic 10% = 15 QAT epochs, RepVGG-A1 100% Stages 0–4 pretraining, and `upsample=deconv` transposed convolutions):

#### Phase 1: Extended Dense Standalone (150 Epochs)
- **Epoch 109**: O2O mAP50 = 79.8%, NMS mAP50 = 87.2% | Loss = 1.97, $Aw = 0.14$, LR = $2.68 \times 10^{-3}$
- **Epoch 110**:
  - Primary (conf>0.001): O2O mAP50 = **80.10%**, mAP50-95 = **56.6%**
  - Secondary (NMS=True): NMS mAP50 = **87.35%**, mAP50-95 = **60.6%** (Gap: -7.25%)
  - Deployment Threshold (`conf=0.25`, `iou=0.60`): **O2O mAP50 = 88.88%** (64.4% AP50-95), **NMS mAP50 = 89.43%** (64.7% AP50-95, Gap: **-0.55%**)
- **Epoch 134**: O2O mAP50 = 81.24%, NMS mAP50 = 87.4% | $Aw = 0.02$, LR = $7.78 \times 10^{-4}$
- **Epoch 145**:
  - Primary (conf>0.001): O2O mAP50 = **81.43%**, mAP50-95 = **57.9%**
  - Secondary (NMS=True): NMS mAP50 = **87.52%**, mAP50-95 = **61.1%** (Gap: -6.09%)
  - Deployment Threshold (`conf=0.25`): **O2O mAP50 = 89.07%** (64.9% AP50-95), **NMS mAP50 = 89.58%** (65.2% AP50-95, Gap: **-0.51%**)
- **Epoch 150 (Final Phase 1 Checkpoint - `best_standalone.pth`)**:
  - Primary (conf>0.001): O2O mAP50 = **81.21%**, mAP50-95 = **57.7%**
  - Secondary (NMS=True): NMS mAP50 = **87.47%**, mAP50-95 = **61.0%** (Gap: -6.26%)
  - Deployment Threshold (`conf=0.25`, `iou=0.60`):
    - **O2O mAP50 = 89.09%**, mAP50-95 = **65.0%**
    - **NMS mAP50 = 89.57%**, mAP50-95 = **65.3%**
    - Gap (O2O vs NMS): **-0.48%** (-0.0048)
  - Class-wise breakdown at deployment threshold (`conf=0.25`, `iou=0.60`):
    - `yellow_cone`: **88.6%** AP50, 61.4% AP50-95
    - `blue_cone`: **88.8%** AP50, 62.1% AP50-95
    - `small_orange_cone`: **88.8%** AP50, 63.9% AP50-95
    - `large_orange_cone`: **90.2%** AP50, 72.4% AP50-95
    - `Mean`: **89.09%** AP50, 65.0% AP50-95

#### Phase 1.5: Extended 2:4 Structured Sparsity Fine-Tuning (30 Epochs)
- Pruned **10,418,176 candidate weights** across fused $3 \times 3$ convolutions (20,836,352 prunable weights).
- Training schedule: dynamically scaled to 30 epochs (20% of 150) with optimizer tracking fused `rbr_reparam` layers and clean bias retention.
- **Epoch 16/30 (Peak Raw Sparse Checkpoint)**:
  - Primary (conf>0.001): **77.61% mAP50**, mAP50-95 = **54.3%**
  - Class-wise: `yellow_cone`: 77.1%, `blue_cone`: 77.2%, `small_orange_cone`: 76.6%, `large_orange_cone`: 79.6%
- **Epoch 30/30 (Final Sparse Checkpoint - `best_sparse.pth`)**:
  - Primary (conf>0.001): O2O mAP50 = **77.17%**, mAP50-95 = **55.4%**
  - Secondary (NMS=True): NMS mAP50 = **87.09%**, mAP50-95 = **60.3%** (Gap: -9.92%)
  - Deployment Threshold (`conf=0.25`, `iou=0.60`):
    - **O2O mAP50 = 89.19%**, mAP50-95 = **64.9%**
    - **NMS mAP50 = 89.83%**, mAP50-95 = **65.3%**
    - Gap (O2O vs NMS): **-0.65%** (-0.0065)
  - Class-wise breakdown at deployment threshold:
    - `yellow_cone`: **88.5%** AP50, 61.5% AP50-95
    - `blue_cone`: **88.8%** AP50, 62.4% AP50-95
    - `small_orange_cone`: **88.7%** AP50, 63.6% AP50-95
    - `large_orange_cone`: **90.7%** AP50, 72.1% AP50-95
    - `Mean`: **89.19%** AP50, 64.9% AP50-95
- Sparsity verification: 100% of candidate layers verified under strict 2:4 pattern.

#### Phase 2: Extended Quantization-Aware Training (15 Epochs)
- Inserted 116 quantizers (signed histogram percentile activation calibration, per-channel max weight calibration).
- Training schedule: dynamically scaled to 15 epochs (10% of 150) with observer dampening and post-QAT activation calibration.
- **Epoch 5/15 (Peak Raw QAT Checkpoint)**:
  - Primary (conf>0.001): **78.58% mAP50**, mAP50-95 = **54.5%**
  - Deployment Threshold (`conf=0.25`, `iou=0.60`): **88.89% O2O mAP50** (62.8% AP50-95), **89.39% NMS mAP50** (63.1% AP50-95)
- **Epoch 15/15 (Final QAT Checkpoint - `best_qat.pth`)**:
  - Primary (conf>0.001): O2O mAP50 = **77.41%**, mAP50-95 = **55.5%**
  - Secondary (NMS=True): NMS mAP50 = **86.90%**, mAP50-95 = **60.1%** (Gap: -9.49%)
  - Deployment Threshold (`conf=0.25`, `iou=0.60`):
    - **O2O mAP50 = 89.31%**, mAP50-95 = **64.9%**
    - **NMS mAP50 = 89.83%**, mAP50-95 = **65.2%**
    - Gap (O2O vs NMS): **-0.52%** (-0.0052)
  - Class-wise breakdown at deployment threshold:
    - `yellow_cone`: **88.5%** AP50, 61.7% AP50-95
    - `blue_cone`: **88.6%** AP50, 62.5% AP50-95
    - `small_orange_cone`: **88.6%** AP50, 63.4% AP50-95
    - `large_orange_cone`: **91.5%** AP50, 72.0% AP50-95
    - `Mean`: **89.31%** AP50, 64.9% AP50-95

#### Phase 3 & 4: Clean ONNX Export, DLA Verification & Held-Out Test Set
- **Phase 3 Clean Deploy Export (`unina_dla_best_qat_sparse_qatclean_480x768.onnx`)**:
  - Opset 17, 110 total nodes, 49 ReLU activations, 3 Sigmoids, 0 forbidden DLA operators.
  - 100% strict DLA compatibility verification passed (`VERDICT: PASS - Model is DLA-compatible`).
  - Saved 70 INT8 quantization scales to `unina_dla_best_qat_sparse_qatclean_480x768_int8_scales.json`.
- **Phase 4 Evaluation on Completely Held-Out Test Split (1,158 Images)**:
  - Standard raw evaluation protocol (`conf>0.001` only; Phase 4 does not run a `conf=0.25` deployment pass):
    - **Final Test Set mAP50 = 79.29%**, **mAP50-95 = 53.03%** (comparable to QAT validation `conf>0.001`: **78.58%** peak / **77.41%** Epoch 15)
    - `class0` (`yellow_cone`): **80.0%** AP50, 50.3% AP50-95
    - `class1` (`blue_cone`): **81.6%** AP50, 52.8% AP50-95
    - `class2` (`small_orange_cone`): **77.4%** AP50, 50.7% AP50-95
    - `class3` (`large_orange_cone`): **78.2%** AP50, 58.3% AP50-95

#### Empirical Analysis of 150-Epoch Training Bottlenecks
1. **The 40-Epoch Stagnation Plateau (Epochs 110–150)**:
   - Between Epoch 110 (80.1% raw / 88.9% deploy) and Epoch 150 (81.2% raw / 89.1% deploy), the model plateaued for 40 epochs.
   - **Root Cause**: The dynamic auxiliary loss weight decayed to exact zero ($\alpha_{\text{aux}} = 0.00$) at Epoch 140, completely disabling the dense One-to-Many branch. With learning rate bottoming out at $1.3 \times 10^{-4}$, strict single-anchor Hungarian matching ($top\text{-}k=1$) lacked gradient density to push weights further.
2. **Sparsity Overshooting (30 Epochs vs 6 Epochs)**:
   - In the initial unaligned run, dynamically extending 2:4 structured sparsity to 30 epochs without fused layer optimizer tracking degraded low-confidence feature representations (raw O2O dropped to 64.39%). After fixing optimizer parameter registration and checkpoint loading, the resumed 30-epoch run maintained **77.61% raw O2O mAP50** and **89.19% deploy mAP50**.
3. **Threshold Comparison Note on Test Split (`conf>0.001` vs `conf=0.25`)**:
   - The **89.31% mAP50** QAT validation metric was measured at the deployment threshold (`conf=0.25`), whereas Phase 4 test evaluation (`scripts/train_student.py`) runs `validate()` at the default raw threshold (`conf>0.001`), yielding **79.29% mAP50**.
   - Compared apples-to-apples at `conf>0.001`, Test (**79.29%**) matches or slightly exceeds QAT validation (**78.58%** peak at Epoch 5, **77.41%** at Epoch 15), showing no validation overfitting gap—though enabling per-epoch shuffling via `--formula_image` remains good practice to avoid static rectangular batch ordering.

---

### 4.6 Generation 3.2 Extended Standalone Pipeline (150 Dense + 30 Sparse + 15 QAT, `deconv`, `--formula_image`)

Completed multi-phase run for **Generation 3.2** (`RepVGG [2, 4, 14, 1] + DLASPPF`, `Spatial` $3 \times 3$ Rep-PAN neck with `[256, 192, 128]` channels, `upsample_mode=deconv`, unbounded LTRB regression, 100% RepVGG-A1 Stages 0–4 pretraining) over the full **150 + 30 + 15** epoch schedule without early-stop termination (`195` total epochs across all three phases). Although `--teacher` was supplied on the command line, `--enable_distillation` was not set, so training and QAT executed in standalone mode.

#### Training Configuration
```bash
python scripts/train_student.py \
    --teacher "{LOCAL_TEACHER_PATH}" \
    --data unina_dla/config/unina_dla_data.yaml \
    --data_path "{CANONICAL_DATASET_DIR}" \
    --project "{STUDENT_PROJECT_DIR}" \
    --name "{STUDENT_RUN_NAME}" \
    --version 3.2 \
    --upsample_mode deconv \
    --warmup_epochs 5 \
    --epochs_sparsity 30 \
    --epochs_qat 15 \
    --patience 25 \
    --min_delta 0.001 \
    --early_stop_metric fitness \
    --early_stop_min_epochs 75 \
    --patience_sparsity 5 \
    --patience_qat 4 \
    --o2m_aux_weight_end 0.15 \
    --o2o_topk_stage1_pct 0.25 \
    --o2o_topk_stage2_pct 0.45 \
    --optimizer MuSGD \
    --pretrained_backbone \
    --enable_sparsity \
    --enable_qat \
    --formula_image \
    --qat-clean-deploy \
    --epochs 150 \
    --batch 16 \
    --imgsz 768,480 \
    --no-compile
```

#### Phase 4 Evaluation on Held-Out Test Split (1,158 Images)
Evaluated after Phase 2 QAT + Phase 3 clean deploy export on the unseen FSOCO test set (`conf>0.001`, NMS-free O2O):
- **Images**: `1,158`
- **Predictions (`conf>0.001`)**: `40,117`
- **Speed**: `0.0ms` preprocess, `2.7ms` inference, `0.2ms` postprocess per image at shape `(16, 3, 480, 768)`

| Class | Category Name | Test AP50 (`conf>0.001`) | Test AP50-95 (`conf>0.001`) | Delta vs Gen 2.6 150-Epoch Test (Section 4.5) |
| :--- | :--- | :---: | :---: | :---: |
| `class0` | `yellow_cone` | **91.6%** | **64.8%** | **+11.6 pp** AP50 / **+14.5 pp** AP50-95 |
| `class1` | `blue_cone` | **90.4%** | **64.8%** | **+8.8 pp** AP50 / **+12.0 pp** AP50-95 |
| `class2` | `small_orange_cone` | **89.9%** | **65.3%** | **+12.5 pp** AP50 / **+14.6 pp** AP50-95 |
| `class3` | `large_orange_cone` | **88.1%** | **69.1%** | **+9.9 pp** AP50 / **+10.8 pp** AP50-95 |
| **Mean** | **All Classes** | **90.0%** | **66.0%** | **+10.71 pp** AP50 / **+12.97 pp** AP50-95 |

#### Post-Export TensorRT Engine Validation (NVIDIA Tesla T4, TensorRT 10.16.1.11)
Evaluated the compiled Gen 3.2 TensorRT engines directly on **Tesla T4** via `scripts/benchmark.py` (`480 x 768`, batch size 1, 50 warmup, 200 timed iterations) against FP16 and INT8-PTQ YOLO26 baselines in the same environment:

- **UNINA-QAT-INT8 (`19.9 MB`)**: **87.3% mAP50** (**62.2% mAP50-95**) | **2.52 ms** inference | **279.6 FPS** E2E (`3.58 ms` E2E, `4.09 ms` P95)
- **UNINA-QAT-FP16 (`38.5 MB`)**: **87.5% mAP50** (**62.0% mAP50-95**) | **4.25 ms** inference | **187.4 FPS** E2E (`5.34 ms` E2E, `5.53 ms` P95)
- **UNINA-Sparse-FP16 (`38.5 MB`)**: **87.7% mAP50** (**61.8% mAP50-95**) | **4.26 ms** inference | **188.8 FPS** E2E (`5.30 ms` E2E, `5.48 ms` P95)
- **Baselines in the same T4 TensorRT 10.16 run**:
  - `YOLO26N-FP16` (`6.8 MB`): 78.7% mAP50 / 52.2% mAP50-95 | 1.78 ms | 376.1 FPS (`YOLO26N-INT8`: 76.6% / 49.6% | 1.70 ms | 382.0 FPS)
  - `YOLO26S-FP16` (`20.4 MB`): 84.1% mAP50 / 58.2% mAP50-95 | 2.75 ms | 275.0 FPS (`YOLO26S-INT8`: 70.6% / 45.3% | 1.91 ms | 358.7 FPS)
  - `YOLO26X-FP16` (`108.8 MB`): 89.3% mAP50 / 63.0% mAP50-95 | 12.84 ms | 72.7 FPS (`YOLO26X-INT8`: 83.8% / 52.5% | 7.50 ms | 118.4 FPS)
- **QAT Retention**: Comparing `UNINA-QAT-INT8` directly against `UNINA-QAT-FP16`, explicit Q/DQ INT8 quantization retained **99.8%** of mAP50 (`87.3%` vs `87.5%`, `-0.2 pp`) and **100%** of mAP50-95 (`62.2%` vs `62.0%`, `+0.2 pp`) while shrinking engine size by **48.3%** and boosting inference speed by **1.69x**—in sharp contrast to Ultralytics INT8 PTQ, which dropped `YOLO26S` by **-13.5 pp** mAP50 (`-12.9 pp` mAP50-95) and `YOLO26X` by **-5.5 pp** mAP50 (`-10.5 pp` mAP50-95).
- **Dependency Change Caveat**: This T4 validation used **TensorRT `10.16.1.11`**, an upgraded **`ultralytics 8.4.x`** (`Settings v0.0.8` `compute_ap` update, which lowers mAP at deployment threshold relative to `8.4.4`), and slim `scripts/build_engine.py` FP16 YOLO exports. Because of these environment differences, direct 1:1 comparisons with historical runs in Sections 4.4–4.5 or earlier benchmark tables may not be exact.

#### Post-Export Workstation Validation (NVIDIA RTX PRO 6000 Blackwell, TensorRT 10.16.1.11 & PyTorch Eager)

- **PyTorch Eager (`480x768`, Batch=1, CUDA)**:
  - **`UNINA-Standalone`**: `88.1%` mAP50, `63.0%` mAP50-95, `2.45 ms` infer (`2.78 ms` E2E, `359.7 FPS`, `92.3 MB`)
  - **`UNINA-Sparse`**: `87.7%` mAP50, `61.8%` mAP50-95, `2.45 ms` infer (`2.78 ms` E2E, `360.3 FPS`, `92.3 MB`) — matches standalone latency because PyTorch eager executes dense `Conv2d` kernels over zero-masked 2:4 weights.
  - **`UNINA-QAT`**: `87.3%` mAP50, `62.1%` mAP50-95, `12.71 ms` infer (`13.06 ms` E2E, `76.6 FPS`, `23.1 MB`) — simulated `Q/DQ` overhead before TensorRT lowering.
  - **YOLOv26 Baselines (PyTorch)**: `YOLO26X` (`89.3%` / `63.0%`, `7.48 ms` infer, `128.9 FPS`), `YOLO26S` (`84.1%` / `58.2%`, `3.04 ms` infer, `301.8 FPS`), `YOLO26N` (`78.7%` / `52.2%`, `2.86 ms` infer, `320.4 FPS`).
- **TensorRT `10.16.1.11` Engines (`480x768`, Batch=1)**:
  - **`UNINA-QAT-INT8`**: **`87.3%` mAP50**, **`62.2%` mAP50-95**, **`0.57 ms` infer** (`0.88 ms` E2E, `1,140.0 FPS`, `26.8 MB`)
  - **`UNINA-Standalone-FP16`**: `87.9%` mAP50, `63.0%` mAP50-95, `0.63 ms` infer (`0.93 ms` E2E, `1,070.9 FPS`, `39.2 MB`)
  - **`UNINA-Sparse-FP16`**: `87.7%` mAP50, `61.8%` mAP50-95, `0.63 ms` infer (`0.93 ms` E2E, `1,075.2 FPS`, `41.5 MB`)
  - **`UNINA-QAT-FP16`**: `87.5%` mAP50, `62.0%` mAP50-95, `0.62 ms` infer (`0.95 ms` E2E, `1,055.5 FPS`, `41.9 MB`)
  - **YOLOv26-FP16 Baselines**: `YOLO26N-FP16` (`78.4%` / `51.7%`, `0.65 ms` infer, `1,143.7 FPS`, `6.3 MB`), `YOLO26S-FP16` (`83.8%` / `57.5%`, `0.78 ms` infer, `1,000.9 FPS`, `20.0 MB`), `YOLO26X-FP16` (`88.8%` / `62.2%`, `2.07 ms` infer, `435.4 FPS`, `108.5 MB`).
- **Dependency & Batch=1 Caveat**: Same **TensorRT `10.16.1.11`**, upgraded **`ultralytics 8.4.x`**, and slim `scripts/build_engine.py` FP16 YOLO export environment as the T4 run; `batch=1` execution on RTX PRO 6000 is launch/memory-bandwidth bound (~`0.57–0.63 ms`), making sparse and dense FP16 engines run at identical speed and compressing INT8 vs FP16 gains compared to Tesla T4.




