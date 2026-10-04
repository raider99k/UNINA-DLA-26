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

| Metric | Gen 1 (Initial Distilled) | Gen 2 (SOD Rebalanced) | Gen 2.5 (Production Champion) | Gen 3.1 (Spatial Neck Experiment) |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone** | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` | RepVGG `[2, 4, 14, 1]` + DLASPPF |
| **Neck Channels** | `[128, 256, 256]` | `[192, 192, 128]` | `[256, 192, 128]` | `[256, 192, 128]` |
| **Neck Type** | Pointwise 1x1 | Pointwise 1x1 | Pointwise 1x1 | Spatial Rep-PAN (1x1 -> 3x3) |
| **Deploy Params** | 24.11M | 18.55M | **21.26M** | 22.82M |
| **Deploy FLOPs** | 37.2 GFLOPs | 32.2 GFLOPs | **55.4 GFLOPs** | 60.9 GFLOPs |
| **Phase 1 Dense mAP50 (O2O)** | 83.87% | 84.14% | **84.78%** | 84.69% |
| **Phase 1 Dense mAP50 (NMS)** | 85.61% | 85.45% | **86.08%** | 85.98% |
| **O2O vs NMS Gap (Dense)** | -1.74% | -1.31% | **-1.30%** | -1.29% |
| **Phase 1.5 Sparse mAP50 (O2O)** | 81.30% | 81.42% | **81.97%** | 81.89% |
| **Phase 2 QAT mAP50 (O2O)** | 82.59% | 82.80% | **83.78%** | 83.19% |
| **CBUF Peak SRAM Banks** | 10 / 16 banks | 7 / 16 banks | **9 / 16 banks** | 9 / 16 banks |

### Key Takeaways
1. **Gen 2.5 is the Production Champion**:
   - Reached **84.78% mAP50** in dense O2O mode and **83.78% mAP50** after INT8 QAT.
   - Shifting channels to $P_3=256$ provided substantial representational capacity for small, distant cones without exceeding the 1 MiB CBUF SRAM budget (9 banks used out of 16).
2. **Gen 3.1 Spatial Neck Trade-Off**:
   - Adding $P_5$ DLASPPF and $3 \times 3$ RepVGG spatial neck blocks yielded statistical parity in dense training (84.69% vs 84.78%), but introduced 1.56M additional parameters, 5.5 GFLOPs, and lower INT8 QAT retention (83.19% vs 83.78%). Gen 2.5 remains the cleaner, faster choice.

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
where $\alpha_{\text{start}} = 0.8$ decays monotonically to $\alpha_{\text{end}} = 0.0$ at the end of Phase 1. This completely removes multi-positive gradient interference in late training, allowing the O2O branch to reach within $-1.30\%$ of NMS performance.

### Stable Wise-IoU v3 Loss
Box regression uses Wise-IoU v3 with detached diagonal:
$$\beta = \frac{\mathcal{L}_{\text{IoU}}^*}{\overline{\mathcal{L}_{\text{IoU}}}}, \quad r = \frac{\beta}{\delta \alpha^{\beta - \delta}}, \quad \mathcal{L}_{\text{WIoU}} = r \times \mathcal{R}_{\text{WIoU}} \times \mathcal{L}_{\text{IoU}}$$
Detaching the bounding box diagonal (`diag2.detach()`) eliminates gradient anomalies and stabilizes bounding box regression on small objects.

---

## 4. Multi-Phase Progression for Generation 2.5

Detailed phase progression for the champion Generation 2.5 model (30-epoch training schedule):

### Phase 1: Dense Standalone (30 Epochs)
- **Epoch 1**: O2O mAP50 = 67.4%, NMS mAP50 = 74.2% (Gap: -6.8%)
- **Epoch 15**: O2O mAP50 = 80.3%, NMS mAP50 = 82.9% (Gap: -2.6%)
- **Epoch 30**: O2O mAP50 = **84.78%**, NMS mAP50 = **86.08%** (Gap: **-1.30%**, mAP50-95: **59.2%**)

### Phase 1.5: 2:4 Structured Sparsity Fine-Tuning (6 Epochs)
- Pruned **9,975,808 parameters** (50.0% of candidate 2D conv weights).
- Effective compute reduced from 56.43 GFLOPs to **29.93 GFLOPs**.
- Final Sparse mAP50: **81.97%** O2O (**83.70%** NMS), retaining 96.7% of dense capability.

### Phase 2: Quantization-Aware Training (3 Epochs)
- Calibrated 120 quantizers using Histogram activation calibration and Max weight calibration.
- Observer dampening applied in epochs 2–3 with final BatchNorm re-estimation.
- Final QAT mAP50: **83.78%** O2O (**84.82%** NMS), recovering +1.81% from the sparse baseline.
- Final O2O vs NMS gap after INT8 quantization: **-1.05%**. Deployed INT8 engine size: **19.8 MB**.
