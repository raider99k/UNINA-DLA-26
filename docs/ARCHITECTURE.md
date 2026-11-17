# UNINA-DLA Architecture Specification

Technical documentation of the UNINA-DLA neural network architecture, hardware-aware design decisions, mathematical formulations, and execution constraints for the NVIDIA Jetson Orin Deep Learning Accelerator (DLA 2.0).

---

## 1. Design Overview & Edge Constraints

In high-speed autonomous systems such as Formula Student Driverless vehicles, perception latency directly limits vehicle state estimation and trajectory tracking. Running modern object detectors on the primary embedded GPU creates compute contention with other critical tasks, including LiDAR processing, SLAM, and Model Predictive Control (MPC).

UNINA-DLA is designed to execute with **100% residency on the NVIDIA Jetson Orin Deep Learning Accelerator (NVDLA 2.0)**, avoiding GPU fallback layers and memory transfer barriers. Offloading 2D cone detection to the DLA leaves the GPU available for downstream autonomy tasks.

To achieve full DLA residency and low latency, UNINA-DLA departs from stock YOLO architectures:
- **Backbone**: Structural re-parameterization (RepVGG-A1 adapted) replaces multi-branch residual topologies, providing high compute intensity in a single-path $3 \times 3$ convolutional pipeline, optionally topped with a DLA-native Spatial Pyramid Pooling Fast (`DLASPPF`) block on $P_5$.
- **Activation Functions**: Uniform ReLU replaces SiLU (Swish), executing in a single clock cycle on DLA Single Data Point (SDP) units and avoiding FP16 precision drops.
- **Feature Neck**: Rep-PAN with bounded Convolution Buffer (CBUF) SRAM memory footprint, configurable $1 \times 1$ pointwise or $1 \times 1 \to 3 \times 3$ spatial fusion, `groups=1/2` on $P_5$, and DLA-compatible upsampling (`nn.Upsample(mode="nearest")` or `DLAUpsample` `ConvTranspose2d`).
- **Detection Head**: Anchor-free YOLO26 One-to-One (O2O) direct coordinate regression head, eliminating Non-Maximum Suppression (NMS) and Distribution Focal Loss (DFL) Softmax operations.

---

## 2. Hardware Constraints: NVIDIA Jetson Orin DLA 2.0

> [!WARNING]
> **Hardware Verification Notice**: While the architecture is theoretically modeled to comply with NVDLA 2.0 specifications and 1 MiB CBUF SRAM limits, actual DLA hardware residency and zero-fallback execution have not been empirically verified on physical Jetson Orin hardware due to a lack of target hardware access.

The Orin SoC includes two independent DLA 2.0 cores. Unlike the streaming multiprocessors (SMs) of the GPU, each DLA is a fixed-function hardware dataflow accelerator comprising:
- **Convolution Core**: Accelerates INT8 (dense and 2:4 structured sparse) and FP16 matrix convolutions.
- **Single Data Point (SDP) Unit**: Pointwise operations, biases, scaling, and activation functions (ReLU, Sigmoid, TanH).
- **Planar Data Processor (PDP)**: Pooling and downsampling operations.
- **Channel Data Processor (CDP)**: Local response normalization.
- **Convolution Buffer (CBUF)**: 1 MiB (1,048,576 bytes) of dedicated on-chip high-bandwidth SRAM per core.

### 2.1 Convolution Buffer (CBUF) Hardware Residency Constraints

The 1 MiB CBUF (divided into 16 banks of 64 KiB on Jetson Orin DLA 2.0, or 512 KiB divided into 16 banks of 32 KiB on Xavier NVDLA 1.0) serves as dedicated on-chip high-bandwidth SRAM. Rather than functioning as a static cache that must hold an entire layer's multi-megabyte parameter tensor simultaneously, **CBUF is an on-chip circular streaming buffer** managed by the Convolution DMA (CDMA) and Convolution Sequence Controller (CSC). The hardware streams activation tiles and weights for one kernel group (MAC channel slice) at a time, cycling through output channels.

#### The Crucial TensorRT DLA CBUF Validation Rule
According to the official NVIDIA TensorRT DLA Architecture Specification, a convolution or deconvolution layer is verified for 100% native DLA execution without GPU fallback if and only if the total CBUF bank requirement satisfies:

$$\text{wtBanksForOneKernel} + \text{minDataBanks} \le 16$$

where the components are calculated as:

$$\text{wtBanksForOneKernel} = \text{uint32}\left(\left\lceil \frac{\text{roundUp}(C_{\text{in\_per\_group}} \times K_H \times K_W \times B_{\text{prec}},\, 128)}{32{,}768} \right\rceil\right)$$

$$\text{minDataBanks} = \text{uint32}\left(\left\lceil \frac{\text{entriesPerDataSlice} \times \text{dilatedKernelHt}}{256} \right\rceil\right)$$

with the intermediate data dimensions:

$$\text{entriesPerDataSlice} = \text{uint32}\left(\left\lceil \left\lceil \frac{C_{\text{in}} \times B_{\text{prec}}}{32} \right\rceil \times \frac{W_{\text{in}}}{4} \right\rceil\right)$$

$$\text{dilatedKernelHt} = (K_H - 1) \times \text{dilation}_H + 1$$

where $B_{\text{prec}} = 1\text{ byte}$ for INT8 and $2\text{ bytes}$ for FP16, and $C_{\text{in\_per\_group}} = C_{\text{in}} / G$.

#### Architectural Significance for Stage 4 ($P_5$)
Crucially, **$C_{\text{out}}$ does not appear in the numerator of $\text{wtBanksForOneKernel}$**. The CBUF only needs to store the kernel weights for a single output channel slice at any instant during MAC evaluation.

For UNINA-DLA Stage 4 ($C_{\text{in}} = 256$, $C_{\text{out}} = 512$, $3 \times 3$, $W_{\text{in}} = 48$ at stride 16):
- **INT8 ($B_{\text{prec}}=1$)**:
  - $\text{wtBanksForOneKernel} = \left\lceil \frac{\text{roundUp}(256 \times 3 \times 3 \times 1, 128)}{32{,}768} \right\rceil = \left\lceil \frac{2304}{32768} \right\rceil = \mathbf{1\text{ bank}}$
  - $\text{entriesPerDataSlice} = \lceil \lceil 256 / 32 \rceil \times 48 / 4 \rceil = 8 \times 12 = 96$
  - $\text{minDataBanks} = \left\lceil \frac{96 \times 3}{256} \right\rceil = \lceil 1.125 \rceil = \mathbf{2\text{ banks}}$
  - **Total CBUF Banks Needed = $1 + 2 = \mathbf{3\text{ banks} \ll 16\text{ banks}}$**
- **FP16 ($B_{\text{prec}}=2$)**:
  - $\text{wtBanksForOneKernel} = \mathbf{1\text{ bank}}$, $\text{minDataBanks} = \mathbf{3\text{ banks}}$
  - **Total CBUF Banks Needed = $1 + 3 = \mathbf{4\text{ banks} \ll 16\text{ banks}}$**

This proves mathematically that standard un-grouped convolutions (`groups=1`) on Stage 4 consume only **3 out of 16 banks in INT8** (and 4 out of 16 in FP16), well within DLA silicon limits.

### 2.2 DLA Supported Operations
- **Padding Restriction**: DLA requires spatial padding to be strictly less than the kernel size:
  $$\text{Pad} < \text{Kernel Size}$$
- **Activations**: ReLU executes natively in 1 cycle on the SDP unit. Non-linearities like SiLU require runtime interpolation via look-up tables in FP16, halving effective INT8 throughput.
- **Static Dimensions**: DLA requires static tensor shapes determined at TensorRT engine compile time.
- **Avoiding GPU Fallback**: If an operation is unsupported by DLA, TensorRT inserts a GPU fallback node (`kGPU_FALLBACK`). Crossing between the DLA and GPU memory domains requires host synchronization and memory copies, negating the latency benefit. UNINA-DLA enforces zero GPU fallback.

---

## 3. Network Architecture

```
                          [ Input Image: 1x3xHxW ]
                                     |
                                     v
                      +-----------------------------+
                      |    RepVGG-A1 Backbone       |
                      |  (Fused 3x3 Convs + ReLU)   |
                      |   + Optional DLASPPF (P5)   |
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             | P3 (Stride 8)         | P4 (Stride 16)        | P5 (Stride 32)
             v                       v                       v
     +---------------+       +---------------+       +---------------+
     |  Rep-PAN Neck |<----->|  Rep-PAN Neck |<----->|  reduce_p5    |
     | (Nearest/Dec) |       | (Nearest/Dec) |       | (groups=1/2)  |
     +-------+-------+       +-------+-------+       +-------+-------+
             |                       |                       |
             v                       v                       v
     +---------------+       +---------------+       +---------------+
     |  O2O Head P3  |       |  O2O Head P4  |       |  O2O Head P5  |
     | Reg: 4 / Cls  |       | Reg: 4 / Cls  |       | Reg: 4 / Cls  |
     +---------------+       +---------------+       +---------------+
             |                       |                       |
             +-----------------------+-----------------------+
                                     |
                                     v
                  [ Direct Output Tensors: Linear LTRB + Sigmoid Cls ]
                      (Zero NMS, Zero Softmax, Zero DFL)
```

### 3.1 Model Generations & Default Configuration

The codebase supports multiple architecture generations in `unina_dla/model/unina_dla.py` (`UNINA_DLA`), with `detect_architecture_version()` and `detect_upsample_mode()` auto-detecting checkpoint layout during fine-tuning, validation, and export:

| Version | Backbone Blocks | Stage Channels `(P2..P5)` | `DLASPPF` | `RepPAN` Fusion | `Stage 4` / `P5` Groups | Default Upsample | `reg_relu` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3.2` (Current Default)** | `[2, 4, 14, 1]` | `(64, 128, 256, 512)` | Yes | Spatial ($1{\times}1 \to 3{\times}3$) | `1` / `1` | `deconv` | `False` (Unbounded) |
| **`2.6`** | `[2, 4, 14, 1]` | `(64, 128, 256, 512)` | No | Pointwise ($1{\times}1$) | `1` / `1` | `nearest` | `True` |
| **`2.5` (Legacy Default)** | `[2, 4, 14, 1]` | `(64, 128, 256, 512)` | No | Pointwise ($1{\times}1$) | `2` / `2` | `deconv` | `True` |
| **`3.1`** | `[2, 4, 14, 1]` | `(64, 128, 256, 256)` | Yes | Spatial ($1{\times}1 \to 3{\times}3$) | `1` / `1` | `deconv` | `True` |
| **`3`** | `[2, 6, 8, 2]` | `(64, 128, 256, 256)` | Yes | Spatial ($1{\times}1 \to 3{\times}3$) | `1` / `1` | `deconv` | `True` |
| **`2`** | `[2, 4, 14, 1]` | `(64, 128, 256, 512)` | No | Pointwise ($1{\times}1$) | `2` / `2` | `deconv` | `True` |

All generations allocate neck channels as `[256, 192, 128]` for `(P3, P4, P5)`, except Generation `2` which uses `[192, 192, 128]`. Fresh student runs in `scripts/train_student.py` default to `version="3.2"`, while `upsample_mode` can be overridden per run (`nearest`, `deconv`, or `bilinear`).

1. **Generation 3.2 (Current Default)**:
   - **Backbone**: RepVGG-A1 `[2, 4, 14, 1]` with widths `(64, 128, 256, 512)`, `groups=1` throughout, and `DLASPPF` on $P_5$.
   - **Neck**: Spatial Rep-PAN (`1x1 -> 3x3` `RepVGGBlock` fusion) with `[256, 192, 128]` channels and learnable amplitude-preserving `DLAUpsample` deconvolution by default (also compatible with `nearest`).
   - **Head**: Unbounded direct LTRB regression (`reg_relu=False`), removing the ReLU clamp on regression outputs so sub-stride cones can emit negative `l`/`t` (or `r`/`b`) offsets.
   - **ImageNet Pretraining**: Full Stage 0–4 RepVGG-A1 weight transfer via L1-norm importance channel slicing on Stage 4 (`1280 -> 512`).

2. **Generation 2.6**:
   - **Backbone & Neck**: RepVGG-A1 `[2, 4, 14, 1]`, `(64, 128, 256, 512)`, `groups=1` throughout, pointwise $1 \times 1$ Rep-PAN (`[256, 192, 128]`), and zero-parameter `nn.Upsample(scale_factor=2, mode="nearest")` (shedding 212,992 parameters vs `DLAUpsample`).
   - **Characteristics**: 21.93M deploy parameters (with `nearest`) / 22.14M (with `deconv`), 56.0 GFLOPs, 100% Stage 0–4 pretraining coverage, and 3/16 modeled INT8 CBUF banks.

3. **Generation 2.5 (Legacy Default)**:
   - **Backbone & Neck**: `[2, 4, 14, 1]`, `(64, 128, 256, 512)`, `groups=2` on Stage 4 and `reduce_p5`, pointwise $1 \times 1$ Rep-PAN (`[256, 192, 128]`), and `DLAUpsample`.
   - **Characteristics**: 21.26M deploy parameters, 55.4 GFLOPs; Stage 0–3 pretrained (93.8% coverage).

4. **Generations 3.1, 3, and 2**:
   - **Gen 3.1 / 3**: Narrower 256-ch Stage 4 (`[2, 4, 14, 1]` for 3.1; `[2, 6, 8, 2]` for 3) with `DLASPPF` and spatial Rep-PAN (`Gen 3.1`: 22.82M params, 60.9 GFLOPs).
   - **Gen 2**: Slimmer neck `[192, 192, 128]` with `groups=2` on Stage 4 and `reduce_p5`.

---

## 4. Architectural Details

### 4.1 RepVGG Backbone & Structural Re-Parameterization

During training, each RepVGG block comprises three parallel branches:
1. $3 \times 3$ convolution with BatchNorm: $\text{BN}_3(M * W^{(3)})$
2. $1 \times 1$ convolution with BatchNorm: $\text{BN}_1(M * W^{(1)})$
3. Identity mapping with BatchNorm: $\text{BN}_{\text{id}}(M)$ (when input and output shapes match)

$$M_{\text{out}} = \text{ReLU}\left( \text{BN}_3(M * W^{(3)}) + \text{BN}_1(M * W^{(1)}) + \text{BN}_{\text{id}}(M) \right)$$

#### Mathematical Folding for Deployment
At deployment time, the multi-branch structure collapses into a single $3 \times 3$ convolution:

1. **BatchNorm Folding**:
   $$W'_{i, j, :, :} = \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}} W_{i, j, :, :}, \quad b'_i = \beta_i - \frac{\mu_i \gamma_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
2. **Branch Kernel Merging**:
   - The $1 \times 1$ kernel $W'^{(1)}$ is zero-padded to $3 \times 3$ symmetrically.
   - The identity branch is constructed as an identity matrix and zero-padded to $3 \times 3$.
3. **Consolidated Single-Path Kernel**:
   $$W_{\text{deploy}} = W'^{(3)} + \text{PAD}_{3 \times 3}(W'^{(1)}) + \text{PAD}_{3 \times 3}(I)$$
   $$b_{\text{deploy}} = b'^{(3)} + b'^{(1)} + b'_{\text{id}}$$

The deployed layer executes as:
$$M_{\text{out}} = \text{ReLU}\left( M * W_{\text{deploy}} + b_{\text{deploy}} \right)$$

#### `DLASPPF` (DLA-Native SPPF) & ImageNet Pretraining
- **`DLASPPF` (`unina_dla/model/backbone/repvgg_backbone.py`)**: Enabled on $P_5$ in Generations `3`, `3.1`, and `3.2`. Halves channels via a $1 \times 1$ `RepVGGBlock`, applies three chained `MaxPool2d(kernel_size=5, stride=1, padding=2)` operations (mapping to the DLA Planar Data Processor with $\text{pad}=2 < \text{kernel}=5$), concatenates `[x, y1, y2, y3]`, and projects back via a $1 \times 1$ `RepVGGBlock`. During QAT, a `shared_quantizer` is attached so all four concatenated branches share identical INT8 quantization scales (NVDLA DMA zero-requantization Concat rule).
- **ImageNet Pretraining (`load_pretrained_backbone` in `unina_dla/utils/model_utils.py`)**: Translates `timm`, `holocron`, and `DingXiaoH` RepVGG-A1 checkpoints to UNINA-DLA backbone keys. For 512-channel `groups=1` Stage 4 layouts (Gen `2.6` and `3.2`), it selects the top 512 filters from RepVGG-A1's 1,280-channel Stage 4 by L1-norm importance (weighted by BN scale and variance when available).

### 4.2 Rep-PAN Neck
- **SOD-Priority Channels**: Distant cones in racing datasets span few pixels and reside primarily in the $P_3$ stride-8 feature map. Gen `2.5`, `2.6`, `3`, `3.1`, and `3.2` allocate channels as `[256, 192, 128]` for $(P_3, P_4, P_5)$.
- **Pointwise vs. Spatial Fusion (`make_fusion`)**: Generations `2`, `2.5`, and `2.6` fuse concatenated top-down/bottom-up features with a single $1 \times 1$ `RepVGGBlock`, whereas Generations `3`, `3.1`, and `3.2` (`spatial_fusion=True`) use a sequential $1 \times 1 \to 3 \times 3$ `RepVGGBlock` pair.
- **$P_5$ Convolution Grouping Evolution**:
  - **Gen 2.5**: Applied `groups=2` on Stage 4 and `reduce_p5` under the conservative theoretical assumption that entire layer parameter tensors must remain below 1.0 MB.
  - **Gen 2.6 / 3.2**: Restores standard un-grouped convolutions (`groups=1`) across Stage 4 and `reduce_p5`. As proven by the TensorRT DLA validation equation ($\text{wtBanksForOneKernel} + \text{minDataBanks} = 3 \le 16$), un-grouped convolutions are 100% hardware resident. This eliminates channel partitioning and enables full ImageNet pretraining transfer across the entire backbone.
- **Upsampling Modes (`nn.Upsample` vs. `DLAUpsample`)**:
  - **`DLAUpsample` (`upsample_mode="deconv"`)**: Uses `nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, groups=1, bias=False)` with zero padding (ungrouped for strict DLA 2.0 residency). Its kernel is initialized with $2 \times 2$ blocks of `1.0` along the input-output channel diagonal (`get_bilinear_kernel`), preserving exact $1.0\times$ feature amplitude at step 0 (avoiding $0.25\times$ attenuation) while allowing weights to adapt during training.
  - **`nn.Upsample` (`upsample_mode="nearest"`)**: Zero-parameter 2D nearest-neighbor resize with scale factor 2, eliminating 212,992 parameters and avoiding deconvolution-specific sparsity/QAT exclusions. (`upsample_mode="bilinear"` is also supported for experimentation.)

### 4.3 YOLO26 Decoupled Dual Head
Standard anchor-based or NMS-dependent detectors require runtime non-maximum suppression:
- **DFL Removal & `reg_relu` Regression**: Standard YOLO models predict 16-bin box distributions using Softmax and an expectation integral. UNINA-DLA predicts 4 direct LTRB box offsets (`reg_max=0`) linearly without Softmax. Generations `2`–`3.1` apply `nn.ReLU()` at the regression output (`reg_relu=True`, initial biases `[2.0, 4.0, 4.0]`), whereas Generation `3.2` sets `reg_relu=False` (`nn.Identity()`, initial biases `[1.0, 2.0, 2.0]`) so `dist2bbox` can represent negative individual `l`/`t` (or `r`/`b`) offsets for off-center sub-stride cones.
- **Decoupled Training Towers & Prior Initialization**: In `unina_dla/model/head/yolo26_head.py`, each level passes through a shared `RepVGGBlock` stem before splitting into dedicated O2O (`o2o_cls_convs`, `o2o_reg_convs`) and auxiliary O2M (`o2m_cls_convs`, `o2m_reg_convs`) two-block towers. Classification heads use `std=0.01` normal weight initialization with RetinaNet negative prior bias $b = -\ln((1 - \pi)/\pi)$ for $\pi = 0.01$.
- **Deployment Output**: At deployment (`deploy=True`), only the O2O tower is executed. `UNINA_DLA.forward()` applies `.sigmoid()` to the classification logits and returns a 6-tensor tuple:
  $$(\text{reg}_3, \text{reg}_4, \text{reg}_5, \text{cls}_3, \text{cls}_4, \text{cls}_5)$$

---

## 5. Training Methodology

### 5.1 Standalone Ground-Truth Training (Default)
The primary Phase 1 training pipeline (`unina_dla/engines/phase1_distill.py` with `--enable_distillation` unset) trains the student directly on labeled ground truth:
1. **Small-Object Assignment & $top\text{-}k$ Curriculum (`SmallObjectTaskAlignedAssigner` in `v26_loss.py`)**:
   - Expands positive candidate sampling boxes around each GT center to at least $\pm 4.5\text{ px}$ ($0.5625 \times \text{stride}_{P_3}$) so sub-stride cones always overlap $P_3$ anchors.
   - Applies a **minimum-distance fallback** for unassigned GT boxes, restricting small objects ($\text{area} < 1024\text{ px}^2$) to $P_3$ anchors with IoU-calibrated soft target scores.
   - Anneals O2O TAL $top\text{-}k$ across a 3-stage curriculum (`o2o_topk_start -> o2o_topk_mid -> o2o_topk_end`, defaulting to `3 -> 2 -> 1` at `0.15` and `0.35` phase progress in Phase 1; locked to `1` in Sparsity and QAT).
2. **Dual-Head Loss & Dynamic Auxiliary Decay (`DualYOLO26LossAdapter`)**:
   - Combines the O2O deploy-branch loss with the auxiliary O2M branch (`o2m_topk=10`) weighted by a cosine schedule from `o2m_aux_weight_start` ($0.8$) to `o2m_aux_weight_end` ($0.0$ by default, fixed at $0.0$ in Sparsity/QAT).
3. **Classification Loss (`ProgLoss` + Smooth STAL)**:
   - Uses `ProgLoss` ($L_{\text{BCE}}(p, t) \cdot \text{clamp}(1 + |p - t|,\, 1,\, 2)$) modulated by Smooth STAL small-object weighting $w_{\text{STAL}} = 1 + \sigma(10 - \sqrt{\text{area}} / 3.2)$, transitioning smoothly from $2\times$ for tiny cones to $1\times$ for large cones around $32 \times 32\text{ px}$ (`hyp.cls=0.5`).
4. **Box Regression (`Wise-IoU v3` + Normalized L1)**:
   - Combines `Wise-IoU v3` (`wise_iou` with detached enclosing diagonal `diag2.detach()`, `hyp.box=7.5`) and a scale-invariant image-normalized L1 LTRB coordinate loss (`hyp.dfl=0.5`), both weighted by Smooth STAL.
5. **Scheduled O2O Duplicate & Hard-Background Suppression (`v26_loss.py`)**:
   - Ramps in linearly after `dup_start_pct=0.6`: a margin ranking penalty on high-IoU duplicate negative anchors (`dup_loss_weight=0.05`, `dup_margin=0.2`, `dup_iou_thr=0.6`, `dup_topk=20`) and a hard-background BCE penalty (`hard_bg_loss_weight=0.02`, `hard_bg_iou_thr=0.2`, `hard_bg_topk=64`).
6. **Optimizer, EMA & Schedules (`MuSGD`, `UnifiedTrainer`)**:
   - Defaults to hybrid `MuSGD` (`unina_dla/utils/optimizers.py`: 5-step Newton-Schulz orthogonalization + SGD on $\ge 2\text{D}$ weights, Nesterov SGD on $1\text{D}$ parameters, optional `--backbone_lr_ratio`, with `AdamW` and `SGD` also supported), linear warmup, cosine LR decay, `ModelEMA`, `close_mosaic`, `--formula_image` static rectangular canvas with shuffling, and phase-isolated `EarlyStopping`.
   - A `--classic` baseline mode (`ClassicDetectionLoss`, O2M-only $top\text{-}k=10$, NMS validation) is also provided for ablation.

### 5.2 Knowledge Distillation (Experimental)
The repository includes an optional Tri-Vector Knowledge Distillation mode (`--enable_distillation`) from a frozen YOLO26 teacher:
- **Scale-Decoupled Feature (SDF) Distillation**: Mean-magnitude normalized masked MSE with foreground O2O mask injection (so weak teacher activations on small cones are not ignored), a log-magnitude anchor, and optional global relation matching (`--kd_mech_global_rel`, channel + spatial affinity).
- **Logit Distillation**: Temperature-scaled ($T=4.0$) log-sigmoid certainty MSE + foreground/background-weighted BCE (`w_fg=2.0`, `w_bg=1.0`) aligned with teacher O2O masks.
- **Box Distillation**: Confidence-weighted `Wise-IoU v3` between student and teacher O2O LTRB predictions, plus optional CrossKD routing (`--kd_mech_cross_head`) through the frozen teacher O2O head.
- **Uncertainty Balancing & Fade Schedules**: Balanced via `AdaptiveLossBalancer` (`num_losses=3` homoscedastic uncertainty weights, soft-reset at phase transitions), cosine `teacher_fade` ($0.8 \to 0.1$), inverse cosine `task_weight` ($0.35 \to 1.0$), and per-vector `kd_{sdf,logit,box}` multipliers.

*Experimental finding*: Teacher soft labels introduce spatial smoothing across neighboring anchors that degrades Hungarian one-to-one single-positive selection. Standalone ground-truth training achieved higher mAP50 than distilled models, making distillation an optional experimental mode.

### 5.3 2:4 Structured Sparsity Fine-Tuning
To leverage Ampere and Orin Sparse Tensor Cores (`unina_dla/engines/phase1_5_sparsity.py`, `unina_dla/utils/sparsity_utils.py`):
- RepVGG blocks are fused (`switch_to_deploy`) before optimizer construction so AdamW (`lr=1e-4`) tracks `rbr_reparam` weights directly.
- Sensitive structural layers (`DEFAULT_SPARSE_IGNORE_PATTERNS`: `stem`, `upsample`, `DLAUpsample`, and final `cls_preds`/`reg_preds` heads) are excluded. Because name-based exclusions are not guaranteed under APEX ASP, the Pure PyTorch backend prunes 2 of every 4 weights along $C_{\text{in}}$ (50% layer sparsity).
- Fine-tuning enforces zero-masking via both `create_zero_grad_hook` (zeroing pruned-weight gradients prior to clipping/stepping to prevent momentum drift) and `create_zero_clamp_hook` (clamping pruned weights post-step), verified at phase end with `verify_sparsity(ema.ema)`.

### 5.4 Quantization-Aware Training (QAT)
To simulate INT8 DLA numerics (`unina_dla/engines/phase2_qat.py`, `unina_dla/utils/qat_utils.py`):
- **Pre-QAT Fusion**: RepVGG branches are fused (`switch_to_deploy`), followed by `fuse_conv_bn_recursive` before quantizer insertion.
- **Quantizer Placement & Descriptors**: NVIDIA `pytorch_quantization` replaces `Conv2d` with `QuantConv2d` and `ConvTranspose2d` with `PatchedQuantConvTranspose2d` using signed histogram activation descriptors and per-channel max (`axis=0`) weight descriptors. `shared_quantizer` is attached to `DLASPPF` for concat scale invariance.
- **Sensitive Layer Exemption**: Final prediction heads and upsample layers (`ignore_modules=["preds", "upsample"]`) remain in FP32/FP16.
- **Calibration & Observer Dampening**: Calibrated over validation images (`calib_batches=200`, percentile `99.99%`) before fine-tuning (Mosaic disabled). During QAT, `update_weight_quantizers` tracks weight `amax` values until `freeze_quantization_observers` locks all observers in the final 2 epochs, followed by post-QAT `bn_reestimate` (which preserves calibrated `_amax` buffers on fused models).

---

## 6. Export, DLA Verification & Deployment Integration

- **Phase 3 ONNX Export (`unina_dla/engines/phase3_export.py`, `scripts/export_onnx.py`)**:
  - Exports the O2O-only 6-tensor deploy interface (`reg3..reg5`, `cls3..cls5`) at opset 17 using the legacy ONNX tracer (`dynamo=False`) with static shapes by default (optional `--dynamic`, `--external-data`).
  - Supports FP16 export for dense/sparse models, FakeQuant QAT ONNX export, and `--qat-clean-deploy`, which strips quantizer tensors and remaps `PatchedQuantConvTranspose2d` keys into a clean FP32 ONNX paired with `*_int8_scales.json` and `*_manifest.json`.
- **Strict DLA ONNX Verification (`unina_dla/utils/verify_dla_onnx.py`)**:
  - Automatically runs after export (disable with `--no-verify-dla`), validating ONNX schema, `FORBIDDEN_OPS` (`Softmax`, `Gather`, `NonZero`, `TopK`, `QuantizeLinear`, `DequantizeLinear`, `Erf`, `ReduceSum`, `ScatterND`, `NonMaxSuppression`, `Where`), `Conv` dilation/padding ($\text{pad} < \text{kernel}$), `ConvTranspose` rules ($\text{group}=1$, zero padding, $\text{dilation}=1$, $\text{kernel} \le 32$), absence of `o2m` branch leakage, `Add/Conv` fusion ratio, and absence of `Sigmoid -> Mul` (SiLU) patterns.
- **Phase 4 Test Evaluation, TensorRT Build & Ultralytics Wrapper**:
  - `scripts/train_student.py` evaluates the final checkpoint on the `test` split (when available) at both `conf=0.001` and deployment operating thresholds (`--deploy_eval_conf 0.25`, `--deploy_eval_iou 0.6`).
  - `scripts/build_engine.py` compiles TensorRT engines (`trtexec` or TensorRT Python API) with `--dlaCore`, `--useDLACore`, strict DLA / GPU-fallback control, DLA-native `fp16:dla_hwc4` input and `fp16:chw16` output formats, and INT8 calibration cache / dynamic-range injection from QAT `*_int8_scales.json`.
  - `unina_dla/ultralytics_wrapper.py` (`UNINA` / `UNINADetectionPredictor`) and `scripts/convert_to_ultralytics_pt.py` enable Ultralytics API inference over UNINA `.pt`, `.onnx`, and `.engine` artifacts.
