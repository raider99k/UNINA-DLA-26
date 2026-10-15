# UNINA-DLA Architecture Specification

Technical documentation of the UNINA-DLA neural network architecture, hardware-aware design decisions, mathematical formulations, and execution constraints for the NVIDIA Jetson Orin Deep Learning Accelerator (DLA 2.0).

---

## 1. Design Overview & Edge Constraints

In high-speed autonomous systems such as Formula Student Driverless vehicles, perception latency directly limits vehicle state estimation and trajectory tracking. Running modern object detectors on the primary embedded GPU creates compute contention with other critical tasks, including LiDAR processing, SLAM, and Model Predictive Control (MPC).

UNINA-DLA is designed to execute with **100% residency on the NVIDIA Jetson Orin Deep Learning Accelerator (NVDLA 2.0)**, avoiding GPU fallback layers and memory transfer barriers. Offloading 2D cone detection to the DLA leaves the GPU available for downstream autonomy tasks.

To achieve full DLA residency and low latency, UNINA-DLA departs from stock YOLO architectures:
- **Backbone**: Structural re-parameterization (RepVGG-A1 adapted) replaces multi-branch residual topologies, providing high compute intensity in a single-path $3 \times 3$ convolutional pipeline.
- **Activation Functions**: Uniform ReLU replaces SiLU (Swish), executing in a single clock cycle on DLA Single Data Point (SDP) units and avoiding FP16 precision drops.
- **Feature Neck**: Rep-PAN with bounded Convolution Buffer (CBUF) SRAM memory footprint, grouped convolutions on $P_5$, and deconvolutional upsampling (`DLAUpsample`).
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
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             | P3 (Stride 8)         | P4 (Stride 16)        | P5 (Stride 32)
             v                       v                       v
     +---------------+       +---------------+       +---------------+
     |  Rep-PAN Neck |<----->|  Rep-PAN Neck |<----->|  reduce_p5    |
     | (DLAUpsample) |       | (DLAUpsample) |       | (groups=1/2)  |
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
                  [ Direct Output Tensors: Linear LTRB + Cls ]
                      (Zero NMS, Zero Softmax, Zero DFL)
```

### 3.1 Model Generations & Default Configuration

The codebase supports multiple architecture configurations (`unina_dla/model/unina_dla.py`):

1. **Generation 2.6 (Hardware-Verified Production Champion)**:
   - **Backbone**: RepVGG-A1 with block counts `[2, 4, 14, 1]`, channel widths `(64, 128, 256, 512)`, and standard dense convolutions (`groups=1` throughout, including Stage 4).
   - **Neck**: Pointwise $1 \times 1$ Rep-PAN with Small-Object Detection (SOD) priority channels: `[256, 192, 128]`, with `reduce_p5` using `groups=1`.
   - **Upsampling Evolution**: DLA-native zero-parameter nearest upsampling (`nn.Upsample(scale_factor=2, mode="nearest")`) replacing legacy deconvolution (`ConvTranspose2d`), shedding 212,992 parameters and eliminating deconvolution weights from CBUF/DRAM.
   - **ImageNet Pretraining**: 100% backbone pretraining coverage across all 4 stages (0 to 4) via official RepVGG-A1 weights, eliminating random initialization on Stage 4.
   - **Characteristics**: 21.93M deploy parameters (with nearest upsample) / 22.14M (with legacy deconv), 56.0 GFLOPs. Eliminates channel isolation in Stage 4 and P5 neck fusion, while maintaining peak CBUF SRAM usage at only 3 out of 16 banks.

2. **Generation 2.5 (Legacy Production Champion)**:
   - **Backbone**: RepVGG-A1 with block counts `[2, 4, 14, 1]` and channel widths `(64, 128, 256, 512)` using `groups=2` on Stage 4.
   - **Neck**: Pointwise $1 \times 1$ Rep-PAN with SOD channels `[256, 192, 128]` using `groups=2` on `reduce_p5`.
   - **Characteristics**: 21.26M deploy parameters, 55.4 GFLOPs. Introduced under a conservative theoretical heuristic bounding total layer weight volume below 1 MB. Stages 0 to 3 pretrained (93.8% coverage).

3. **Generation 3.1 (Experimental Spatial Neck)**:
   - **Backbone**: Block counts `[2, 4, 14, 1]`, channel widths `(64, 128, 256, 256)`, with DLASPPF on $P_5$.
   - **Neck**: Spatial Rep-PAN with paired $1 \times 1 \to 3 \times 3$ RepVGG fusion blocks and channels `[256, 192, 128]`.
   - **Characteristics**: 22.82M deploy parameters, 60.9 GFLOPs. Reached statistical parity in dense training (84.69% vs 84.78% mAP50) but added 1.56M parameters and experienced slightly higher quantization loss in INT8 (83.19% vs 83.78%).

4. **Generation 2**:
   - Backbone `[2, 4, 14, 1]`, channels `(64, 128, 256, 512)`, neck `[192, 192, 128]`.

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

### 4.2 Rep-PAN Neck
- **SOD-Priority Channels**: Distant cones in racing datasets span few pixels and reside primarily in the $P_3$ stride-8 feature map. Gen 2.5 and Gen 2.6 allocate channels as `[256, 192, 128]` for $(P_3, P_4, P_5)$.
- **$P_5$ Convolution Grouping Evolution**:
  - **Gen 2.5**: Applied `groups=2` on Stage 4 and `reduce_p5` under the conservative theoretical assumption that entire layer parameter tensors must remain below 1.0 MB.
  - **Gen 2.6**: Restores standard un-grouped convolutions (`groups=1`) across Stage 4 and `reduce_p5`. As proven by the TensorRT DLA validation equation ($\text{wtBanksForOneKernel} + \text{minDataBanks} = 3 \le 16$), un-grouped convolutions are 100% hardware resident. This eliminates channel partitioning and enables full ImageNet pretraining transfer across the entire backbone.
- **Upsampling Evolution (`nn.Upsample` vs `DLAUpsample`)**:
  - **Gen 1 & Gen 2.5 (`DLAUpsample`)**: In earlier TensorRT releases (7.x / 8.0 on Xavier), dynamic bilinear interpolations occasionally triggered GPU fallback. UNINA-DLA used `nn.ConvTranspose2d` with fixed stride 2 and fixed diagonal bilinear weights ($65{,}536 + 147{,}456 = 212{,}992$ parameters). While resident, this required storing 213K weights in DRAM/CBUF and adding special exclusion rules to 2:4 sparsity and QAT.
  - **Gen 2.6 (`nn.Upsample`)**: On Jetson Orin DLA 2.0 (TensorRT 8.5+), 2D `Resize` with `mode="nearest"` and integer scale factor 2 is natively accelerated on DLA hardware without GPU fallback. Adopting `nn.Upsample(scale_factor=2, mode="nearest")` eliminates all 212,992 weights, removes all sparsity/QAT deconvolution workarounds, and delivers zero-parameter upsampling.

### 4.3 YOLO26 Decoupled Dual Head
Standard anchor-based or NMS-dependent detectors require runtime non-maximum suppression:
- **DFL Removal**: Standard YOLO models predict 16-bin box distributions using Softmax and an expectation integral. UNINA-DLA predicts 4 direct bounding box coordinates (LTRB offsets) linearly without Softmax.
- **Decoupled Training Towers**: In `unina_dla/model/head/yolo26_head.py`, separate convolutional towers are maintained during training:
  - `o2o_cls_convs` and `o2o_reg_convs`: Specialized for single-positive assignment ($top\text{-}k = 1$).
  - `o2m_cls_convs` and `o2m_reg_convs`: Auxiliary branch for dense supervision ($top\text{-}k = 7..10$).
  Separating the towers prevents dense auxiliary gradients from diluting the sharp peak suppression of the O2O tower.
- **Deployment Output**: At deployment (`deploy=True`), only the O2O tower is kept. The model outputs a tuple of 6 tensors:
  $$(\text{reg}_3, \text{reg}_4, \text{reg}_5, \text{cls}_3, \text{cls}_4, \text{cls}_5)$$
  with class Sigmoids applied. The C++ runtime directly decodes bounding boxes with zero host-GPU copies.

---

## 5. Training Methodology

### 5.1 Standalone Ground-Truth Training (Default)
The primary training pipeline trains the student directly on labeled data using:
1. **Dual-Head Loss Adapter**: Combines O2O loss with auxiliary O2M loss.
2. **Dynamic Auxiliary Loss Decay**: A cosine schedule smoothly fades the auxiliary loss weight:
   $$\alpha_{\text{aux}}(t) = \alpha_{\text{end}} + \frac{1}{2}(\alpha_{\text{start}} - \alpha_{\text{end}})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$
   decaying from $0.8$ to $0.0$, allowing the O2O branch to specialize fully by the end of training.
3. **Wise-IoU v3 Box Loss**: Employs detached diagonal calculation (`diag2.detach()`) to stabilize small-object gradient steps.
4. **Binary Cross-Entropy**: Clean BCE with logits for classification.

### 5.2 Knowledge Distillation (Experimental)
The repository includes an implementation of Tri-Vector Knowledge Distillation (`--enable_distillation`):
- **Scale-Decoupled Feature (SDF) Distillation**: Aligns intermediate student features to teacher representations.
- **Logit Distillation**: Matches classification logit distributions via temperature-scaled BCE.
- **Box Distillation**: Wise-IoU v3 distillation between student and teacher bounding boxes.

*Experimental finding*: Teacher soft labels introduce spatial smoothing across neighboring anchors that degrades Hungarian one-to-one single-positive selection. Standalone ground-truth training achieved $+0.91\%$ higher mAP50 than distilled models, making distillation an optional experimental mode.

### 5.3 2:4 Structured Sparsity Fine-Tuning
To leverage Ampere and Orin hardware Sparse Tensor Cores:
- In every consecutive 4 values along input channels, 2 are pruned ($50\%$ sparsity).
- Fine-tuning applies weight-mask and gradient-mask hooks after each optimizer step.

### 5.4 Quantization-Aware Training (QAT)
- **Pre-QAT Fusion**: RepVGG blocks are fused into single-path convolutions *before* quantizer insertion so quantization simulation accurately reflects inference numerics.
- **Quantization Placement**: NVIDIA `pytorch-quantization` inserts fake-quantization QDQ nodes.
- **Sensitive Layer Exemption**: Coordinate prediction heads (`o2o_reg_preds`) remain in FP16 to avoid metric localization precision loss.
- **Calibration & BN Re-estimation**: Calibrated with histogram activation observers and final BatchNorm re-estimation over un-augmented validation images.
