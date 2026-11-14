"""
Tri-Vector Knowledge Distillation Losses for UNINA-DLA
=======================================================

This module implements the "Tri-Vector" distillation strategy, a specialized framework
designed for Small Object Detection (SOD) on NVIDIA DLA hardware. It is a custom
synthesis of state-of-the-art research (ScaleKD, CVPR 2023) and hardware-aware optimizations.

Architecture Overview:
----------------------
    Vector 1 (SDF):   Feature Mimicry  → Scale-Decoupled + Masked MSE + Magnitude Anchor (Anti-Collapse)
    Vector 2 (Logit): Response Mimicry → Log-Sigmoid MSE (Certainty Matching) + O2O Weighted
    Vector 3 (Box):   Localization     → Wise-IoU v3 (Dynamic Focusing) + O2O Masking

Why NOT Standard Approaches?
----------------------------
    ╔══════════════════════╦═════════════════════════════╦════════════════════════════════════╗
    ║ Component            ║ Industry "Standard"         ║ UNINA-DLA (This Implementation)   ║
    ╠══════════════════════╬═════════════════════════════╬════════════════════════════════════╣
    ║ Feature Loss         ║ L2 on raw features          ║ Masked + Normalized MSE (ScaleKD) ║
    ║                      ║ → Background dominates loss ║ → Foreground-focused              ║
    ╠══════════════════════╬═════════════════════════════╬════════════════════════════════════╣
    ║ Logit Loss           ║ KL-Div with Softmax + Temp  ║ BDKD (Weighted MSE) + O2O Mask    ║
    ║                      ║ → Assumes mutual exclusion  ║ → Weights matched objects higher  ║
    ║                      ║ → Incompatible with BCE     ║ → Structurally aligned with O2O   ║
    ╠══════════════════════╬═════════════════════════════╬════════════════════════════════════╣
    ║ Box Loss             ║ DFL distribution + NMS      ║ Wise-IoU v3 (Dynamic Focusing)    ║
    ║                      ║ → Requires Softmax decode   ║ → Consistent with Task Loss       ║
    ╚══════════════════════╩═════════════════════════════╩════════════════════════════════════╝

References:
-----------
    [1] ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector (CVPR 2023)
        https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.pdf

    [2] Knowledge Distillation: A Good Teacher is Patient... (arXiv:2105.08919)
        https://arxiv.org/abs/2105.08919
        → Confirms MSE often outperforms KL when teacher/student gap is large

    [3] YOLO26 One-to-One Head - Avoids NMS via Hungarian matching during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFDistillationLoss(nn.Module):
    """
    Vector 1: Scale-Decoupled Feature (SDF) Distillation
    =====================================================

    Transfers intermediate feature representations from Teacher to Student while
    explicitly addressing the "small object" problem where background noise dominates.

    Design Rationale:
    -----------------
    Standard feature distillation (raw L2 loss) fails for small objects because:
        - 99% of feature map pixels are background
        - Large background regions dominate the loss gradient
        - Student wastes capacity learning irrelevant texture (clouds, track marks)

    This implementation uses two key techniques from ScaleKD (CVPR 2023):

    1. **Scale Decoupling (Normalization):**
       Divides features by global mean absolute value to decouple "intensity" from
       "spatial pattern". Forces student to learn the *topography* of activations,
       not their absolute magnitude (which depends on network depth/width).

       Formula: F_norm = F / (mean(|F|) + ε)

    2. **Spatial Attention Masking:**
       Generates a mask M from Teacher's activation magnitude, focusing the loss
       *only* on regions the Teacher finds important (i.e., cones, not sky).

       Formula: L_feat = Σ_l || (Norm(φ(F_S)) - Norm(F_T)) ⊙ M ||²

    3. **Magnitude Anchor (Anti-Collapse):**
       Prevents the Student from "cheating" the normalized MSE by outputting near-zero
       feature maps. Forces the Student's average magnitude to stay within a reasonable
       log-ratio of the Teacher's magnitude.

    4. **GT Injection (Blind Teacher Fix):**
       Injects Ground Truth regions into the mask M to ensure distillation occurs
       even if the Teacher misses a small object (False Negative in Teacher).

    Why NOT Cosine Similarity?
    --------------------------
    Some critics suggest Cosine Similarity over MSE to avoid the "strict alignment
    problem." However, for our use case:
        - Our normalization already handles magnitude differences (ScaleKD insight)
        - MSE with masking outperforms Cosine for small, sparse objects (empirical)
        - Cosine loses information about relative channel importance within a patch

    Args:
        mask_type (str): 'soft' (default) for continuous [0,1] mask, 'hard' for binary
        threshold (float): For 'hard' mask, the activation cutoff (default 0.4)
    """

    def __init__(
        self,
        mask_type="soft",
        threshold=0.4,
        level_weights=None,
        global_rel_weight=0.0,
        global_rel_max_tokens=256,
    ):
        super().__init__()
        self.mask_type = mask_type
        self.threshold = threshold
        # Hierarchical feature weighting prioritizing high-resolution shallow layers:
        # P3: 2.0x, P4: 1.5x, P5: 1.0x
        self.level_weights = level_weights or [2.0, 1.5, 1.0]
        # Optional global-relation KD (FGD-style relation matching) over pooled spatial tokens.
        self.global_rel_weight = max(float(global_rel_weight), 0.0)
        self.global_rel_max_tokens = max(int(global_rel_max_tokens), 1)

    def generate_attention_mask(self, teacher_feats, fg_masks=None):
        """
        Generate spatial attention mask from teacher feature activation magnitudes,
        with optional GT/O2O foreground injection (Blind Teacher Fix) to ensure
        small cones receive full distillation attention even if teacher activations are weak.

        Args:
            teacher_feats: list of [B, C, H, W] teacher feature tensors
            fg_masks: optional list of [B, 1, H, W] foreground masks from O2O/GT assigner

        Returns:
            List of [B, 1, H, W] attention masks
        """
        masks = []
        for idx, feat in enumerate(teacher_feats):
            # Per-sample magnitude attention normalized to [0, 1] (strictly 4D [B, 1, H, W])
            magnitude = torch.mean(torch.abs(feat), dim=1, keepdim=True)  # [B, 1, H, W]
            m_max = magnitude.amax(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            norm_mag = magnitude / (m_max + 1e-5)

            if self.mask_type == "hard":
                mask = (norm_mag > self.threshold).float()
            else:
                mask = norm_mag

            if fg_masks is not None and idx < len(fg_masks) and fg_masks[idx] is not None:
                fg = fg_masks[idx].to(device=mask.device, dtype=mask.dtype)
                if fg.shape[-2:] != mask.shape[-2:]:
                    fg = F.interpolate(fg, size=mask.shape[-2:], mode="nearest")
                mask = torch.maximum(mask, fg.clamp(0.0, 1.0))

            masks.append(mask)

        return masks

    def normalize(self, x):
        """
        Normalize feature maps by global mean absolute value per sample.
        Preserves relative responses between channels/spatial locations while decoupling global scale.
        """
        # x: [B, C, H, W]
        mean = x.abs().mean(dim=(1, 2, 3), keepdim=True)
        # Adding epsilon prevents numerical instability on sparse features
        return x / (mean + 1e-5)

    def _pool_for_global_relation(self, x, target_side=16):
        """Pool features to a consistent spatial grid for relation computation."""
        return F.adaptive_avg_pool2d(x, output_size=(target_side, target_side))

    def _global_relation_loss(self, student_feat, teacher_feat):
        """
        Global relation distillation (channel C x C and spatial N x N relation matching).
        Ensures consistent dimensions by pooling both branches to a unified spatial grid.
        """
        side = max(1, min(16, int(self.global_rel_max_tokens**0.5)))
        s_pooled = self._pool_for_global_relation(student_feat, side)
        t_pooled = self._pool_for_global_relation(teacher_feat, side)

        # 1. Channel relation (C x C)
        s_c = s_pooled.flatten(2)  # [B, C, N]
        t_c = t_pooled.flatten(2).detach()  # [B, C, N]
        s_c_norm = F.normalize(s_c, dim=2, eps=1e-6)
        t_c_norm = F.normalize(t_c, dim=2, eps=1e-6)
        s_rel_c = torch.matmul(s_c_norm, s_c_norm.transpose(1, 2))  # [B, C, C]
        t_rel_c = torch.matmul(t_c_norm, t_c_norm.transpose(1, 2))  # [B, C, C]
        loss_channel_rel = F.mse_loss(s_rel_c, t_rel_c)

        # 2. Spatial relation (N x N)
        s_s = s_c.transpose(1, 2)  # [B, N, C]
        t_s = t_c.transpose(1, 2)  # [B, N, C]
        s_s_norm = F.normalize(s_s, dim=2, eps=1e-6)
        t_s_norm = F.normalize(t_s, dim=2, eps=1e-6)
        s_rel_s = torch.matmul(s_s_norm, s_s_norm.transpose(1, 2))  # [B, N, N]
        t_rel_s = torch.matmul(t_s_norm, t_s_norm.transpose(1, 2))  # [B, N, N]
        loss_spatial_rel = F.mse_loss(s_rel_s, t_rel_s)

        return 0.5 * (loss_channel_rel + loss_spatial_rel)

    def forward(
        self,
        student_feats,
        teacher_feats,
        adapters,
        attention_mask=None,
        fg_masks=None,
        return_breakdown=False,
    ):
        """
        Args:
            student_feats: list of student feature maps [P3, P4, P5]
            teacher_feats: list of teacher feature maps [P3, P4, P5]
            adapters: ModuleList of 1x1 convs to match STUDENT channels to TEACHER
            attention_mask: Optional list of pre-calculated [B, 1, H, W] masks.
            fg_masks: Optional list of [B, 1, H, W] foreground masks for GT/O2O injection.
            return_breakdown: If True, return (total_loss, breakdown_dict) for diagnostics.
        """
        loss = 0.0
        magnitude_anchor_loss = 0.0
        global_relation_loss = 0.0

        total_obj_loss = 0.0

        # Use provided mask if available, otherwise generate internal one
        if attention_mask is not None:
            masks = attention_mask
        else:
            masks = self.generate_attention_mask(teacher_feats, fg_masks=fg_masks)

        for i, (s_feat, t_feat, adapter, mask) in enumerate(
            zip(student_feats, teacher_feats, adapters, masks)
        ):
            level_weight = self.level_weights[i] if i < len(self.level_weights) else 1.0

            # 1. Project student channels to teacher dimension
            s_adapted = adapter(s_feat)

            # 2. Magnitude anchor: enforce consistent scale between student and teacher
            s_mag = s_adapted.abs().mean()
            t_mag = t_feat.abs().mean().detach()
            mag_ratio = (s_mag + 1e-5) / (t_mag + 1e-5)
            log_mag_ratio = torch.log(mag_ratio).clamp(-3.0, 3.0)
            magnitude_anchor_loss += log_mag_ratio**2

            # 3. Decouple scale via normalization
            s_norm = self.normalize(s_adapted)
            t_norm = self.normalize(t_feat)

            # Optional global relation distillation
            if self.global_rel_weight > 0.0:
                global_relation_loss += self._global_relation_loss(s_norm, t_norm)

            # 4. Compute masked MSE (spatial mimicry)
            diff = s_norm - t_norm
            weighted_diff = diff * mask

            mask_expanded = mask.expand_as(weighted_diff)
            loss_obj = weighted_diff.pow(2).sum()
            mask_sum = mask_expanded.sum() + 1e-5

            obj_contribution = level_weight * (loss_obj / mask_sum)
            total_obj_loss += (
                obj_contribution.item()
                if hasattr(obj_contribution, "item")
                else float(obj_contribution)
            )

            loss += level_weight * (loss_obj / mask_sum)

        total_loss = loss + 1.0 * magnitude_anchor_loss
        if self.global_rel_weight > 0.0:
            total_loss = total_loss + self.global_rel_weight * global_relation_loss

        # Normalize by number of feature pyramid levels
        num_levels = len(student_feats)
        total_loss = total_loss / num_levels

        if return_breakdown:
            breakdown = {
                "obj": total_obj_loss / num_levels,
                "magnitude": (
                    magnitude_anchor_loss.item()
                    if hasattr(magnitude_anchor_loss, "item")
                    else float(magnitude_anchor_loss)
                )
                / num_levels,
                "global_rel": (
                    (
                        global_relation_loss.item()
                        if hasattr(global_relation_loss, "item")
                        else float(global_relation_loss)
                    )
                    / num_levels
                ),
            }
            return total_loss, breakdown

        return total_loss


class LogitDistillationLoss(nn.Module):
    """
    Vector 2: Logit-Based Response Distillation (BDKD / O2O Aligned)
    ===============================================================

    Transfers teacher classification confidence to the student using
    temperature-scaled log-sigmoid matching with soft-target BCE regularization.

    Formula:
        L_certainty = || log(σ(s/T)) - log(σ(t/T)) ||²
        L_pixel = L_certainty + 0.1 * BCE(s/T, σ(t/T))

    Args:
        w_fg (float): Weight for foreground pixels (assigned positives). Default 2.0.
        w_bg (float): Weight for background pixels. Default 1.0.
        temperature (float): Temperature scaling factor for softening logits. Default 4.0.
    """

    def __init__(self, w_fg=2.0, w_bg=1.0, temperature=4.0):
        super().__init__()
        self.w_fg = w_fg
        self.w_bg = w_bg
        self.T = temperature

    def forward(self, student_logits, teacher_logits, masks=None, return_breakdown=False):
        """
        Args:
            student_logits: list of [B, NC, H, W] tensors
            teacher_logits: list of [B, NC, H, W] tensors
            masks: list of [B, 1, H, W] tensors (typically O2O fg_mask from Assigner)
                   If None, defaults to uniform weighting (w_bg).
            return_breakdown: If True, return (loss, breakdown_dict) for diagnostics.
        """
        if masks is None:
            masks = [None] * len(student_logits)

        # Compute global foreground/background pixel counts across all levels
        global_n_fg = 0.0
        global_n_bg = 0.0

        for s_log, mask in zip(student_logits, masks):
            if mask is not None:
                global_n_fg += mask.sum().item()
                global_n_bg += mask.numel() - mask.sum().item()
            else:
                global_n_bg += s_log.shape[0] * s_log.shape[2] * s_log.shape[3]

        global_n_fg = max(global_n_fg, 1.0)
        global_n_bg = max(global_n_bg, 1.0)

        fg_weight = self.w_fg / global_n_fg
        bg_weight = self.w_bg / global_n_bg

        loss = 0.0
        total_certainty_loss = 0.0
        total_bce_loss = 0.0

        T = self.T
        for s_log, t_log, mask in zip(student_logits, teacher_logits, masks):
            # 1. Numerically stabilized logits
            s_clamped = s_log.clamp(min=-12.0, max=12.0)
            t_clamped = t_log.clamp(min=-12.0, max=12.0)

            s_scaled = s_clamped / T
            t_scaled = t_clamped / T

            s_prob = torch.sigmoid(s_scaled)
            t_prob = torch.sigmoid(t_scaled)

            # Bounded probability MSE (strictly in [0, 1])
            loss_prob = (s_prob - t_prob).pow(2)

            # Temperature-scaled soft-target BCE regularizer
            loss_bce = F.binary_cross_entropy_with_logits(s_scaled, t_prob, reduction="none") * (T ** 2)

            # 2. Foreground vs background loss decomposition
            if mask is not None:
                m_expanded = mask.expand_as(s_prob)
                is_fg = m_expanded > 0.5

                # Stabilized certainty matching: strictly on foreground anchors where confidence is meaningful
                s_log_prob = F.logsigmoid(s_scaled).clamp(min=-8.0)
                t_log_prob = F.logsigmoid(t_scaled).clamp(min=-8.0)
                loss_certainty = (s_log_prob - t_log_prob).pow(2).clamp(max=16.0)

                # Foreground pixel loss combines probability MSE, certainty matching, and BCE
                pixel_fg = loss_prob + 0.1 * loss_certainty + 0.05 * loss_bce

                # Background pixel loss: focal probability MSE + BCE (no certainty matching to prevent background gradient distortion)
                focal_bg = (s_prob - t_prob).abs().pow(1.5).clamp(min=1e-3, max=1.0)
                pixel_bg = (loss_prob + 0.05 * loss_bce) * focal_bg

                pixel_loss = torch.where(is_fg, pixel_fg, pixel_bg)

                weights = m_expanded * fg_weight + (1.0 - m_expanded) * bg_weight
                weighted_loss = pixel_loss * weights
                nc = pixel_loss.shape[1]
                loss += weighted_loss.sum() / nc

                weighted_cert = (loss_certainty * m_expanded * fg_weight).sum() / nc
                weighted_bce = (loss_bce * weights).sum() / nc * 0.05
                total_certainty_loss += (
                    weighted_cert.item() if hasattr(weighted_cert, "item") else float(weighted_cert)
                )
                total_bce_loss += (
                    weighted_bce.item() if hasattr(weighted_bce, "item") else float(weighted_bce)
                )
            else:
                pixel_loss = loss_prob + 0.05 * loss_bce
                loss += pixel_loss.mean() * self.w_bg

        if return_breakdown:
            breakdown = {
                "certainty": total_certainty_loss,
                "bce": total_bce_loss,
                "fg_count": global_n_fg,
                "bg_count": global_n_bg,
            }
            return loss, breakdown

        return loss


class OneToOneBoxLoss(nn.Module):
    """
    Vector 3: One-to-One Box Regression Distillation - Wise-IoU v3
    ===============================================================

    Transfers teacher localization precision to student using Wise-IoU v3
    directly on predicted LTRB box coordinate offsets.

    Input boxes are [B, 4, H, W] where channels are (l, t, r, b) distances
    from anchor grid centers:
        - Box width = l + r, height = t + b
    """

    def __init__(self, use_wiou=True):
        super().__init__()
        self.use_wiou = use_wiou
        self.register_buffer("wiou_mean", torch.tensor(0.7))
        self.wiou_momentum = 0.1  # Momentum for running mean update

    def ltrb_to_xyxy(self, ltrb):
        """
        Convert LTRB (distances from anchor) to XYXY format for IoU calculation.

        Args:
            ltrb: [N, 4] (l, t, r, b) distances from anchor

        Returns:
            xyxy: [N, 4] (x1, y1, x2, y2) anchor-relative coordinates
        """
        left, top, right, bottom = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
        # Anchor is at origin. Box extends [-left, -top] to [right, bottom] from anchor.
        x1 = -left
        y1 = -top
        x2 = right
        y2 = bottom
        return torch.stack([x1, y1, x2, y2], dim=1)

    def bbox_iou_ltrb(self, box1, box2, eps=1e-5):
        """
        Calculate IoU between two sets of LTRB boxes.
        box1, box2: [N, 4] (l, t, r, b) signed distances from grid center.
        Supports negative individual offsets for sub-stride objects while clamping total width/height >= 0.
        """
        # Anchor is at origin (0, 0): x1 = -l, y1 = -t, x2 = r, y2 = b
        # Intersection width = min(x2_1, x2_2) - max(x1_1, x1_2) = min(r1, r2) + min(l1, l2)
        inter_w = (torch.min(box1[:, 0], box2[:, 0]) + torch.min(box1[:, 2], box2[:, 2])).clamp(min=0.0)
        inter_h = (torch.min(box1[:, 1], box2[:, 1]) + torch.min(box1[:, 3], box2[:, 3])).clamp(min=0.0)
        inter_area = inter_w * inter_h

        w1 = (box1[:, 0] + box1[:, 2]).clamp(min=0.0)
        h1 = (box1[:, 1] + box1[:, 3]).clamp(min=0.0)
        w2 = (box2[:, 0] + box2[:, 2]).clamp(min=0.0)
        h2 = (box2[:, 1] + box2[:, 3]).clamp(min=0.0)

        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - inter_area + eps

        iou = inter_area / union
        return iou.clamp(min=0.0, max=1.0)

    def wise_iou_ltrb(self, pred, target, alpha=1.9, delta=3):
        """
        Wise-IoU v3 for LTRB format boxes.

        Adapted from box_ops.wise_iou but specifically for LTRB (distance from anchor) format.
        Supports signed individual LTRB offsets for sub-stride cones.

        Args:
            pred: [N, 4] (l, t, r, b) student predictions
            target: [N, 4] (l, t, r, b) teacher predictions

        Returns:
            iou: Standard IoU [N]
            l_wiou: Wise-IoU v3 loss [N]
        """
        # Standard IoU for LTRB (supports signed individual offsets with non-negative total w, h)
        iou = self.bbox_iou_ltrb(pred, target)

        # Distance Penalty (R_WIoU)
        # Anchor-relative centers: cx = (x1 + x2)/2 = (r - l)/2, cy = (y1 + y2)/2 = (b - t)/2
        pred_cx = (pred[:, 2] - pred[:, 0]) * 0.5
        pred_cy = (pred[:, 3] - pred[:, 1]) * 0.5
        target_cx = (target[:, 2] - target[:, 0]) * 0.5
        target_cy = (target[:, 3] - target[:, 1]) * 0.5

        # Smallest enclosing box dimensions in anchor-relative coordinates:
        # enc_w = max(r_p, r_t) - min(-l_p, -l_t) = max(r_p, r_t) + max(l_p, l_t)
        box_w = (torch.max(pred[:, 0], target[:, 0]) + torch.max(pred[:, 2], target[:, 2])).clamp(min=1e-7)
        box_h = (torch.max(pred[:, 1], target[:, 1]) + torch.max(pred[:, 3], target[:, 3])).clamp(min=1e-7)

        # Distance and diagonal (denominator detached per WIoU paper to prevent gradient leak)
        dist2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        diag2 = (box_w**2 + box_h**2 + 1e-7).detach()

        # Distance attention (clamped to prevent overflow)
        dist_ratio = (dist2 / diag2).clamp(max=10.0)
        dist_attn = torch.exp(dist_ratio)

        # WIoU v1 Loss
        l_iou = 1.0 - iou
        l_wiou_v1 = dist_attn * l_iou

        # WIoU v3: Dynamic Non-Monotonic Focusing with running mean
        l_iou_detach = l_iou.detach()
        l_avg = max(1.0 - self.wiou_mean, 0.01)
        beta = (l_iou_detach / (l_avg + 1e-7)).clamp(min=0.1, max=10.0)

        # Focusing Factor r
        exponent = (beta - delta).clamp(min=-10.0, max=10.0)
        r = (beta / (delta * torch.pow(alpha, exponent) + 1e-7)).clamp(min=0.01, max=10.0)

        # Final Loss
        l_wiou = r * l_wiou_v1

        return iou, l_wiou

    def forward(
        self,
        student_boxes,
        teacher_boxes,
        masks=None,
        confidence_weights=None,
        return_breakdown=False,
    ):
        """
        Args:
            student_boxes: list of [B, 4, H, W] tensors
            teacher_boxes: list of [B, 4, H, W] tensors
            masks: list of [B, 1, H, W] foreground masks
            confidence_weights: list of [B, 1, H, W] teacher confidence maps
            return_breakdown: If True, return (loss, breakdown_dict) for diagnostics.
        """
        # Sum all losses across pyramid levels before global count normalization
        total_loss_sum = torch.zeros((), device=student_boxes[0].device, dtype=student_boxes[0].dtype)
        total_fg_count = 0.0
        total_iou_sum = 0.0
        total_conf_sum = 0.0

        if masks is None:
            masks = [torch.ones_like(b[:, 0:1, ...]) for b in teacher_boxes]

        if confidence_weights is None:
            confidence_weights = [None] * len(masks)

        for s_box, t_box, mask, conf in zip(
            student_boxes, teacher_boxes, masks, confidence_weights
        ):
            B, C, H, W = s_box.shape

            s_flat = s_box.permute(0, 2, 3, 1).reshape(-1, 4)
            t_flat = t_box.permute(0, 2, 3, 1).reshape(-1, 4)
            m_flat = mask.permute(0, 2, 3, 1).reshape(-1, 1).squeeze(-1)

            if conf is not None:
                c_flat = conf.permute(0, 2, 3, 1).reshape(-1, 1).squeeze(-1)
            else:
                c_flat = None

            fg_indices = m_flat > 0.5
            n_fg = fg_indices.sum().item()

            if n_fg > 0:
                s_fg = s_flat[fg_indices]
                t_fg = t_flat[fg_indices]

                if self.use_wiou:
                    iou, pixel_loss = self.wise_iou_ltrb(s_fg, t_fg)

                    batch_iou_sum = iou.sum().item()
                    batch_iou_count = n_fg
                    if batch_iou_count > 0:
                        batch_mean_iou = batch_iou_sum / batch_iou_count
                        self.wiou_mean.fill_(
                            (1.0 - self.wiou_momentum) * self.wiou_mean.item()
                            + self.wiou_momentum * batch_mean_iou
                        )
                else:
                    iou = self.bbox_iou_ltrb(s_fg, t_fg)
                    pixel_loss = 1.0 - iou

                total_iou_sum += iou.sum().item() if hasattr(iou, "item") else float(iou.sum())

                # Weight distillation loss by teacher confidence to attenuate noisy empty predictions
                current_weights = torch.ones_like(pixel_loss)
                if c_flat is not None:
                    c_fg = c_flat[fg_indices]
                    current_weights = c_fg
                    total_conf_sum += c_fg.sum().item()
                else:
                    total_conf_sum += n_fg

                total_loss_sum = total_loss_sum + (pixel_loss * current_weights).sum()
                total_fg_count += n_fg

        if total_fg_count > 0:
            final_loss = total_loss_sum / total_fg_count
        else:
            final_loss = total_loss_sum + 0.0 * sum(b.sum() for b in student_boxes)

        if return_breakdown:
            breakdown = {
                "iou_mean": total_iou_sum / max(total_fg_count, 1),
                "wiou_loss": (total_loss_sum / max(total_fg_count, 1)).item()
                if hasattr(total_loss_sum, "item")
                else float(total_loss_sum / max(total_fg_count, 1)),
                "conf_weight_mean": total_conf_sum / max(total_fg_count, 1),
                "fg_count": total_fg_count,
            }
            return final_loss, breakdown

        return final_loss


class AdaptiveLossBalancer(nn.Module):
    """
    Automatic multi-task loss balancing using normalized homoscedastic uncertainty.
    Normalizes each loss component to a shared reference scale before uncertainty weighting
    to eliminate numerical magnitude disparities (e.g. Box KD ~0.01 vs Logit KD ~0.40)
    that would otherwise overweight Box KD by ~40x and suppress Logit KD.

    Formula:
        ref_scale = mean(running_scales)
        L_norm,i = (L_i / running_scale_i) * ref_scale
        L_total = Σ [ exp(-s_i) * L_norm,i + ref_scale * s_i ]
        where s_i is clamped to [-1.5, 1.5].
    """

    def __init__(self, num_losses=3, momentum=0.05):
        super().__init__()
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.register_buffer("running_scales", torch.ones(num_losses))
        self.momentum = momentum

    def forward(self, losses):
        """
        Args:
            losses: list of unweighted loss tensors [L_sdf, L_logit, L_box]
        Returns:
            weighted_loss: Scalar total loss
        """
        # Ensure losses are valid tensors on the balancer's device
        safe_losses = []
        for loss_item in losses:
            if torch.is_tensor(loss_item):
                safe_losses.append(loss_item)
            else:
                safe_losses.append(torch.tensor(float(loss_item), device=self.running_scales.device))

        if self.training:
            with torch.no_grad():
                for i, loss in enumerate(safe_losses):
                    curr_val = loss.detach().abs().clamp(min=1e-4)
                    self.running_scales[i] = (
                        1 - self.momentum
                    ) * self.running_scales[i] + self.momentum * curr_val

        # Reference scale: mean magnitude across active components
        ref_scale = self.running_scales.mean().clamp(min=1e-4)

        total_loss = torch.zeros((), device=safe_losses[0].device, dtype=safe_losses[0].dtype)
        for i, loss in enumerate(safe_losses):
            scale = self.running_scales[i].clamp(min=1e-4)
            # Normalize to common reference scale so all vectors compete on equal footing
            norm_loss = (loss / scale) * ref_scale

            # Bounded log-variance in [-1.5, 1.5] prevents suppression or runaway weights
            s = self.log_vars[i].clamp(min=-1.5, max=1.5)
            precision = torch.exp(-s)
            total_loss = total_loss + precision * norm_loss + ref_scale * s

        return total_loss

    def get_weights(self):
        """
        Returns the effective weights on each raw loss component L_i.
        W_i = exp(-s_i) * (ref_scale / running_scale_i)
        """
        ref_scale = self.running_scales.mean().clamp(min=1e-4)
        precision = torch.exp(-self.log_vars.clamp(min=-1.5, max=1.5)).detach()
        return precision * (ref_scale / self.running_scales.clamp(min=1e-4))

    def get_distillation_weights(self):
        """
        Returns live (gradient-enabled) distillation weights for Teacher Fade penalty.

        Unlike get_weights(), this returns LIVE tensors (not detached) so the optimizer
        can backpropagate through the fade penalty to reduce distillation weights.

        Returns:
            Tensor of shape [3]: weights for [SDF, Logit, Box] distillation losses.
            Task loss (index 3) is intentionally excluded from Teacher Fade.
        """
        # Indices 0, 1, 2 are SDF, Logit, Box distillation losses
        # Apply same clamp as forward() for consistency
        clamped = self.log_vars[:3].clamp(min=-1.5, max=1.5)
        return torch.exp(-clamped)

    def soft_reset(self, target_max_weight: float = 5.0):
        """
        Safety Shift for phase transitions (Sparsity/QAT).
        Preserves relative importance ratios while reducing absolute gradient magnitude.
        """
        with torch.no_grad():
            self.log_vars.data.clamp_(min=-1.6, max=1.6)
            min_log_var = torch.min(self.log_vars.data)
            max_weight = torch.exp(-min_log_var)

            if max_weight > target_max_weight:
                shift = (
                    -torch.log(torch.tensor(target_max_weight, device=self.log_vars.device))
                    - min_log_var
                )
                self.log_vars.data += shift

                downscale_factor = (max_weight / target_max_weight).item()
                print(
                    f"  > Downscaled weights by {downscale_factor:.1f}x (max: {max_weight.item():.1f} → {target_max_weight})"
                )
            else:
                print(
                    f"  > No shift needed (max weight {max_weight.item():.2f} <= {target_max_weight})"
                )
