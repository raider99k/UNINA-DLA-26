import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors

from unina_dla.utils.box_ops import wise_iou, xywh2xyxy


class ProgLoss(nn.Module):
    """
    Progressive Loss (ProgLoss): Difficulty-Based Sample Weighting.

    This is a UNINA-DLA variant that re-weights classification loss based on
    prediction difficulty (hard example mining). Samples with higher |p - t|
    receive up to 2x weight, focusing training on challenging predictions.

    Formula: L = BCE(p, t) * (1 + |p - t|)

    Note: This differs from official YOLO26 ProgLoss which uses epoch-based
    scheduling of loss component weights. Our approach provides complementary
    benefits by focusing on hard examples within each batch.
    """

    def __init__(self, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        loss = self.bce(pred, target)

        # Calculate Difficulty (L1 distance)
        with torch.no_grad():
            if self.use_sigmoid:
                prob = pred.sigmoid()
            else:
                prob = pred
            diff = torch.abs(prob - target)

        # Progressive weighting: up to 2x for hardest examples
        # Clamped to [1.0, 2.0] to prevent loss spikes on uncertain predictions
        prog_weight = torch.clamp(1.0 + diff, min=1.0, max=2.0)

        return loss * prog_weight


class YOLO26DetectionLoss(nn.Module, v8DetectionLoss):
    """
    YOLO26 Detection Loss (UNINA-DLA Variant).

    Key Features:
    1. No DFL (Direct Regression) - Matches official YOLO26
    2. ProgLoss (Difficulty Weighting) - Hard example mining per batch
    3. STAL (Post-hoc Small Object Weighting) - 2x loss for area < 1024 px²
    4. O2O-Curriculum Assignment (top-k annealed to strict O2O)
    5. Optional aux output (fg_mask) for structurally-aligned distillation

    Implementation Notes:
    - ProgLoss: Uses difficulty-based weighting (|p-t|) rather than epoch scheduling
    - STAL: Applied as post-hoc loss weighting rather than label assignment modification

    These design choices optimize for DLA distillation stability while achieving
    similar objectives to official YOLO26 (small object focus, convergence speed).

    Args:
        model: The model containing head configuration
        tal_topk: Initial Top-K for TaskAlignedAssigner (annealed externally to 1)
        return_aux: If True, __call__ returns (loss, loss_items, aux_dict) with fg_mask
    """

    def __init__(
        self,
        model,
        tal_topk=1,
        tal_topk2=None,
        return_aux=False,
        dup_loss_weight=0.0,
        dup_iou_thr=0.6,
        dup_margin=0.2,
        dup_topk=20,
        hard_bg_loss_weight=0.0,
        hard_bg_iou_thr=0.2,
        hard_bg_topk=64,
        dup_start_pct=0.6,
    ):
        nn.Module.__init__(self)
        v8DetectionLoss.__init__(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)

        self.return_aux = return_aux

        # 1. One-to-One Assignment (Strict) with Minimum-Distance Fallback
        # Standard assigner may fail for objects smaller than stride (no anchor center inside GT).
        # We wrap it with fallback logic to force-assign the nearest anchor.
        self.assigner_topk = max(1, int(tal_topk))
        self.assigner_topk2 = None if tal_topk2 is None else max(1, int(tal_topk2))
        self.assigner = self._build_assigner(self.assigner_topk, self.assigner_topk2)

        # 2. Classification Loss (Stable BCEWithLogitsLoss without heuristic difficulty spikes)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Hyperparameters
        self.hyp.box = 7.5
        self.hyp.cls = 0.5
        self.hyp.dfl = 0.0  # Disabled: Wise-IoU v3 provides dynamic non-monotonic focusing without L1 conflict

        # Direct coordinate regression (4 coords, no DFL)
        self.reg_max = 0
        self.no = self.nc + 4

        # Wise-IoU v3 running state
        self.register_buffer("wiou_mean", torch.tensor(0.5))
        self.momentum = 0.03

        # Explicit training mode flag (fallback if not managed by nn.Module hierarchy)
        # Ensures self.training is always available even when loss used standalone
        if not hasattr(self, "training"):
            self.training = True

        # Optional duplicate/FP suppression for stricter NMS-free behavior.
        # All terms are O2O-head only by wiring in adapters (O2M keeps recall-focused supervision).
        self.dup_loss_weight = max(0.0, float(dup_loss_weight))
        self.dup_iou_thr = min(max(float(dup_iou_thr), 0.0), 1.0)
        self.dup_margin = float(dup_margin)
        self.dup_topk = max(1, int(dup_topk))
        self.hard_bg_loss_weight = max(0.0, float(hard_bg_loss_weight))
        self.hard_bg_iou_thr = min(max(float(hard_bg_iou_thr), 0.0), 1.0)
        self.hard_bg_topk = max(1, int(hard_bg_topk))
        self.dup_start_pct = min(max(float(dup_start_pct), 0.0), 1.0)
        self.dup_progress = 1.0
        self.last_dup_loss = 0.0
        self.last_hard_bg_loss = 0.0

    def _build_assigner(self, topk: int, topk2: int | None = None):
        kwargs = dict(
            topk=max(1, int(topk)),
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
        )
        if hasattr(self, "stride") and self.stride is not None:
            try:
                kwargs["stride"] = self.stride.tolist()
            except Exception:
                pass
        if topk2 is not None:
            kwargs["topk2"] = max(1, int(topk2))
        try:
            return TaskAlignedAssigner(**kwargs)
        except TypeError:
            # Backward compatibility with older TaskAlignedAssigner signatures.
            kwargs.pop("topk2", None)
            kwargs.pop("stride", None)
            return TaskAlignedAssigner(**kwargs)

    def set_assigner_topk(self, topk: int, topk2: int | None = None):
        """
        Update TaskAlignedAssigner top-k dynamically (assignment curriculum).
        """
        topk = max(1, int(topk))
        self.assigner_topk = topk
        if topk2 is not None:
            self.assigner_topk2 = max(1, int(topk2))
        # Keep existing assigner object when possible to avoid unnecessary churn.
        if hasattr(self.assigner, "topk"):
            self.assigner.topk = topk
            if hasattr(self.assigner, "topk2") and self.assigner_topk2 is not None:
                self.assigner.topk2 = self.assigner_topk2
        else:
            self.assigner = self._build_assigner(topk, self.assigner_topk2)

    def set_dup_suppression_progress(self, progress: float):
        """Update phase progress [0, 1] for staged duplicate suppression."""
        self.dup_progress = min(max(float(progress), 0.0), 1.0)

    def _dup_schedule_scale(self):
        """Linear ramp for suppression terms to avoid early-epoch recall shock."""
        p = min(max(float(self.dup_progress), 0.0), 1.0)
        s = float(self.dup_start_pct)
        if p <= s:
            return 0.0
        if s >= 1.0:
            return 1.0
        return min(1.0, (p - s) / max(1e-6, (1.0 - s)))

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scale_tensor)
            # Convert cxcywh to xyxy for assigner and loss calculations
            out[..., 1:5] = xywh2xyxy(out[..., 1:5])
        return out

    def bbox_decode(self, anchor_points, pred_dist, eps=1e-2):
        """
        Decode predicted box coordinates (anchor-free).
        pred_dist: [..., 4] (reg)
        Returns: xyxy
        """
        # Add additive epsilon to ensure non-negative/non-zero values while preserving gradients
        pred_dist = pred_dist + eps
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl(dummy)
        feats = preds[1] if isinstance(preds, tuple) else preds

        # Join reg and cls if tuple (Standard convention in UNINA code)
        # preds is (reg_outputs, cls_outputs)
        if isinstance(preds, tuple):
            # reg: [B, 4, H, W], cls: [B, NC, H, W]
            feats = [torch.cat([r, c], 1) for r, c in zip(preds[0], preds[1])]

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode Boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy (Grid Units)
        pred_bboxes_pixels = pred_bboxes * stride_tensor  # xyxy (Pixel Units)

        # Assign (One-to-One) with Minimum-Distance Fallback
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Minimum-distance fallback: ensure tiny objects (< stride) receive positive anchor assignment
        anchor_points_px = anchor_points * stride_tensor  # [N_anchors, 2]

        for b in range(batch_size):
            # Check each GT in this batch
            gt_mask_b = mask_gt[b, :, 0]  # [max_gt]
            n_gt = gt_mask_b.sum().int().item()

            if n_gt == 0:
                continue

            for g in range(n_gt):
                gt_box = gt_bboxes[b, g]  # [4] xyxy
                gt_label = gt_labels[b, g]  # [1]

                # Check if this GT has any assigned anchors.
                # Use assigner-provided target_gt_idx instead of approximate box matching:
                # nearby cones can share similar x1 and create false positives.
                gt_center = torch.stack([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
                assigned_mask = fg_mask[b] & (target_gt_idx[b] == g)

                if assigned_mask.sum() == 0:
                    # No assignment found! Force-assign nearest anchor.
                    # Prefer currently unassigned anchors to preserve O2O uniqueness.
                    anchor_centers = anchor_points_px  # [N, 2]
                    dists = (anchor_centers[:, 0] - gt_center[0]) ** 2 + (
                        anchor_centers[:, 1] - gt_center[1]
                    ) ** 2

                    available = ~fg_mask[b]
                    if available.any():
                        masked_dists = dists.masked_fill(~available, float("inf"))
                        nearest_idx = masked_dists.argmin()
                    else:
                        # Extremely rare: all anchors already assigned.
                        # Reassign nearest anchor and clear previous class target to avoid mixed labels.
                        nearest_idx = dists.argmin()
                        target_scores[b, nearest_idx].zero_()

                    # Force-assign this anchor
                    fg_mask[b, nearest_idx] = True
                    target_gt_idx[b, nearest_idx] = g
                    target_bboxes[b, nearest_idx] = gt_box
                    # Ensure one-hot class target at this anchor (strict O2O semantics).
                    target_scores[b, nearest_idx].zero_()
                    target_scores[b, nearest_idx, gt_label.long().item()] = 1.0

        target_scores_sum = target_scores.sum().clamp(min=1.0)

        # Scale-targeted hybrid loss: smooth sigmoid transition from NWD (small objects) to WIoU (large objects)
        target_ws = target_bboxes[..., 2] - target_bboxes[..., 0]
        target_hs = target_bboxes[..., 3] - target_bboxes[..., 1]
        target_areas = target_ws * target_hs

        # Transition centered at 32x32 pixels (1024 area), guarded against negative area underflow
        alpha_nwd = torch.sigmoid(10 - (target_areas.clamp(min=0).sqrt() / 3.2))

        # Adaptive weight boost for small object classification (Smooth STAL)
        # 2.0 for tiny objects, 1.0 for large objects
        stal_smooth_weights = 1.0 + alpha_nwd

        # Box Loss (IoU) - with NaN prevention guards
        num_fg = fg_mask.sum()

        if num_fg > 0:
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # Clamp boxes to prevent degenerate (zero/negative area) boxes
            # Add small epsilon to ensure minimum box size
            pred_boxes_fg = pred_bboxes_pixels[fg_mask].clamp(min=0)
            target_boxes_fg = target_bboxes[fg_mask].clamp(min=0)

            # Wise-IoU v3 (Dynamic Non-Monotonic Focusing)
            # Import moved to top of file for performance

            # 1. Update Running Mean (Training Only)
            if self.training:
                # Calculate standard IoU for tracking (detached to avoid graph issues)
                with torch.no_grad():
                    iou_static = wise_iou(pred_boxes_fg, target_boxes_fg)[0]  # Just get IoU
                    self.wiou_mean = (
                        1 - self.momentum
                    ) * self.wiou_mean + self.momentum * iou_static.mean().item()

            # 2. Robust Wise-IoU v3 Loss (Dynamic Non-Monotonic Focusing without NWD gradient fighting)
            iou, loss_box_wiou = wise_iou(pred_boxes_fg, target_boxes_fg, mean_iou=self.wiou_mean)

            # Clamp IoU to [0, 1] for metrics
            iou = iou.clamp(min=0.0, max=1.0)

            # Apply Smooth STAL weight and accumulate
            stal_fg = stal_smooth_weights[fg_mask]
            loss[0] = (loss_box_wiou * weight.squeeze() * stal_fg).sum() / target_scores_sum
        else:
            # No foreground samples in this batch - skip box loss
            loss[0] = torch.tensor(0.0, device=self.device, dtype=loss.dtype)

        # Cls Loss (Stable BCEWithLogits)
        loss_cls = self.bce(pred_scores, target_scores.to(dtype))  # [B, N, C]

        # Apply Smooth STAL to Cls Loss (FG only).
        # Background anchors inherit a target box index from assigner bookkeeping,
        # so applying STAL to all anchors would unintentionally upweight BG BCE.
        cls_stal_weights = torch.ones_like(stal_smooth_weights)
        cls_stal_weights[fg_mask] = stal_smooth_weights[fg_mask]
        # Explicit broadcasting: [B, N, 1] * [B, N, C]
        loss_cls = loss_cls * cls_stal_weights.unsqueeze(-1)

        loss[1] = loss_cls.sum() / target_scores_sum

        # Suppression state maintained for logging compatibility without unstable heuristic loops
        self.last_dup_loss = 0.0
        self.last_hard_bg_loss = 0.0
        suppression_loss = torch.zeros((), device=self.device, dtype=loss.dtype)

        # 3. Direct Coordinate Regression: Normalized L1 Loss
        # Disabled when dfl == 0.0 to allow Wise-IoU v3 scale-invariant optimization without rigid gradient conflict
        dfl_gain = getattr(self.hyp, "dfl", 0.0)
        if num_fg > 0 and dfl_gain > 0.0:
            target_ltrb = bbox2dist(anchor_points, target_bboxes / stride_tensor)
            # Normalize ltrb by image size for scale invariance
            target_ltrb_norm = target_ltrb * stride_tensor
            target_ltrb_norm[..., 0::2] /= imgsz[1]
            target_ltrb_norm[..., 1::2] /= imgsz[0]
            pred_ltrb_norm = pred_distri * stride_tensor
            pred_ltrb_norm[..., 0::2] /= imgsz[1]
            pred_ltrb_norm[..., 1::2] /= imgsz[0]
            loss_l1 = (
                F.l1_loss(pred_ltrb_norm[fg_mask], target_ltrb_norm[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss[2] = loss_l1.sum() / target_scores_sum
        else:
            loss[2] = torch.tensor(0.0, device=self.device, dtype=loss.dtype)

        # Gains
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= dfl_gain

        total_loss = (loss.sum() + suppression_loss) * batch_size

        # Prepare aux output for distillation (O2O alignment)
        if self.return_aux:
            # Compute per-level spatial dimensions for splitting fg_mask
            feat_shapes = [(f.shape[2], f.shape[3]) for f in feats]  # [(H0,W0), (H1,W1), (H2,W2)]
            aux = {
                "fg_mask": fg_mask,  # [B, N] where N = sum(Hi*Wi)
                "feat_shapes": feat_shapes,  # For splitting back to per-level
                "anchor_points": anchor_points,
                "stride_tensor": stride_tensor,
            }
            return total_loss, loss.detach(), aux

        return total_loss, loss.detach()
