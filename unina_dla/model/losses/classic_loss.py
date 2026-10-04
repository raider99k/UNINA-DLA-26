import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from unina_dla.utils.box_ops import xywh2xyxy


class ClassicDetectionLoss(nn.Module, v8DetectionLoss):
    """
    Classic Detection Loss (Standard YOLOv8-like).

    Used for "Classic Mode" baseline training.

    Key Characteristics:
    - One-to-Many Assignment (topk=10)
    - Standard BCE for Classification
    - Standard CIoU/DFL for Box Regression (No WIoU, No NWD)
    - No Special "S" or "M" sizing logic
    """

    def __init__(self, model, tal_topk=10):
        nn.Module.__init__(self)
        v8DetectionLoss.__init__(self, model, tal_topk=tal_topk)

        # Enforce Standard Assignment (One-to-Many)
        # alpha=0.5, beta=6.0 are standard TAL params
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

        # Standard BCE (No ProgLoss)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Override self.no if reg_max=0 (Direct Regression) to avoid shape mismatch
        # v8DetectionLoss calculates self.no = nc + reg_max * 4
        # If reg_max=0, it might default to something else or be 0.
        # We need to ensure it matches the model output.
        if self.reg_max == 0:
            self.no = self.nc + 4

        # Standard Hyperparams (Reset to v8 defaults if they were changed)
        self.hyp.box = 7.5
        self.hyp.cls = 0.5
        self.hyp.dfl = 1.5

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets (same as v26_loss/v8DetectionLoss)."""
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
            # Convert cxcywh (standard YOLO format) to xyxy (assigner format)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5])
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted box coordinates."""
        # Additive epsilon to prevent gradient annihilation at zero
        pred_dist = pred_dist + 1e-5
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )

        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Forward pass for loss calculation (Overrides __call__ from v8DetectionLoss)."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        # Handle Tuple input (Reg, Cls) -> Concat for v8DetectionLoss structure
        if isinstance(preds, tuple):
            # reg: [B, 4, H, W], cls: [B, NC, H, W]
            # But v8DetectionLoss usually expects a list of [B, 4+NC, H, W] if not split
            # Actually v8DetectionLoss forward expects `preds` to be the list of detection heads outputs.
            # In UNINA_DLA, preds is ([reg_list], [cls_list]).
            # We need to reshape/stride/concat to match what v8DetectionLoss expects?
            # Or just manually extract like v26_loss does.

            # Let's copy the extraction logic from v26_loss to be safe and compatible with our model.
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

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy (Grid Units)

        # Assignment (One-to-Many via topk=10)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Loss Calculation
        # Cls Loss (Standard BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Box Loss (CIoU)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()
