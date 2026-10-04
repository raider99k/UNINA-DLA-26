"""
UNINA-DLA Custom Validator.

WHY THIS EXISTS (DO NOT REFACTOR TO OFFICIAL WITHOUT CAREFUL CONSIDERATION):
-------------------------------------------------------------------------------
1. CUSTOM MODEL OUTPUT FORMAT: The UNINA_DLA model outputs a 2-tuple (reg_outputs, cls_outputs)
   during training and a 6-tuple (reg3, reg4, reg5, cls3, cls4, cls5) during deployment.
   Ultralytics' DetectionValidator expects their internal YOLO wrapper format.

2. DIRECT REGRESSION DECODING: YOLO26 uses Direct Regression (4 coords: l,t,r,b) instead of
   DFL (4*16=64 bins). This validator handles both cases via the `reg_max` detection,
   but the decoding logic is tightly coupled to our `box_ops.decode_bboxes`.

3. NMS-FREE SUPPORT: YOLO26 is NMS-free (One-to-One assignment). This validator auto-detects
   `reg_max=0` and bypasses NMS entirely. Official validator may not handle this correctly
   for our custom head structure.

4. ANCHOR GENERATION: We generate anchors on-the-fly from feature map shapes. This is
   tightly coupled to our specific stride configuration [8, 16, 32].

SWITCHING RISK: High. Would require:
- Wrapping UNINA_DLA in an Ultralytics-compatible interface
- Reformatting outputs on every inference call
- Potentially breaking QAT and distillation pipelines

RECOMMENDATION: Keep this custom validator. The matching logic in UNINAMetrics now
uses official Ultralytics matching algorithm (iterative thresholding), so mAP
calculations are accurate and comparable to official benchmarks.
"""

import time

import torch
from tqdm import tqdm

from unina_dla.utils.box_ops import decode_bboxes, non_max_suppression
from unina_dla.utils.metrics import UNINAMetrics
from unina_dla.utils.visualization import save_val_predictions


class Profile:
    """
    Ultralytics-style profiler that accumulates time across multiple context manager calls.

    Uses CUDA synchronization for accurate GPU timing when device is CUDA.
    Time is accumulated in self.t (seconds) across all __enter__/__exit__ pairs.
    """

    def __init__(self, device=None):
        self.device = device
        self.t = 0.0  # Total accumulated time in seconds
        self.start = 0.0

    def __enter__(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.t += time.perf_counter() - self.start


def validate(
    model,
    dataloader,
    device,
    conf_thres=0.001,
    iou_thres=0.6,
    strides=None,
    num_classes=1,
    names=None,
    save_dir=None,
    nms=None,
    prefix="",
    max_det=300,
):
    model.eval()
    max_det = max(1, int(max_det))

    # Dynamic parameter logic
    reg_max = 0  # Default (Direct Regression)
    if hasattr(model, "head"):
        if hasattr(model.head, "reg_max"):
            reg_max = model.head.reg_max
        if strides is None and hasattr(
            model.head, "stride"
        ):  # use model stride if available and not overridden
            strides = model.head.stride
            if isinstance(strides, torch.Tensor):
                strides = strides.cpu().numpy().tolist()

    # Fallback/Safety
    if strides is None:
        strides = [8, 16, 32]
        print("WARNING: Strides not found in model, using default [8, 16, 32]")

    # Auto-detect NMS requirement: YOLO26 (reg_max=0) is NMS-free
    if nms is None:
        nms = False if reg_max == 0 else True
        if not nms:
            print("INFO: NMS-free validation enabled (YOLO26/O2O detected).")

    metrics = UNINAMetrics(num_classes=num_classes, names=names, save_dir=save_dir, prefix=prefix)

    # Validation Loop
    # Save validation sample predictions for visual inspection
    if save_dir:
        print(f"Saving debug images to {save_dir}...")
        save_val_predictions(
            model,
            dataloader,
            device,
            names,
            save_dir,
            num_samples=3,
            conf_thres=0.25,
            prefix=prefix,
        )

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    # Anchor Cache
    cached_anchors = None
    cached_strides = None
    last_shape = None

    # Ultralytics-style timing: (preprocess, inference, postprocess)
    dt = (Profile(device=device), Profile(device=device), Profile(device=device))
    total_images = 0
    total_preds_above_thresh = 0
    input_shape = None  # Track for speed report

    for batch in pbar:
        bs = batch["img"].shape[0]

        # Preprocess timing
        with dt[0]:
            imgs = batch["img"].to(device).float() / 255.0

        current_shape = imgs.shape[2:]  # (H, W)
        if input_shape is None:
            input_shape = imgs.shape  # (B, C, H, W)

        # Process Targets
        targets = []
        if "bboxes" in batch:
            b_idx = batch["batch_idx"].to(device)
            b_cls = batch["cls"].to(device).view(-1, 1)
            b_box = batch["bboxes"].to(device)

            # Convert xywh normalized to xyxy absolute
            h, w = imgs.shape[2:]
            b_box_abs = b_box.clone()
            b_box_abs[:, 0] *= w
            b_box_abs[:, 1] *= h
            b_box_abs[:, 2] *= w
            b_box_abs[:, 3] *= h

            x1 = b_box_abs[:, 0] - b_box_abs[:, 2] / 2
            y1 = b_box_abs[:, 1] - b_box_abs[:, 3] / 2
            x2 = b_box_abs[:, 0] + b_box_abs[:, 2] / 2
            y2 = b_box_abs[:, 1] + b_box_abs[:, 3] / 2

            b_box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            gt_data = torch.cat([b_cls, b_box_xyxy], dim=1)

            for i in range(bs):
                mask = b_idx == i
                if mask.sum() > 0:
                    targets.append(gt_data[mask])
                else:
                    targets.append(torch.zeros((0, 5), device=device))
        else:
            targets = [torch.zeros((0, 5), device=device) for _ in range(bs)]

        # Inference timing
        with dt[1], torch.no_grad():
            preds = model(imgs)

            # Handle Deploy (6-tuple) and Train modes (2/3/5-tuples)
            if isinstance(preds, tuple) and len(preds) == 6:
                # Deploy/Fused mode: (reg3, reg4, reg5, cls3, cls4, cls5)
                preds_reg = preds[:3]
                preds_cls = preds[3:]
                is_deployed = True
            elif isinstance(preds, tuple) and len(preds) == 5:
                # Dual-head train mode: (o2o_box, o2o_cls, neck, o2m_box, o2m_cls)
                preds_reg, preds_cls, _, _, _ = preds
                is_deployed = False
            elif len(preds) == 2:
                # Training mode (Legacy): ([reg_list], [cls_list])
                preds_reg, preds_cls = preds
                is_deployed = False
            elif len(preds) == 3:
                # Training mode (New): ([reg_list], [cls_list], [neck_feats])
                preds_reg, preds_cls, _ = preds
                is_deployed = False
            else:
                raise ValueError(f"Unknown model output format. Length: {len(preds)}")

        # Postprocess timing (anchors, decoding, NMS)
        with dt[2]:
            # Anchors (Cached)
            if current_shape != last_shape:
                anchors_list = []
                strides_list = []
                for i, stride in enumerate(strides):
                    # preds_cls[i] is [B, nc, H, W]
                    # H, W derived from current_shape and stride
                    H, W = int(current_shape[0] / stride), int(current_shape[1] / stride)
                    sy, sx = torch.meshgrid(
                        torch.arange(H, device=device) + 0.5,
                        torch.arange(W, device=device) + 0.5,
                        indexing="ij",
                    )
                    anchors = torch.stack((sx, sy), -1).reshape(-1, 2) * stride
                    anchors_list.append(anchors)
                    strides_list.append(torch.full((H * W, 1), stride, device=device))

                cached_anchors = torch.cat(anchors_list)
                cached_strides = torch.cat(strides_list)
                last_shape = current_shape

            all_anchors = cached_anchors
            all_strides = cached_strides

            # Process Outputs
            pred_cls = []
            pred_reg = []

            for i in range(len(strides)):
                # In deploy mode, sigmoid is already applied in forward()
                c = preds_cls[i].permute(0, 2, 3, 1).reshape(bs, -1, num_classes)
                if not is_deployed:
                    c = c.sigmoid()
                pred_cls.append(c)

                # YOLO26: Direct Regression (4 values: x, y, w, h)
                if reg_max > 1:
                    # DFL Logic (Legacy)
                    r = preds_reg[i].permute(0, 2, 3, 1).reshape(bs, -1, 4 * reg_max)
                else:
                    # Direct Regression (v26)
                    # shape [B, 4, H, W] -> [B, H*W, 4]
                    r = preds_reg[i].permute(0, 2, 3, 1).reshape(bs, -1, 4)

                pred_reg.append(r)

            pred_cls = torch.cat(pred_cls, 1)
            pred_reg = torch.cat(pred_reg, 1)

            # Precompute DFL weight if needed (Optimized)
            dfl_weight = None
            if reg_max > 1:
                dfl_weight = torch.arange(reg_max, device=device).float()

            pred_boxes = decode_bboxes(
                pred_reg, all_anchors, all_strides, xywh=False, dfl_weight=dfl_weight
            )

            # NMS Prep
            # Objectness is max class score here (since no obj head)
            conf, j = pred_cls.max(dim=-1, keepdim=True)

            if nms:
                # Traditional NMS Mode
                nms_input = torch.cat([pred_boxes, conf, pred_cls], dim=-1)
                dets = non_max_suppression(
                    nms_input, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
                )
            else:
                # O2O Mode: Just filter by conf_thres and format as [xyxy, conf, cls]
                dets = []
                for i in range(bs):
                    mask = conf[i].view(-1) > conf_thres
                    if mask.any():
                        # Format: [N, 6] -> (x1, y1, x2, y2, conf, cls)
                        d = torch.cat(
                            [pred_boxes[i][mask], conf[i][mask], j[i][mask].float()], dim=-1
                        )
                        # Keep NMS-free behavior while matching standard detector cap.
                        # This avoids metric bias from unbounded low-quality tail predictions.
                        if d.shape[0] > max_det:
                            keep = d[:, 4].argsort(descending=True)[:max_det]
                            d = d[keep]
                        dets.append(d)
                    else:
                        dets.append(torch.zeros((0, 6), device=device))

            total_preds_above_thresh += sum(len(d) for d in dets)

        # Track image count
        total_images += bs

        metrics.update(dets, targets)

    # Compute Ultralytics-style speed metrics (ms per image)
    speed = {
        k: (dt[i].t / total_images * 1000) if total_images > 0 else 0.0
        for i, k in enumerate(["preprocess", "inference", "postprocess"])
    }

    # Print validation summary in Ultralytics format
    print("\nValidation Summary:")
    print(f"  Images: {total_images}")
    print(f"  Predictions (conf>{conf_thres}): {total_preds_above_thresh}")
    if total_images > 0 and input_shape is not None:
        print(
            f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, "
            f"{speed['postprocess']:.1f}ms postprocess per image at shape {tuple(input_shape)}"
        )
    else:
        print("  Speed: N/A")

    # Compute metrics first to populate self._results
    stats = metrics.compute()

    # Generate metric evaluation plots
    if save_dir:
        metrics.plot_all(save_dir=save_dir, prefix=prefix)

    return stats, metrics
