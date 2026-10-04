"""
UNINA-DLA Bounding Box Operations.

Provides hardware-native bounding box decoding, anchor generation, and
numerically stabilized Wise-IoU loss formulations adapted for DLA execution.
"""

import torch

try:
    import torchvision
except ImportError:
    torchvision = None


def dfl_integral(dfl_preds, reg_max=16, weight=None):
    """
    [LEGACY] Used for Distribution Focal Loss (DFL).
    UNUSED in YOLO26 (which uses Direct Regression), but kept for backward compatibility.

    Decodes DFL distribution into scalar coordinates.
    Assumes dfl_preds are unnormalized logits, and applies softmax.

    Args:
        dfl_preds: [B, 4*reg_max, H, W] or [B, N, 4*reg_max] (logits)
        reg_max: number of bins
        weight: optional precomputed weight tensor (1D: [reg_max])
    Returns:
        coords: [B, 4, H, W] or [B, N, 4]
    """
    # Assuming dfl_preds is logits, we need softmax
    # Shape handling
    if dfl_preds.dim() == 4:
        B, C, H, W = dfl_preds.shape
        # [B, 4, 16, H, W]
        dfl_preds = dfl_preds.view(B, 4, reg_max, H, W)
        dfl_preds = dfl_preds.softmax(dim=2)
        # Weight [0, 1, ..., 15]
        if weight is None:
            weight = torch.arange(reg_max, device=dfl_preds.device).float()
        weight = weight.view(1, 1, -1, 1, 1)
        # Sum(P * W)
        return (dfl_preds * weight).sum(dim=2)  # [B, 4, H, W]
    else:
        # Flattened case [B, N, 4*reg_max]
        B, N, C = dfl_preds.shape
        dfl_preds = dfl_preds.view(B, N, 4, reg_max)
        dfl_preds = dfl_preds.softmax(dim=3)
        if weight is None:
            weight = torch.arange(reg_max, device=dfl_preds.device).float()
        weight = weight.view(1, 1, 1, -1)
        return (dfl_preds * weight).sum(dim=3)  # [B, N, 4]


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchors from features.
    Args:
        feats: list of features
        strides: list of strides
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def decode_bboxes(dfl_preds, anchors, strides, xywh=False, dfl_weight=None):
    """
    Decode DFL predictions to bounding boxes.
    dfl_preds: [B, N, 4*reg_max] (flattened) or [B, N, 4] (if already integrated)
    anchors: [N, 2]
    strides: [N, 1]
    dfl_weight: optional precomputed weight tensor
    """
    if dfl_preds.shape[-1] > 4:
        # Assuming reg_max=16
        reg_max = dfl_preds.shape[-1] // 4
        tlrb = dfl_integral(dfl_preds, reg_max=reg_max, weight=dfl_weight)
    else:
        tlrb = dfl_preds

    tlrb = tlrb * strides
    boxes = dist2bbox(tlrb, anchors, xywh=xywh)
    return boxes


def xywh2xyxy(x):
    """
    Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2].
    x: [..., 4]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, xywh=False):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    prediction: [B, N, 4+C] or [B, N, 4+reg_max+C]?
                Assuming prediction is [B, N, 4+C] where 4 is xywh or xyxy
    """
    # Checks
    assert 0 <= conf_thres <= 1, (
        f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    )
    assert 0 <= iou_thres <= 1, (
        f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"
    )

    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 6), device=prediction.device) for _ in range(bs)]

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]
        # if box is xywh, convert to xyxy
        if xywh:
            box = xywh2xyxy(box)

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_det:
            x = x[x[:, 4].argsort(descending=True)[:max_det]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * 7680  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        if torchvision is None:
            raise ImportError("torchvision is required for IoU NMS: pip install torchvision")
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]

    return output


def wise_iou(pred, target, alpha=1.9, delta=3, mean_iou=None):
    """
    Wise-IoU v3 (WIoU v3) Implementation - Numerically Stable.

    A dynamic non-monotonic focusing mechanism for bounding box regression.
    Formula: L_WIoU = r * R_WIoU * L_IoU

    Numerical stability guards:
    - Clamped areas to prevent negative or zero values
    - Clamped l_avg to prevent division by zero
    - Clamped beta to prevent exponential overflow
    - Clamped r to prevent gradient explosion
    - Final NaN replacement guard

    Args:
        pred: Predicted boxes [N, 4] (x1, y1, x2, y2)
        target: Target boxes [N, 4] (x1, y1, x2, y2)
        alpha: Hyperparameter for gradient gain (default 1.9)
        delta: Hyperparameter for outlier degree (default 3)
        mean_iou: Running mean of IoU for v3 dynamic focusing.
                  If None, uses batch mean (behaves like WIoU v2).

    Returns:
         iou: Standard IoU (for metrics)
         l_wiou: Wise-IoU Loss (1 - IoU_wiou)
    """
    # Clamp boxes to prevent degenerate (negative) coordinates
    pred = pred.clamp(min=0)
    target = target.clamp(min=0)

    # Standard IoU
    tl = torch.max(pred[:, :2], target[:, :2])
    br = torch.min(pred[:, 2:], target[:, 2:])

    # Clamp areas to prevent negative/zero (degenerate boxes)
    area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1).clamp(min=1e-7)
    area_t = torch.prod(target[:, 2:] - target[:, :2], 1).clamp(min=1e-7)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1).clamp(min=0) * en
    iou = area_i / (area_p + area_t - area_i + 1e-7)

    # Clamp IoU to valid range
    iou = iou.clamp(min=0.0, max=1.0)

    # Distance Penalty (R_WIoU)
    # Center points
    cp = (pred[:, :2] + pred[:, 2:]) / 2
    ct = (target[:, :2] + target[:, 2:]) / 2

    # Enclosing Box
    b_tl = torch.min(pred[:, :2], target[:, :2])
    b_br = torch.max(pred[:, 2:], target[:, 2:])
    box_w = (b_br[:, 0] - b_tl[:, 0]).clamp(min=1e-5)
    box_h = (b_br[:, 1] - b_tl[:, 1]).clamp(min=1e-5)

    # Distance^2 / Diagonal^2
    # WIoU formulation (Tong et al.): The enclosing box diagonal is detached (W_g*, H_g*)
    # to prevent backpropagation from artificially expanding predicted bounding boxes.
    dist2 = torch.sum((cp - ct) ** 2, dim=1)
    diag2 = (box_w**2 + box_h**2 + 1e-5).detach()

    # Clamp the ratio to prevent exp() overflow
    dist_ratio = (dist2 / diag2).clamp(max=10.0)

    # Distance Attention
    dist_attn = torch.exp(dist_ratio)

    # WIoU v1 Loss
    l_iou = 1.0 - iou
    l_wiou_v1 = dist_attn * l_iou

    # WIoU v3: Dynamic Non-Monotonic Focusing
    l_iou_detach = l_iou.detach()

    if mean_iou is not None:
        # WIoU v3: Use running history (clamp l_avg to avoid division by zero)
        l_avg = max(1.0 - mean_iou, 0.01)
        beta = l_iou_detach / (l_avg + 1e-5)
    else:
        # WIoU v2: Use batch mean
        batch_mean = l_iou_detach.mean() + 1e-5
        beta = l_iou_detach / batch_mean

    # Clamp beta to prevent exponential overflow
    beta = beta.clamp(min=0.1, max=10.0)

    # Focusing Factor r = beta / (delta * alpha^(beta - delta))
    exponent = (beta - delta).clamp(min=-10.0, max=10.0)
    r = beta / (delta * torch.pow(alpha, exponent) + 1e-5)

    # Clamp r to prevent gradient explosion
    r = r.clamp(min=0.01, max=10.0)

    # Final Loss
    l_wiou = r * l_wiou_v1

    return iou, l_wiou


def wasserstein_loss(pred, target, eps=1e-5, constant=12.0):
    """
    Normalized Wasserstein Distance (NWD) for Small Object Detection.

    Treats bounding boxes as 2D Gaussian distributions and computes the
    Wasserstein-2 distance between them. Unlike IoU, NWD provides smooth
    gradients even when boxes don't overlap (IoU=0), which is critical
    for small/distant objects in Formula Student racing.

    Why NWD over IoU for small objects?
    ------------------------------------
    - IoU degrades mathematically when objects are < 16x16 pixels
    - Slight pixel shifts cause IoU to fluctuate wildly (0.0 → 0.5 → 0.0)
    - Gradients vanish or explode, causing unstable training
    - NWD treats boxes as probability distributions, giving "how close" info
      even for non-overlapping predictions

    Reference: https://arxiv.org/abs/2110.13389

    Args:
        pred: Predicted boxes [N, 4] in xyxy format
        target: Target boxes [N, 4] in xyxy format
        eps: Small epsilon for numerical stability
        constant: Normalization constant for image scale.
                  Default 12.0 works well for 640x640 images.
                  Adjust proportionally for other resolutions.

    Returns:
        loss: NWD loss [N], range [0, 1] where 0 = perfect match
    """
    # xyxy -> cx, cy, w, h
    b1_cx = (pred[:, 0] + pred[:, 2]) / 2
    b1_cy = (pred[:, 1] + pred[:, 3]) / 2
    b1_w = (pred[:, 2] - pred[:, 0]) + eps
    b1_h = (pred[:, 3] - pred[:, 1]) + eps

    b2_cx = (target[:, 0] + target[:, 2]) / 2
    b2_cy = (target[:, 1] + target[:, 3]) / 2
    b2_w = (target[:, 2] - target[:, 0]) + eps
    b2_h = (target[:, 3] - target[:, 1]) + eps

    # Wasserstein-2 distance between two 2D axis-aligned Gaussians
    # W2² = ||μ1 - μ2||² + ||σ1 - σ2||²
    # For boxes: σ = (w/2, h/2) approximation
    center_dist2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    wh_dist2 = ((b1_w - b2_w) ** 2 + (b1_h - b2_h) ** 2) / 4

    wasserstein_2 = center_dist2 + wh_dist2

    # Normalize to [0, 1] using exponential decay
    # nwd ∈ [0, 1] where 1 = perfect match, 0 = far apart
    nwd = torch.exp(-torch.sqrt(wasserstein_2 + eps) / constant)

    # Return loss (1 - nwd) so 0 = perfect, 1 = worst
    return 1.0 - nwd
