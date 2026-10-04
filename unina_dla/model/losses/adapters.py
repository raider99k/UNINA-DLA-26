import torch

from unina_dla.model.losses.classic_loss import ClassicDetectionLoss
from unina_dla.model.losses.v26_loss import YOLO26DetectionLoss


def split_fg_mask_to_levels(fg_mask, feat_shapes):
    """
    Split flattened O2O fg_mask [B, N] back to per-level spatial masks.
    """
    batch_size = fg_mask.shape[0]
    masks = []
    offset = 0

    for h, w in feat_shapes:
        level_size = h * w
        level_mask = fg_mask[:, offset : offset + level_size]
        masks.append(level_mask.view(batch_size, 1, h, w).float())
        offset += level_size

    return masks


def _extract_head_strides(head):
    if hasattr(head, "strides"):
        return list(head.strides)
    if hasattr(head, "stride"):
        return list(head.stride) if hasattr(head.stride, "__iter__") else [head.stride]

    n_levels = len(head.o2o_reg_preds) if hasattr(head, "o2o_reg_preds") else 3
    return [8.0 * (2**i) for i in range(n_levels)]


def _build_mock_model(real_model, cfg):
    class MockHead:
        def __init__(self, nc, stride):
            self.nc = nc
            self.stride = torch.tensor(stride).to(next(real_model.parameters()).device)
            self.reg_max = 0
            self.no = nc + 4

    class MockModel:
        def __init__(self, args, head_attrs):
            self.args = args
            self.model = [MockHead(**head_attrs)]

        def parameters(self):
            return real_model.parameters()

    head = real_model.head
    head_attrs = {
        "nc": head.num_classes,
        "stride": _extract_head_strides(head),
    }
    return MockModel(cfg, head_attrs)


class YOLO26LossAdapter:
    """
    Backward-compatible O2O-only adapter.
    """

    def __init__(
        self,
        real_model,
        cfg,
        return_aux=True,
        dup_loss_weight=0.0,
        dup_iou_thr=0.6,
        dup_margin=0.2,
        dup_topk=20,
        hard_bg_loss_weight=0.0,
        hard_bg_iou_thr=0.2,
        hard_bg_topk=64,
        dup_start_pct=0.6,
    ):
        self.device = next(real_model.parameters()).device
        self.return_aux = return_aux
        self.mock_model = _build_mock_model(real_model, cfg)
        self.loss_fn = YOLO26DetectionLoss(
            self.mock_model,
            return_aux=return_aux,
            dup_loss_weight=dup_loss_weight,
            dup_iou_thr=dup_iou_thr,
            dup_margin=dup_margin,
            dup_topk=dup_topk,
            hard_bg_loss_weight=hard_bg_loss_weight,
            hard_bg_iou_thr=hard_bg_iou_thr,
            hard_bg_topk=hard_bg_topk,
            dup_start_pct=dup_start_pct,
        )
        self.loss_fn.device = self.device
        # Dedicated mask extractor: keeps teacher-mask generation isolated from
        # the task loss state used for optimization.
        self.mask_loss_fn = YOLO26DetectionLoss(
            self.mock_model,
            tal_topk=1,
            return_aux=True,
            dup_loss_weight=0.0,
            hard_bg_loss_weight=0.0,
        )
        self.mask_loss_fn.device = self.device

    def set_assigner_topk(self, topk: int, topk2=None):
        if hasattr(self.loss_fn, "set_assigner_topk"):
            try:
                self.loss_fn.set_assigner_topk(topk, topk2=topk2)
            except TypeError:
                self.loss_fn.set_assigner_topk(topk)

    def set_dup_suppression_progress(self, progress: float):
        if hasattr(self.loss_fn, "set_dup_suppression_progress"):
            self.loss_fn.set_dup_suppression_progress(progress)

    def compute_o2o_masks(self, o2o_preds, batch, topk=None, topk2=None):
        loss_fn = self.mask_loss_fn
        prev_mode = loss_fn.training
        prev_topk = getattr(loss_fn, "assigner_topk", None)
        prev_topk2 = getattr(loss_fn, "assigner_topk2", None)
        if (topk is not None or topk2 is not None) and hasattr(loss_fn, "set_assigner_topk"):
            effective_topk = prev_topk if topk is None else topk
            try:
                loss_fn.set_assigner_topk(effective_topk, topk2=topk2)
            except TypeError:
                loss_fn.set_assigner_topk(effective_topk)
        try:
            loss_fn.train(False)
            with torch.no_grad():
                _, _, aux = loss_fn(o2o_preds, batch)
            return split_fg_mask_to_levels(aux["fg_mask"], aux["feat_shapes"])
        finally:
            if (
                (topk is not None or topk2 is not None)
                and prev_topk is not None
                and hasattr(loss_fn, "set_assigner_topk")
            ):
                try:
                    loss_fn.set_assigner_topk(prev_topk, topk2=prev_topk2)
                except TypeError:
                    loss_fn.set_assigner_topk(prev_topk)
            loss_fn.train(prev_mode)

    def __call__(self, preds, batch):
        result = self.loss_fn(preds, batch)
        if not self.return_aux:
            return result

        loss, loss_items, aux = result
        o2o_masks = split_fg_mask_to_levels(aux["fg_mask"], aux["feat_shapes"])
        return loss, loss_items, o2o_masks


class DualYOLO26LossAdapter:
    """
    Dual-branch task-loss adapter:
    - O2O branch: YOLO26DetectionLoss (deploy branch objective)
    - O2M branch: YOLO26DetectionLoss with denser assignment (auxiliary branch)

    This keeps assignment and supervision semantics consistent across branches
    while preserving one-to-many auxiliary pressure through a higher top-k.
    """

    def __init__(
        self,
        real_model,
        cfg,
        return_aux=True,
        o2m_topk=10,
        aux_weight_start=1.0,
        aux_weight_end=0.3,
        dup_loss_weight=0.0,
        dup_iou_thr=0.6,
        dup_margin=0.2,
        dup_topk=20,
        hard_bg_loss_weight=0.0,
        hard_bg_iou_thr=0.2,
        hard_bg_topk=64,
        dup_start_pct=0.6,
    ):
        self.device = next(real_model.parameters()).device
        self.return_aux = return_aux
        self.mock_model = _build_mock_model(real_model, cfg)

        # Keep this attribute name for trainer compatibility.
        self.loss_fn = YOLO26DetectionLoss(
            self.mock_model,
            return_aux=return_aux,
            dup_loss_weight=dup_loss_weight,
            dup_iou_thr=dup_iou_thr,
            dup_margin=dup_margin,
            dup_topk=dup_topk,
            hard_bg_loss_weight=hard_bg_loss_weight,
            hard_bg_iou_thr=hard_bg_iou_thr,
            hard_bg_topk=hard_bg_topk,
            dup_start_pct=dup_start_pct,
        )
        self.loss_fn.device = self.device
        # Dedicated O2O mask loss used only to extract teacher-aligned masks for KD.
        # Isolated to avoid mutating the optimization loss state (assigner top-k,
        # wiou running mean, training/eval toggles).
        self.mask_loss_fn = YOLO26DetectionLoss(
            self.mock_model,
            tal_topk=1,
            return_aux=True,
            dup_loss_weight=0.0,
            hard_bg_loss_weight=0.0,
        )
        self.mask_loss_fn.device = self.device

        # Keep O2M in the same objective family as O2O.
        # The only intentional difference is denser assignment (top-k > 1).
        self.o2m_loss_fn = YOLO26DetectionLoss(
            self.mock_model,
            tal_topk=o2m_topk,
            return_aux=False,
            dup_loss_weight=0.0,
            hard_bg_loss_weight=0.0,
        )
        self.o2m_loss_fn.device = self.device

        self.aux_weight_start = float(aux_weight_start)
        self.aux_weight_end = float(aux_weight_end)
        self.current_aux_weight = float(aux_weight_start)

    def set_assigner_topk(self, topk: int, topk2=None):
        if hasattr(self.loss_fn, "set_assigner_topk"):
            try:
                self.loss_fn.set_assigner_topk(topk, topk2=topk2)
            except TypeError:
                self.loss_fn.set_assigner_topk(topk)

    def set_dup_suppression_progress(self, progress: float):
        if hasattr(self.loss_fn, "set_dup_suppression_progress"):
            self.loss_fn.set_dup_suppression_progress(progress)

    def set_aux_weight(self, weight: float):
        self.current_aux_weight = max(0.0, float(weight))

    def _o2o_forward(self, o2o_preds, batch):
        result = self.loss_fn(o2o_preds, batch)
        if not self.return_aux:
            loss, loss_items = result
            return loss, loss_items, None

        loss, loss_items, aux = result
        o2o_masks = split_fg_mask_to_levels(aux["fg_mask"], aux["feat_shapes"])
        return loss, loss_items, o2o_masks

    def compute_o2o_masks(self, o2o_preds, batch, topk=None, topk2=None):
        loss_fn = self.mask_loss_fn
        prev_mode = loss_fn.training
        prev_topk = getattr(loss_fn, "assigner_topk", None)
        prev_topk2 = getattr(loss_fn, "assigner_topk2", None)
        if (topk is not None or topk2 is not None) and hasattr(loss_fn, "set_assigner_topk"):
            effective_topk = prev_topk if topk is None else topk
            try:
                loss_fn.set_assigner_topk(effective_topk, topk2=topk2)
            except TypeError:
                loss_fn.set_assigner_topk(effective_topk)
        try:
            loss_fn.train(False)
            with torch.no_grad():
                _, _, aux = loss_fn(o2o_preds, batch)
            return split_fg_mask_to_levels(aux["fg_mask"], aux["feat_shapes"])
        finally:
            if (
                (topk is not None or topk2 is not None)
                and prev_topk is not None
                and hasattr(loss_fn, "set_assigner_topk")
            ):
                try:
                    loss_fn.set_assigner_topk(prev_topk, topk2=prev_topk2)
                except TypeError:
                    loss_fn.set_assigner_topk(prev_topk)
            loss_fn.train(prev_mode)

    def __call__(self, preds, batch, aux_weight=None):
        o2o_preds = None
        o2m_preds = None

        if isinstance(preds, dict):
            o2o_preds = (preds["o2o_box"], preds["o2o_cls"])
            if "o2m_box" in preds and "o2m_cls" in preds:
                o2m_preds = (preds["o2m_box"], preds["o2m_cls"])
        elif isinstance(preds, tuple):
            if len(preds) == 2:
                o2o_preds = preds
            elif len(preds) == 4:
                o2o_preds = (preds[0], preds[1])
                o2m_preds = (preds[2], preds[3])
            else:
                raise ValueError(f"Unsupported dual loss tuple length: {len(preds)}")
        else:
            raise TypeError("DualYOLO26LossAdapter expects dict or tuple predictions.")

        o2o_loss, o2o_items, o2o_masks = self._o2o_forward(o2o_preds, batch)

        # O2O-only mode (used for teacher mask extraction or fallback compatibility)
        if o2m_preds is None:
            breakdown = {
                "l_o2o_sum": o2o_loss.detach(),
                "l_o2m_sum": torch.tensor(0.0, device=self.device),
                "o2m_weight": 0.0,
                "o2o_items": o2o_items.detach().clone(),
                "o2m_items": torch.zeros_like(o2o_items),
            }
            return o2o_loss, o2o_items, o2o_masks, breakdown

        w_aux = self.current_aux_weight if aux_weight is None else float(aux_weight)
        o2m_loss, o2m_items = self.o2m_loss_fn(o2m_preds, batch)

        total_loss = o2o_loss + (w_aux * o2m_loss)
        total_items = o2o_items.detach().clone() + (o2m_items.detach().clone() * w_aux)

        breakdown = {
            "l_o2o_sum": o2o_loss.detach(),
            "l_o2m_sum": o2m_loss.detach(),
            "o2m_weight": float(w_aux),
            "o2o_items": o2o_items.detach().clone(),
            "o2m_items": o2m_items.detach().clone(),
        }
        return total_loss, total_items, o2o_masks, breakdown


class ClassicLossAdapter:
    """
    Adapter for baseline classic training with one-to-many assignment.
    """

    def __init__(self, real_model, cfg):
        self.device = next(real_model.parameters()).device
        self.mock_model = _build_mock_model(real_model, cfg)
        self.loss_fn = ClassicDetectionLoss(self.mock_model, tal_topk=10)
        self.loss_fn.device = self.device

    def __call__(self, preds, batch):
        loss, loss_items = self.loss_fn(preds, batch)
        return loss, loss_items, None
