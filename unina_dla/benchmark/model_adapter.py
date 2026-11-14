"""
UNINA-DLA Model Adapter Module.

Provides unified adapters for benchmarking across diverse model formats:
- UNINA-DLA PyTorch Checkpoints (.pth: FP32, Distilled, 2:4 Sparse, INT8 QAT)
- Ultralytics UNINA Converted Models (.pt)
- Standard Ultralytics YOLO / Teacher Models (.pt)
- ONNX Models (.onnx) via ONNX Runtime (CUDA / TensorRT / CPU)
- TensorRT Engines (.engine / .plan) for Jetson Orin DLA Core 0/1 and GPU
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from unina_dla.benchmark.config import ModelSpec
from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version, detect_upsample_mode
from unina_dla.utils.box_ops import decode_bboxes, non_max_suppression
from unina_dla.utils.metrics import UNINAMetrics
from unina_dla.utils.model_utils import model_summary, strip_engine_metadata

_ANCHOR_CACHE: Dict[Tuple[Tuple[Tuple[int, int], ...], str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _decode_unina_6tensors(
    preds: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    image_shape: Tuple[int, int],
    strides: List[int],
    num_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    nms: bool = False,
    max_det: int = 300,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Decodes UNINA 6-tensor output: (reg3, reg4, reg5, cls3, cls4, cls5).
    Applies direct LTRB decoding + O2O Hungarian filtering (or optional IoU NMS).
    Uses cached anchor grids to minimize GPU allocations and latency.
    """
    if len(preds) != 6:
        raise ValueError(f"Expected 6 tensors for UNINA deploy format, got {len(preds)}")

    preds_reg = preds[:3]
    preds_cls = preds[3:]
    bs = preds_reg[0].shape[0]
    dev = device or preds_reg[0].device
    dev_key = str(dev)

    # Check or populate anchor cache (avoids repeated meshgrid & cat allocations)
    shapes_key = tuple((int(r.shape[2]), int(r.shape[3])) for r in preds_reg)
    cache_key = (shapes_key, dev_key)

    if cache_key in _ANCHOR_CACHE:
        anchors_cat, strides_cat = _ANCHOR_CACHE[cache_key]
    else:
        all_anchors = []
        all_strides = []
        for i, stride in enumerate(strides):
            h, w = shapes_key[i]
            sy, sx = torch.meshgrid(
                torch.arange(h, device=dev, dtype=torch.float32) + 0.5,
                torch.arange(w, device=dev, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            anchors = torch.stack((sx, sy), dim=-1).reshape(-1, 2) * stride
            stride_t = torch.full((h * w, 1), stride, device=dev, dtype=torch.float32)
            all_anchors.append(anchors)
            all_strides.append(stride_t)
        anchors_cat = torch.cat(all_anchors, dim=0)
        strides_cat = torch.cat(all_strides, dim=0)
        _ANCHOR_CACHE[cache_key] = (anchors_cat, strides_cat)

    pred_cls_list = []
    pred_reg_list = []
    for i in range(3):
        cls_t = preds_cls[i]
        c = cls_t.permute(0, 2, 3, 1).reshape(bs, -1, num_classes)
        pred_cls_list.append(c)
        r = preds_reg[i].permute(0, 2, 3, 1).reshape(bs, -1, 4)
        pred_reg_list.append(r)

    pred_cls_cat = torch.cat(pred_cls_list, dim=1)
    pred_reg_cat = torch.cat(pred_reg_list, dim=1)

    pred_boxes = decode_bboxes(pred_reg_cat, anchors_cat, strides_cat, xywh=False)
    conf, cls_idx = pred_cls_cat.max(dim=-1, keepdim=True)

    if nms:
        # Traditional NMS Mode
        nms_input = torch.cat([pred_boxes, conf, pred_cls_cat], dim=-1)
        return non_max_suppression(
            nms_input, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
        )

    # Native UNINA O2O Mode: parallel GPU top-k filtering (zero-sync sorted candidate selection)
    dets: List[torch.Tensor] = []
    conf_flat = conf.squeeze(-1)  # [bs, N]
    k = min(max_det, conf_flat.shape[1])
    top_scores, top_indices = torch.topk(conf_flat, k=k, dim=1, largest=True, sorted=True)

    for i in range(bs):
        mask = top_scores[i] > conf_thres
        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if valid_idx.numel() > 0:
            sel_idx = top_indices[i][valid_idx]
            d = torch.cat(
                [pred_boxes[i][sel_idx], top_scores[i][valid_idx, None], cls_idx[i][sel_idx].float()],
                dim=-1,
            )
            dets.append(d)
        else:
            dets.append(torch.zeros((0, 6), device=dev))
    return dets


class BaseModelAdapter(ABC):
    """Abstract base adapter for any object detection candidate model."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        self.spec = spec
        self.device = device
        self.num_classes = num_classes
        self.names = names or {i: f"class_{i}" for i in range(num_classes)}
        self.imgsz = imgsz
        self.name = spec.name
        self.path = Path(spec.path)
        self.backend = "Generic"
        self.strides = [8, 16, 32]

    def to_device(self, device: torch.device) -> None:
        """Move underlying model weights to specified device if applicable."""
        self.device = device

    def get_file_size_mb(self) -> float:
        """Return model file size on disk in megabytes."""
        if self.path.exists():
            return self.path.stat().st_size / (1024.0 * 1024.0)
        return 0.0

    def get_weight_size_mb(self) -> float:
        """
        Return pure model weight footprint in megabytes (excluding optimizer/EMA duplicate states).
        Falls back to disk file size for engine/onnx models.
        """
        params = self.get_num_params()
        if params is not None and params > 0:
            if getattr(self.spec, "qat", False) or "[INT8" in getattr(self, "backend", ""):
                bytes_per_param = 1.0
            elif getattr(self.spec, "half", False) or "(FP16)" in getattr(self, "backend", ""):
                bytes_per_param = 2.0
            else:
                bytes_per_param = 4.0
            return (params * bytes_per_param) / (1024.0 * 1024.0)
        return self.get_file_size_mb()

    @abstractmethod
    def get_num_params(self) -> Optional[int]:
        """Return total parameter count, or None if not applicable."""
        pass

    @abstractmethod
    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        """Return estimated GFLOPs, or None if not applicable."""
        pass

    @abstractmethod
    def forward_raw(self, x: torch.Tensor) -> Any:
        """Execute raw forward pass on device tensor [B, 3, H, W]."""
        pass

    @abstractmethod
    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """Convert raw model output to list of detection tensors [N, 6] (x1, y1, x2, y2, conf, cls)."""
        pass

    def warmup(self, dummy_input: torch.Tensor, iterations: int = 30) -> None:
        """Run warm-up iterations to stabilize clocks and memory buffers."""
        with torch.no_grad():
            for _ in range(iterations):
                preds = self.forward_raw(dummy_input)
                _ = self.postprocess(
                    preds,
                    (dummy_input.shape[2], dummy_input.shape[3]),
                    conf_thres=self.spec.extra_args.get("conf", 0.25),
                    nms=self.spec.nms,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def eval_validation_loader(
        self,
        dataloader: Any,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        nms: Optional[bool] = None,
        max_det: int = 300,
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation dataloader using standard UNINAMetrics.
        Matches official Ultralytics matching mathematics.
        """
        target_h, target_w = getattr(self, "imgsz", (640, 640))
        use_nms = self.spec.nms if nms is None else nms
        metrics = UNINAMetrics(num_classes=self.num_classes, names=self.names)

        with torch.no_grad():
            for batch in dataloader:
                imgs = batch["img"].to(self.device).float() / 255.0
                bs, _, h, w = imgs.shape

                # Process ground truth
                targets = []
                if "bboxes" in batch:
                    b_idx = batch["batch_idx"].to(self.device)
                    b_cls = batch["cls"].to(self.device).view(-1, 1)
                    b_box = batch["bboxes"].to(self.device)

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
                            targets.append(torch.zeros((0, 5), device=self.device))
                else:
                    targets = [torch.zeros((0, 5), device=self.device) for _ in range(bs)]

                # Check if input batch needs letterbox adaptation to model's expected shape
                if (h, w) != (target_h, target_w):
                    scale = min(target_h / h, target_w / w)
                    new_h, new_w = int(round(h * scale)), int(round(w * scale))
                    pad_top = (target_h - new_h) // 2
                    pad_left = (target_w - new_w) // 2

                    if (new_h, new_w) != (h, w):
                        imgs_res = torch.nn.functional.interpolate(
                            imgs, size=(new_h, new_w), mode="bilinear", align_corners=False
                        )
                    else:
                        imgs_res = imgs
                    imgs_padded = torch.full(
                        (bs, 3, target_h, target_w),
                        114.0 / 255.0,
                        dtype=imgs.dtype,
                        device=imgs.device,
                    )
                    imgs_padded[:, :, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = imgs_res
                    imgs = imgs_padded

                    # Adjust ground truth bounding boxes proportionally
                    adjusted_targets = []
                    for t in targets:
                        if t.shape[0] > 0:
                            t_adj = t.clone()
                            t_adj[:, 1] = t_adj[:, 1] * scale + pad_left
                            t_adj[:, 2] = t_adj[:, 2] * scale + pad_top
                            t_adj[:, 3] = t_adj[:, 3] * scale + pad_left
                            t_adj[:, 4] = t_adj[:, 4] * scale + pad_top
                            adjusted_targets.append(t_adj)
                        else:
                            adjusted_targets.append(t)
                    targets = adjusted_targets
                    cur_h, cur_w = target_h, target_w
                else:
                    cur_h, cur_w = h, w

                # Inference
                raw_preds = self.forward_raw(imgs)

                # Postprocess
                dets = self.postprocess(
                    raw_preds,
                    image_shape=(cur_h, cur_w),
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    nms=use_nms,
                    max_det=max_det,
                )
                dets = [d.to(self.device) for d in dets]
                targets = [t.to(self.device) for t in targets]

                metrics.update(dets, targets)

        stats = metrics.compute()
        return stats


class UNINAPyTorchAdapter(BaseModelAdapter):
    """Adapter for native UNINA-DLA PyTorch checkpoints (.pth)."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        super().__init__(spec, device, num_classes, names, imgsz)
        self.backend = f"PyTorch ({device.type.upper()})"
        self.model = self._load_model()
        self.model.to(self.device).eval()

    def to_device(self, device: torch.device) -> None:
        self.device = device
        if hasattr(self, "model") and hasattr(self.model, "to"):
            self.model.to(device)

    def _load_model(self) -> nn.Module:
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.path}")

        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)

        # Auto-detect QAT checkpoint
        is_qat = self.spec.qat
        if not is_qat and isinstance(ckpt, dict):
            sample_keys = []
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                sample_keys = list(ckpt["model"].keys())
            elif "ema" in ckpt and isinstance(ckpt["ema"], dict):
                sample_keys = list(ckpt["ema"].keys())
            else:
                sample_keys = list(ckpt.keys())
            for k in sample_keys:
                if "_amax" in k or "quant" in k:
                    is_qat = True
                    break

        # Select weights: prefer EMA for non-QAT models if available (matches training validation)
        if not is_qat and isinstance(ckpt, dict):
            if "ema" in ckpt and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
                raw_sd = ckpt["ema"]
                print(f"[UNINA Adapter] Using EMA weights from {self.path.name} (matches training validation).")
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                raw_sd = ckpt["model"]
            else:
                raw_sd = ckpt
        elif isinstance(ckpt, dict) and "model" in ckpt:
            raw_sd = ckpt["model"]
        else:
            raw_sd = ckpt

        # Clean state_dict: remove compiler wrappers and profiling artifacts
        state_dict = {}
        for k, v in raw_sd.items():
            if k.endswith(("total_ops", "total_params")):
                continue
            k_clean = k.replace("_orig_mod.", "")
            state_dict[k_clean] = v

        # Detect if checkpoint has structurally fused RepVGG blocks
        is_fused = any("rbr_reparam" in k for k in state_dict.keys())

        # Build architecture: auto-detect generation and upsample mode from state_dict
        detected_version = detect_architecture_version(state_dict)
        detected_upsample = detect_upsample_mode(state_dict)
        model = UNINA_DLA(
            num_classes=self.num_classes,
            deploy=is_fused,
            version=detected_version,
            upsample_mode=detected_upsample,
        )

        if is_qat:
            try:
                from unina_dla.utils.qat_utils import prepare_qat_model

                model.eval()
                # deploy=True informs prepare_qat_model that blocks are already fused, avoiding re-fusion warnings
                model = prepare_qat_model(model, verification_input_size=None)
                self.backend += " [INT8 QAT]"
            except ImportError:
                print(f"[WARNING] QAT utils not found for {self.path}; loading as standard.")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Filter out known auxiliary keys (e.g. o2m branch when evaluating in deploy mode)
        filtered_missing = [
            k
            for k in missing
            if not (k.startswith("head.o2m_") or k.endswith(("total_ops", "total_params")))
        ]
        filtered_unexpected = [
            k
            for k in unexpected
            if not (k.startswith("head.o2m_") or k.endswith(("total_ops", "total_params")))
        ]

        if filtered_missing:
            print(
                f"[WARNING] Model load missing {len(filtered_missing)} keys for {self.path.name}: {filtered_missing[:5]}..."
            )
        if filtered_unexpected:
            print(
                f"[WARNING] Model load unexpected {len(filtered_unexpected)} keys for {self.path.name}: {filtered_unexpected[:5]}..."
            )

        model.eval()
        if not is_qat and not is_fused:
            # Structurally re-parameterize RepVGG blocks into single 3x3 convs
            model.switch_to_deploy()
            self.backend += " [Fused Deploy]"
        elif not is_qat and is_fused:
            self.backend += " [Fused Deploy]"
        else:
            model.deploy = True
            if hasattr(model, "head") and hasattr(model.head, "deploy"):
                model.head.deploy = True
            if is_qat:
                try:
                    from unina_dla.utils.qat_utils import enable_quantization
                    enable_quantization(model, device=self.device)
                except Exception:
                    pass

        if self.spec.half and self.device.type == "cuda" and not is_qat:
            model = model.half()
            self.backend += " (FP16)"

        return model

    def get_num_params(self) -> Optional[int]:
        return sum(p.numel() for p in self.model.parameters())

    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        try:
            summary = model_summary(
                self.model,
                input_size=(1, 3, imgsz[0], imgsz[1]),
                verbose=False,
                device=str(self.device),
            )
            return summary.get("gflops", None)
        except Exception:
            return None
        finally:
            self.to_device(self.device)

    def forward_raw(self, x: torch.Tensor) -> Any:
        if self.spec.half and x.dtype != torch.float16 and self.device.type == "cuda":
            x = x.half()
        try:
            p = next(self.model.parameters())
            if p.device != x.device:
                self.model.to(x.device)
                self.device = x.device
        except (StopIteration, Exception):
            pass
        return self.model(x)

    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        if isinstance(raw_preds, tuple) and len(raw_preds) == 6:
            return _decode_unina_6tensors(
                raw_preds,
                image_shape=image_shape,
                strides=self.strides,
                num_classes=self.num_classes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nms=nms,
                max_det=max_det,
                device=self.device,
            )
        raise ValueError(f"Unexpected raw prediction format from UNINA PyTorch model: {type(raw_preds)}")


class UNINAUltralyticsPTAdapter(BaseModelAdapter):
    """Adapter for UNINA converted Ultralytics models (.pt)."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        super().__init__(spec, device, num_classes, names, imgsz)
        self.backend = f"UNINA Ultralytics .pt ({device.type.upper()})"
        self.wrapped_model = self._load()

    def to_device(self, device: torch.device) -> None:
        self.device = device
        if hasattr(self, "wrapped_model") and hasattr(self.wrapped_model, "to"):
            self.wrapped_model.to(device)

    def _load(self) -> nn.Module:
        ckpt = torch.load(self.path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model = ckpt["model"]
        else:
            model = ckpt
        model = model.to(self.device).eval()
        if self.spec.half and self.device.type == "cuda":
            model = model.half()
        return model

    def get_num_params(self) -> Optional[int]:
        return sum(p.numel() for p in self.wrapped_model.parameters())

    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        try:
            summary = model_summary(
                self.wrapped_model,
                input_size=(1, 3, imgsz[0], imgsz[1]),
                verbose=False,
                device=str(self.device),
            )
            return summary.get("gflops", None)
        except Exception:
            return None
        finally:
            self.to_device(self.device)

    def forward_raw(self, x: torch.Tensor) -> Any:
        if self.spec.half and x.dtype != torch.float16 and self.device.type == "cuda":
            x = x.half()
        try:
            p = next(self.wrapped_model.parameters())
            if p.device != x.device:
                self.wrapped_model.to(x.device)
                self.device = x.device
        except (StopIteration, Exception):
            pass
        return self.wrapped_model(x)

    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        return _decode_unina_6tensors(
            raw_preds,
            image_shape=image_shape,
            strides=self.strides,
            num_classes=self.num_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nms=nms,
            max_det=max_det,
            device=self.device,
        )


class YOLOAdapter(BaseModelAdapter):
    """Adapter for standard Ultralytics YOLO models (Teacher / Baselines)."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        super().__init__(spec, device, num_classes, names, imgsz)
        self.backend = f"Ultralytics YOLO ({device.type.upper()})"
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(str(self.path))
            try:
                self.yolo.fuse()
                self.backend += " [Fused]"
            except Exception:
                pass
            self.yolo.to(str(self.device))
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "eval"):
                self.yolo.model.eval()
        except ImportError as e:
            raise ImportError("Ultralytics library is required to benchmark YOLO models: pip install ultralytics") from e

    def to_device(self, device: torch.device) -> None:
        self.device = device
        if hasattr(self, "yolo"):
            try:
                self.yolo.to(str(device))
            except Exception:
                pass
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "to"):
                self.yolo.model.to(device)

    def get_num_params(self) -> Optional[int]:
        try:
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "parameters"):
                return sum(p.numel() for p in self.yolo.model.parameters())
        except Exception:
            pass
        return None

    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        try:
            if hasattr(self.yolo, "model"):
                summary = model_summary(
                    self.yolo.model,
                    input_size=(1, 3, imgsz[0], imgsz[1]),
                    verbose=False,
                    device=str(self.device),
                )
                return summary.get("gflops", None)
        except Exception:
            pass
        finally:
            self.to_device(self.device)
        return None

    def forward_raw(self, x: torch.Tensor) -> Any:
        try:
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "parameters"):
                p = next(self.yolo.model.parameters())
                if p.device != x.device:
                    self.yolo.to(str(x.device))
        except (StopIteration, Exception):
            pass
        return self.yolo.model(x)

    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = True,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        try:
            from ultralytics.utils import nms as ultra_nms
            # Standard YOLO output shape is [B, 4+nc, anchors]
            p = raw_preds[0] if isinstance(raw_preds, (list, tuple)) else raw_preds
            return ultra_nms.non_max_suppression(
                p,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )
        except Exception:
            # Fallback for unexpected shapes
            bs = raw_preds[0].shape[0] if isinstance(raw_preds, (list, tuple)) else raw_preds.shape[0]
            return [torch.zeros((0, 6), device=self.device) for _ in range(bs)]

    def eval_validation_loader(
        self,
        dataloader: Any,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        nms: Optional[bool] = None,
        max_det: int = 300,
    ) -> Dict[str, Any]:
        """Use Ultralytics native validator if available for exact consistency."""
        try:
            # Run through dataloader evaluation
            return super().eval_validation_loader(
                dataloader, conf_thres=conf_thres, iou_thres=iou_thres, nms=True, max_det=max_det
            )
        except Exception as e:
            print(f"[WARNING] Fallback in YOLO evaluation: {e}")
            return super().eval_validation_loader(
                dataloader, conf_thres=conf_thres, iou_thres=iou_thres, nms=True, max_det=max_det
            )


class ONNXAdapter(BaseModelAdapter):
    """Adapter for ONNX models (.onnx) using ONNX Runtime (CUDA / TensorRT / CPU)."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        super().__init__(spec, device, num_classes, names, imgsz)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime is required to benchmark ONNX models: pip install onnxruntime-gpu") from e

        available_providers = ort.get_available_providers()
        providers = []

        if device.type == "cuda":
            if "TensorrtExecutionProvider" in available_providers and spec.extra_args.get("use_trt_ep", False):
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available_providers:
                device_id = device.index if device.index is not None else 0
                providers.append(("CUDAExecutionProvider", {"device_id": device_id}))

        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.backend = f"ONNX Runtime ({self.active_provider})"

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_unina_6tensors = (len(self.output_names) == 6)

    def get_num_params(self) -> Optional[int]:
        try:
            import onnx
            model = onnx.load(str(self.path))
            total_params = 0
            for init in model.graph.initializer:
                total_params += int(np.prod(init.dims))
            return total_params
        except Exception:
            return None

    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        return None

    def forward_raw(self, x: torch.Tensor) -> Any:
        x_np = x.detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: x_np})
        # Convert outputs back to torch.Tensor on device
        return tuple(torch.from_numpy(arr).to(self.device) for arr in outputs)

    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        if self.is_unina_6tensors:
            return _decode_unina_6tensors(
                raw_preds,
                image_shape=image_shape,
                strides=self.strides,
                num_classes=self.num_classes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nms=nms,
                max_det=max_det,
                device=self.device,
            )
        # Fallback for standard 1-tensor output
        try:
            from ultralytics.utils import nms as ultra_nms
            p = raw_preds[0]
            return ultra_nms.non_max_suppression(
                p, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
            )
        except Exception:
            bs = raw_preds[0].shape[0]
            return [torch.zeros((0, 6), device=self.device) for _ in range(bs)]


class TensorRTAdapter(BaseModelAdapter):
    """Adapter for TensorRT engines (.engine / .plan) targeting Jetson Orin DLA or GPU."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        num_classes: int = 4,
        names: Optional[Dict[int, str]] = None,
        imgsz: Tuple[int, int] = (640, 640),
    ):
        super().__init__(spec, device, num_classes, names, imgsz)
        target_name = f"DLA-{spec.dla_core}" if spec.dla_core is not None else "GPU"
        self.backend = f"TensorRT ({target_name})"
        self._init_runtime()

    def _init_runtime(self) -> None:
        """Initialize direct native TensorRT execution context."""
        try:
            import tensorrt as trt

            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.path, "rb") as f:
                raw_bytes = f.read()
            engine_bytes = strip_engine_metadata(raw_bytes)
            if len(engine_bytes) != len(raw_bytes):
                stripped = len(raw_bytes) - len(engine_bytes)
                print(f"[TensorRT] Stripped {stripped}-byte Ultralytics metadata header from {self.path.name}")
            with trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None (check TensorRT version/GPU match).")
            self.context = self.engine.create_execution_context()
            self.is_raw_trt = True

            # Use dedicated non-default CUDA stream to avoid TRT enqueueV3 synchronization warnings
            if self.device.type == "cuda":
                self.trt_stream = torch.cuda.Stream(device=self.device)
            else:
                self.trt_stream = None

            print(f"[TensorRT] Loaded via direct native TensorRT runtime: {self.path.name}")

            # Discover engine native static input resolution
            is_trt10 = hasattr(self.engine, "num_io_tensors")
            n_tensors = self.engine.num_io_tensors if is_trt10 else self.engine.num_bindings
            for i in range(n_tensors):
                name = self.engine.get_tensor_name(i) if is_trt10 else self.engine.get_binding_name(i)
                is_input = (
                    (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                    if is_trt10
                    else self.engine.binding_is_input(i)
                )
                if is_input:
                    shape = tuple(self.engine.get_tensor_shape(name)) if is_trt10 else tuple(self.engine.get_binding_shape(i))
                    if len(shape) == 4 and shape[2] > 0 and shape[3] > 0:
                        self.imgsz = (int(shape[2]), int(shape[3]))
                        print(f"[TensorRT] Engine input '{name}' native resolution: (H={self.imgsz[0]}, W={self.imgsz[1]})")
                    break
        except Exception as trt_err:
            raise RuntimeError(
                f"Failed to load TensorRT engine '{self.path}'. Ensure TensorRT is installed on this NVIDIA system. Error: {trt_err}"
            )

    def get_num_params(self) -> Optional[int]:
        return None

    def get_gflops(self, imgsz: Tuple[int, int]) -> Optional[float]:
        return None

    def forward_raw(self, x: torch.Tensor) -> Any:
        return self._execute_raw_trt(x)

    def _init_buffers(self, input_shape: Tuple[int, ...]) -> None:
        """Pre-allocate reusable device buffers for zero-allocation inference timing."""
        import tensorrt as trt

        self._cached_input_shape = input_shape
        self._trt_bindings = []
        self._trt_outputs = []
        self._io_tensor_names = []

        is_trt10 = hasattr(self.engine, "num_io_tensors")
        n_tensors = self.engine.num_io_tensors if is_trt10 else self.engine.num_bindings

        for i in range(n_tensors):
            name = self.engine.get_tensor_name(i) if is_trt10 else self.engine.get_binding_name(i)
            is_input = (
                (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                if is_trt10
                else self.engine.binding_is_input(i)
            )
            self._io_tensor_names.append((name, is_input))

            if is_input:
                if hasattr(self.context, "set_input_shape"):
                    ok = self.context.set_input_shape(name, input_shape)
                    if not ok:
                        engine_shape = (
                            tuple(self.engine.get_tensor_shape(name))
                            if is_trt10
                            else tuple(self.engine.get_binding_shape(i))
                        )
                        raise ValueError(
                            f"TensorRT failed to set input shape {input_shape} for '{name}'. "
                            f"Engine expects dimensions {engine_shape}. "
                            f"Please verify resolution matches engine export shape."
                        )
                in_dtype = (
                    trt.nptype(self.engine.get_tensor_dtype(name))
                    if hasattr(self.engine, "get_tensor_dtype")
                    else (
                        trt.nptype(self.engine.get_binding_dtype(i))
                        if hasattr(self.engine, "get_binding_dtype")
                        else np.float32
                    )
                )
                self._input_torch_dtype = torch.float16 if in_dtype == np.float16 else torch.float32
                self._trt_bindings.append(0)
            else:
                shape = (
                    tuple(self.context.get_tensor_shape(name))
                    if hasattr(self.context, "get_tensor_shape")
                    else tuple(self.engine.get_binding_shape(i))
                )
                dtype = (
                    trt.nptype(self.engine.get_tensor_dtype(name))
                    if hasattr(self.engine, "get_tensor_dtype")
                    else np.float32
                )
                torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
                out = torch.empty(shape, dtype=torch_dtype, device=self.device)
                self._trt_outputs.append((name, out))
                self._trt_bindings.append(out.data_ptr())

    def _execute_raw_trt(self, x: torch.Tensor) -> Any:
        # Re-initialize device buffers if input shape changes
        if getattr(self, "_cached_input_shape", None) != tuple(x.shape):
            self._init_buffers(tuple(x.shape))

        # Auto-cast input tensor to match engine input precision (e.g. FP16 engine with FP32 dataloader input)
        target_dtype = getattr(self, "_input_torch_dtype", None)
        if target_dtype is not None and x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        # Assign input pointer
        for i, (name, is_input) in enumerate(self._io_tensor_names):
            if is_input:
                self._trt_bindings[i] = x.data_ptr()
                if hasattr(self.context, "set_tensor_address"):
                    self.context.set_tensor_address(name, x.data_ptr())

        # Execute using appropriate API on dedicated CUDA stream
        stream_ptr = (
            self.trt_stream.cuda_stream
            if self.trt_stream
            else torch.cuda.current_stream(self.device).cuda_stream
        )

        if hasattr(self.context, "set_tensor_address"):
            for name, out in self._trt_outputs:
                self.context.set_tensor_address(name, out.data_ptr())
            self.context.execute_async_v3(stream_ptr)
        elif hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_ptr)
        elif hasattr(self.context, "execute_v2"):
            self.context.execute_v2(self._trt_bindings)
        else:
            raise RuntimeError("Unsupported TensorRT execution context interface.")

        if self.trt_stream:
            # Sync torch current stream with trt_stream
            torch.cuda.current_stream(self.device).wait_stream(self.trt_stream)

        # Map outputs to canonical UNINA 6-tensor order if matching names are found
        out_dict = {name: out for name, out in self._trt_outputs}
        canonical_6 = ["reg3", "reg4", "reg5", "cls3", "cls4", "cls5"]
        if all(k in out_dict for k in canonical_6):
            return tuple(out_dict[k] for k in canonical_6)

        return tuple(out for _, out in self._trt_outputs)

    def postprocess(
        self,
        raw_preds: Any,
        image_shape: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.6,
        nms: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        if isinstance(raw_preds, (list, tuple)) and len(raw_preds) == 6:
            return _decode_unina_6tensors(
                raw_preds,
                image_shape=image_shape,
                strides=self.strides,
                num_classes=self.num_classes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nms=nms,
                max_det=max_det,
                device=self.device,
            )
        # Standard 1-tensor output (e.g. YOLO TensorRT engine)
        if isinstance(raw_preds, (list, tuple)) and len(raw_preds) >= 1:
            try:
                from ultralytics.utils import nms as ultra_nms

                p = raw_preds[0]
                if (
                    p.shape[-1] != 6
                    and p.shape[-1] == 4 + self.num_classes
                    and p.shape[1] != 4 + self.num_classes
                ):
                    p = p.transpose(1, 2)
                return ultra_nms.non_max_suppression(
                    p, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
                )
            except Exception:
                pass
        return [torch.zeros((0, 6), device=self.device) for _ in range(1)]


def load_model_adapter(
    spec: ModelSpec,
    device: torch.device,
    num_classes: int = 4,
    names: Optional[Dict[int, str]] = None,
    imgsz: Tuple[int, int] = (640, 640),
) -> BaseModelAdapter:
    """
    Factory function to instantiate the proper model adapter based on model type or file extension.
    """
    p = Path(spec.path)
    ext = p.suffix.lower()
    m_type = spec.model_type.lower()

    if m_type == "tensorrt" or ext in (".engine", ".plan"):
        return TensorRTAdapter(spec, device, num_classes, names, imgsz)

    if m_type == "onnx" or ext == ".onnx":
        return ONNXAdapter(spec, device, num_classes, names, imgsz)

    if m_type == "unina_pth" or ext == ".pth":
        return UNINAPyTorchAdapter(spec, device, num_classes, names, imgsz)

    if ext == ".pt":
        # Check if converted UNINA .pt or standard YOLO
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu")
                if isinstance(ckpt, dict) and "unina_conversion" in ckpt:
                    return UNINAUltralyticsPTAdapter(spec, device, num_classes, names, imgsz)
            except Exception:
                pass
        return YOLOAdapter(spec, device, num_classes, names, imgsz)

    # Default fallback to PyTorch adapter
    return UNINAPyTorchAdapter(spec, device, num_classes, names, imgsz)
