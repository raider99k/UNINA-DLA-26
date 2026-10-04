"""
Ultralytics compatibility wrappers for UNINA-DLA inference.

This module provides:
- A custom Ultralytics predictor that decodes UNINA 6-output tensors.
- A Model wrapper that plugs the predictor into Ultralytics APIs.
- Helpers to convert UNINA training checkpoints to Ultralytics-loadable .pt files.
- Helpers to embed metadata into TensorRT .engine files for AutoBackend.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from ultralytics import __version__ as ULTRALYTICS_VERSION
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import nms, ops

from unina_dla.model.unina_dla import UNINA_DLA


def _coerce_names(names: dict[int, str] | Iterable[str] | None, num_classes: int) -> dict[int, str]:
    if names is None:
        return {i: str(i) for i in range(num_classes)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


class UNINADetectionPredictor(DetectionPredictor):
    """Detection predictor that understands UNINA 6-output (reg*/cls*) tensors."""

    @staticmethod
    def _looks_like_unina_outputs(preds: Any) -> bool:
        if not isinstance(preds, (list, tuple)):
            return False
        if len(preds) < 2 or len(preds) % 2 != 0:
            return False
        return all(torch.is_tensor(p) and p.ndim == 4 for p in preds)

    def _ordered_output_names(self, n_outputs: int) -> list[str] | None:
        names = getattr(self.model, "output_names", None)
        if not names or len(names) != n_outputs:
            return None
        names = list(names)
        if getattr(self.model, "engine", False):
            # AutoBackend sorts engine outputs at runtime.
            return sorted(names)
        return names

    def _split_outputs(
        self, preds: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        n_outputs = len(preds)
        n_levels = n_outputs // 2

        names = self._ordered_output_names(n_outputs)
        if names is not None:
            reg_map: dict[int, torch.Tensor] = {}
            cls_map: dict[int, torch.Tensor] = {}
            reg_fallback = 0
            cls_fallback = 0

            for name, tensor in zip(names, preds):
                lname = name.lower()
                m = re.search(r"(\d+)$", lname)
                level_id = int(m.group(1)) if m else -1

                if lname.startswith("reg"):
                    if level_id < 0:
                        level_id = reg_fallback
                        reg_fallback += 1
                    reg_map[level_id] = tensor
                elif lname.startswith("cls"):
                    if level_id < 0:
                        level_id = cls_fallback
                        cls_fallback += 1
                    cls_map[level_id] = tensor

            if len(reg_map) == n_levels and len(cls_map) == n_levels:
                shared = sorted(set(reg_map) & set(cls_map))
                if len(shared) == n_levels:
                    reg_levels = [reg_map[i] for i in shared]
                    cls_levels = [cls_map[i] for i in shared]
                    return reg_levels, cls_levels

        # Fallback: exported UNINA .pt path keeps reg tensors first, cls second.
        reg_levels = list(preds[:n_levels])
        cls_levels = list(preds[n_levels:])
        return reg_levels, cls_levels

    @staticmethod
    def _decode_ltrb(
        reg: torch.Tensor, cls: torch.Tensor, image_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one feature level.

        reg: [B, 4, H, W] (LTRB distances in grid units)
        cls: [B, C, H, W] (class probabilities or logits)
        image_shape: (H_img, W_img)
        """
        bs, reg_c, h, w = reg.shape
        _, cls_c, h2, w2 = cls.shape
        if reg_c != 4:
            raise ValueError(f"Expected 4 regression channels, got {reg_c}.")
        if h != h2 or w != w2:
            raise ValueError(
                "Regression and classification feature maps have mismatched spatial sizes."
            )

        reg_flat = reg.permute(0, 2, 3, 1).reshape(bs, -1, 4)
        cls_flat = cls.permute(0, 2, 3, 1).reshape(bs, -1, cls_c)

        # Exported UNINA deploy models already apply sigmoid to cls; handle raw logits too.
        if (cls_flat.max() > 1.0) or (cls_flat.min() < 0.0):
            cls_flat = cls_flat.sigmoid()

        img_h, img_w = image_shape
        stride_h = float(img_h) / float(h)
        stride_w = float(img_w) / float(w)
        stride = (stride_h + stride_w) * 0.5

        sy, sx = torch.meshgrid(
            torch.arange(h, device=reg.device, dtype=reg.dtype) + 0.5,
            torch.arange(w, device=reg.device, dtype=reg.dtype) + 0.5,
            indexing="ij",
        )
        anchors = torch.stack((sx, sy), dim=-1).reshape(-1, 2) * stride  # [HW, 2]

        ltrb = reg_flat * stride
        lt, rb = torch.split(ltrb, 2, dim=-1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        boxes = torch.cat((x1y1, x2y2), dim=-1)
        return boxes, cls_flat

    @staticmethod
    def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
        xywh = boxes.clone()
        xywh[..., 0] = (boxes[..., 0] + boxes[..., 2]) * 0.5
        xywh[..., 1] = (boxes[..., 1] + boxes[..., 3]) * 0.5
        xywh[..., 2] = boxes[..., 2] - boxes[..., 0]
        xywh[..., 3] = boxes[..., 3] - boxes[..., 1]
        return xywh

    def _decode_all_levels(
        self, preds: list[torch.Tensor] | tuple[torch.Tensor, ...], img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reg_levels, cls_levels = self._split_outputs(preds)
        if len(reg_levels) != len(cls_levels):
            raise ValueError("UNINA reg/cls level count mismatch.")

        all_boxes = []
        all_scores = []
        image_shape = (int(img.shape[2]), int(img.shape[3]))
        for reg, cls in zip(reg_levels, cls_levels):
            boxes, scores = self._decode_ltrb(reg, cls, image_shape=image_shape)
            all_boxes.append(boxes)
            all_scores.append(scores)

        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)

    def _apply_o2o_filter(self, boxes: torch.Tensor, scores: torch.Tensor) -> list[torch.Tensor]:
        conf_thres = float(self.args.conf)
        max_det = max(1, int(self.args.max_det))
        classes = getattr(self.args, "classes", None)

        output = []
        for b, s in zip(boxes, scores):
            conf, cls_idx = s.max(dim=-1, keepdim=True)
            keep = conf[:, 0] > conf_thres
            if keep.any():
                det = torch.cat((b[keep], conf[keep], cls_idx[keep].float()), dim=-1)
                if classes is not None:
                    class_tensor = torch.as_tensor(classes, device=det.device)
                    det = det[(det[:, 5:6] == class_tensor.view(1, -1)).any(1)]
                if det.shape[0] > max_det:
                    topk = det[:, 4].argsort(descending=True)[:max_det]
                    det = det[topk]
                output.append(det)
            else:
                output.append(torch.zeros((0, 6), device=b.device, dtype=b.dtype))
        return output

    def _apply_forced_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> list[torch.Tensor]:
        pred_xywh = self._xyxy_to_xywh(boxes)
        pred_bn = torch.cat((pred_xywh, scores), dim=-1).permute(0, 2, 1).contiguous()
        return nms.non_max_suppression(
            pred_bn,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0,
            end2end=False,
        )

    def postprocess(self, preds, img, orig_imgs, **kwargs):  # noqa: D401
        if not self._looks_like_unina_outputs(preds):
            return super().postprocess(preds, img, orig_imgs, **kwargs)

        boxes, scores = self._decode_all_levels(preds, img=img)
        if bool(getattr(self.args, "nms", False)):
            dets = self._apply_forced_nms(boxes, scores)
        else:
            dets = self._apply_o2o_filter(boxes, scores)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(dets, img, orig_imgs)


class UNINAYOLO(Model):
    """Ultralytics Model wrapper that swaps in UNINA-aware detection postprocess."""

    def __init__(
        self,
        model: str | Path = "unina_dla.engine",
        task: str | None = "detect",
        verbose: bool = False,
    ):
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": UNINADetectionPredictor,
            }
        }


class UNINAUltralyticsModule(torch.nn.Module):
    """
    Lightweight wrapper that gives UNINA_DLA the attributes expected by AutoBackend.

    AutoBackend assumes in-memory modules expose methods/attrs like:
    - `fuse(verbose=...)`
    - `yaml` with `channels`
    - `names`, `stride`, `task`, `end2end`, `args`
    """

    def __init__(self, core: UNINA_DLA, names: dict[int, str]) -> None:
        super().__init__()
        self.core = core
        self.names = names
        self.stride = getattr(core, "stride", torch.tensor([8.0, 16.0, 32.0]))
        self.task = "detect"
        self.end2end = True
        self.yaml = {"channels": 3}
        self.args = {"task": "detect", "imgsz": 640, "batch": 1, "nms": False}

    def forward(self, x, *args, **kwargs):
        return self.core(x)

    def fuse(self, verbose: bool = True):
        # Core is already switched to deploy/fused structure before wrapping.
        return self


def build_unina_deploy_model(
    checkpoint: str | Path,
    num_classes: int = 4,
    names: dict[int, str] | Iterable[str] | None = None,
) -> tuple[UNINAUltralyticsModule, list[str], list[str]]:
    """Load UNINA training checkpoint (.pth) into deploy-mode UNINA_DLA module."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = UNINA_DLA(num_classes=num_classes, deploy=False)
    model.eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.switch_to_deploy()
    model.eval()

    names_map = _coerce_names(names, num_classes=num_classes)
    model.names = names_map
    model.nc = num_classes
    model.task = "detect"
    model.end2end = True
    if hasattr(model, "head") and hasattr(model.head, "stride"):
        stride = model.head.stride
        model.stride = (
            stride.float() if torch.is_tensor(stride) else torch.tensor(stride, dtype=torch.float32)
        )
    else:
        model.stride = torch.tensor([8.0, 16.0, 32.0])

    wrapped = UNINAUltralyticsModule(model, names=names_map)
    wrapped.args = {
        "task": "detect",
        "imgsz": 640,
        "batch": 1,
        "nms": False,
    }
    return wrapped, list(missing), list(unexpected)


def convert_checkpoint_to_ultralytics_pt(
    checkpoint: str | Path,
    output_pt: str | Path,
    num_classes: int = 4,
    names: dict[int, str] | Iterable[str] | None = None,
    imgsz: int | tuple[int, int] = 640,
    conf: float = 0.25,
    iou: float = 0.6,
) -> Path:
    """
    Convert UNINA .pth (state_dict checkpoint) to Ultralytics-loadable .pt.

    The produced file is intended to be loaded with UNINAYOLO, not stock YOLO.
    """
    model, missing, unexpected = build_unina_deploy_model(
        checkpoint=checkpoint,
        num_classes=num_classes,
        names=names,
    )

    if isinstance(imgsz, int):
        imgsz_meta = (imgsz, imgsz)
    else:
        imgsz_meta = tuple(int(x) for x in imgsz)

    train_args = {
        "task": "detect",
        "imgsz": imgsz_meta,
        "conf": float(conf),
        "iou": float(iou),
        "nms": False,
        "batch": 1,
    }

    payload = {
        "model": model,
        "ema": None,
        "train_args": train_args,
        "date": datetime.now().isoformat(),
        "version": ULTRALYTICS_VERSION,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "unina_conversion": {
            "source_checkpoint": str(checkpoint),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
    }

    out = Path(output_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def build_ultralytics_metadata(
    names: dict[int, str] | Iterable[str],
    imgsz: tuple[int, int] = (640, 640),
    stride: int = 32,
    end2end: bool = True,
    dynamic: bool = False,
    batch: int = 1,
    channels: int = 3,
    nms_enabled: bool = False,
    dla: int | None = None,
) -> dict[str, Any]:
    """Build metadata dictionary compatible with Ultralytics AutoBackend."""
    names_map = _coerce_names(
        names, num_classes=len(list(names)) if not isinstance(names, dict) else len(names)
    )
    meta = {
        "description": "UNINA-DLA custom model",
        "author": "UNINA-DLA",
        "date": datetime.now().isoformat(),
        "version": ULTRALYTICS_VERSION,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": int(stride),
        "task": "detect",
        "batch": int(batch),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "names": names_map,
        "channels": int(channels),
        "end2end": bool(end2end),
        "args": {
            "dynamic": bool(dynamic),
            "nms": bool(nms_enabled),
        },
    }
    if dla is not None:
        meta["dla"] = int(dla)
    return meta


def _strip_existing_engine_metadata(raw: bytes) -> bytes:
    """If engine already has Ultralytics metadata prefix, strip it."""
    if len(raw) < 8:
        return raw
    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
    if meta_len <= 0 or (4 + meta_len) >= len(raw):
        return raw
    try:
        json.loads(raw[4 : 4 + meta_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    return raw[4 + meta_len :]


def embed_metadata_in_engine(
    engine_path: str | Path,
    metadata: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Prepend Ultralytics metadata header to TensorRT .engine file.

    This is required for engines built with plain trtexec if you want AutoBackend
    to recover task/names/end2end metadata.
    """
    engine_path = Path(engine_path)
    out = Path(output_path) if output_path is not None else engine_path

    raw = engine_path.read_bytes()
    raw_engine = _strip_existing_engine_metadata(raw)

    meta_json = json.dumps(metadata)
    payload = (
        len(meta_json).to_bytes(4, byteorder="little", signed=True)
        + meta_json.encode("utf-8")
        + raw_engine
    )
    out.write_bytes(payload)
    return out
