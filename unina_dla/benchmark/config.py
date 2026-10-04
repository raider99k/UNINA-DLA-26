"""
UNINA-DLA Benchmark Configuration Module.

Defines configuration data structures and parsing utilities for multi-model
benchmarking across accuracy (intelligence), speed (latency/throughput), and footprint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml


def parse_resolution(imgsz: Union[str, int, List[int], Tuple[int, ...]]) -> Tuple[int, int]:
    """
    Parse image resolution into canonical (Height, Width) tuple.

    Supports:
        - int: 640 -> (640, 640)
        - str: "640" -> (640, 640)
        - str: "768,480" or "768x480" (W, H widescreen) -> (480, 768)
        - str: "480,768" or "480x768" (H, W) -> (480, 768)
        - list/tuple: [480, 768] (H, W) -> (480, 768)
        - list/tuple: [768, 480] (W, H widescreen) -> (480, 768)
    """
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return (int(imgsz[0]), int(imgsz[0]))
        elif len(imgsz) == 2:
            p0, p1 = int(imgsz[0]), int(imgsz[1])
            # In widescreen/landscape convention (W >= H), map to canonical (Height, Width)
            if p0 > p1:
                return (p1, p0)
            return (p0, p1)
        else:
            raise ValueError(f"Resolution tuple/list must have 1 or 2 elements, got {imgsz}")

    if isinstance(imgsz, int):
        return (imgsz, imgsz)

    if isinstance(imgsz, str):
        val = imgsz.strip().lower()
        delimiter = "," if "," in val else ("x" if "x" in val else None)
        if delimiter:
            parts = [int(p.strip()) for p in val.split(delimiter) if p.strip()]
            if len(parts) == 1:
                return (parts[0], parts[0])
            elif len(parts) == 2:
                p0, p1 = parts[0], parts[1]
                # In widescreen/landscape convention (W >= H), map to canonical (Height, Width)
                if p0 > p1:
                    return (p1, p0)
                return (p0, p1)
            else:
                raise ValueError(f"Invalid resolution format: {imgsz}")
        return (int(val), int(val))

    raise TypeError(f"Unsupported resolution type: {type(imgsz)}")


@dataclass
class ModelSpec:
    """Specification of a single candidate model to benchmark."""
    name: str
    path: str
    model_type: str = "auto"  # auto, unina_pth, unina_pt, yolo, onnx, tensorrt
    qat: bool = False
    nms: bool = False
    half: bool = False
    dla_core: Optional[int] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, spec_str: str) -> ModelSpec:
        """
        Parse model specification from CLI string.
        Examples:
            - 'runs/best_qat.pth'
            - 'UNINA-QAT=runs/best_qat.pth'
            - 'DLA0-Engine=runs/model.engine:dla=0'
        """
        name = None
        path = spec_str
        dla_core = None
        qat = False
        nms = False
        half = False

        if "=" in spec_str and not spec_str.startswith("="):
            parts = spec_str.split("=", 1)
            name = parts[0].strip()
            path = parts[1].strip()

        option_tokens = []
        has_win_drive = len(path) >= 2 and path[0].isalpha() and path[1] == ":"
        if has_win_drive:
            remainder = path[2:]
            if ":" in remainder:
                subparts = remainder.split(":")
                path = path[:2] + subparts[0]
                option_tokens = subparts[1:]
        elif ":" in path:
            subparts = path.split(":")
            path = subparts[0]
            option_tokens = subparts[1:]

        p = Path(path)
        if not name:
            name = p.stem

        model_type = "auto"
        for token in option_tokens:
            token = token.strip().lower()
            if token.startswith("dla="):
                dla_core = int(token.split("=")[1])
            elif token == "qat":
                qat = True
            elif token == "nms":
                nms = True
            elif token in ("half", "fp16"):
                half = True
            elif token in ("trt", "engine", "tensorrt"):
                model_type = "tensorrt"

        return cls(
            name=name,
            path=path,
            model_type=model_type,
            qat=qat,
            nms=nms,
            half=half,
            dla_core=dla_core,
        )


@dataclass
class BenchmarkConfig:
    """Master benchmark configuration."""
    models: List[ModelSpec] = field(default_factory=list)
    data: Optional[str] = None
    data_path: Optional[str] = None
    imgsz: Tuple[int, int] = (640, 640)  # (H, W)
    batch_size: int = 1
    device: str = "0"
    half: bool = False
    warmup: int = 50
    iterations: int = 200
    conf_thres: float = 0.25
    iou_thres: float = 0.6
    nms: bool = False
    speed_only: bool = False
    trt: bool = False
    max_det: int = 300
    num_workers: int = 4
    output_dir: Path = field(default_factory=lambda: Path("runs/benchmark"))
    save_md: bool = True
    save_json: bool = True
    save_csv: bool = True
    plot: bool = True
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> BenchmarkConfig:
        """Load benchmark configuration from a YAML file."""
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config file not found: {yaml_path}")

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        models = []
        raw_models = data.get("models", [])
        for m in raw_models:
            if isinstance(m, str):
                models.append(ModelSpec.from_string(m))
            elif isinstance(m, dict):
                models.append(ModelSpec(
                    name=m.get("name", Path(m.get("path", "model")).stem),
                    path=m.get("path", ""),
                    model_type=m.get("type", "auto"),
                    qat=m.get("qat", False),
                    nms=m.get("nms", False),
                    half=m.get("half", False),
                    dla_core=m.get("dla_core", None),
                    extra_args=m.get("extra_args", {})
                ))

        imgsz_raw = data.get("imgsz", (640, 640))
        imgsz = parse_resolution(imgsz_raw)

        return cls(
            models=models,
            data=data.get("data", data.get("dataset", None)),
            data_path=data.get("data_path", None),
            imgsz=imgsz,
            batch_size=int(data.get("batch_size", data.get("batch", 1))),
            device=str(data.get("device", "0")),
            half=bool(data.get("half", False)),
            warmup=int(data.get("warmup", 50)),
            iterations=int(data.get("iterations", data.get("trials", 200))),
            conf_thres=float(data.get("conf_thres", data.get("conf", 0.25))),
            iou_thres=float(data.get("iou_thres", data.get("iou", 0.6))),
            nms=bool(data.get("nms", False)),
            speed_only=bool(data.get("speed_only", False)),
            max_det=int(data.get("max_det", 300)),
            num_workers=int(data.get("num_workers", 4)),
            output_dir=Path(data.get("output_dir", "runs/benchmark")),
            save_md=bool(data.get("save_md", True)),
            save_json=bool(data.get("save_json", True)),
            save_csv=bool(data.get("save_csv", True)),
            plot=bool(data.get("plot", True)),
            verbose=bool(data.get("verbose", True)),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BenchmarkConfig:
        """Create configuration from parsed CLI arguments."""
        # If a config YAML is provided, start from that as base
        if getattr(args, "config", None) and Path(args.config).exists():
            base = cls.from_yaml(args.config)
        else:
            base = cls()

        # Override models if supplied via CLI
        models = []
        if getattr(args, "models", None):
            for m_str in args.models:
                models.append(ModelSpec.from_string(m_str))
        elif getattr(args, "model_dir", None):
            model_dir = Path(args.model_dir)
            pattern_filter = getattr(args, "filter", "*.pth,*.pt,*.onnx,*.engine")
            patterns = [p.strip() for p in pattern_filter.split(",")]
            for pat in patterns:
                for f in sorted(model_dir.glob(pat)):
                    models.append(ModelSpec.from_string(str(f)))

        if models:
            base.models = models

        if getattr(args, "data", None) is not None:
            base.data = args.data
        if getattr(args, "data_path", None) is not None:
            base.data_path = args.data_path
        if getattr(args, "imgsz", None) is not None:
            base.imgsz = parse_resolution(args.imgsz)
        if getattr(args, "batch", None) is not None:
            base.batch_size = args.batch
        if getattr(args, "device", None) is not None:
            base.device = str(args.device)
        if getattr(args, "half", None) is not None:
            base.half = args.half
        if getattr(args, "warmup", None) is not None:
            base.warmup = args.warmup
        if getattr(args, "iterations", None) is not None:
            base.iterations = args.iterations
        if getattr(args, "conf", None) is not None:
            base.conf_thres = args.conf
        if getattr(args, "iou", None) is not None:
            base.iou_thres = args.iou
        if getattr(args, "nms", None) is not None:
            base.nms = args.nms
        if getattr(args, "speed_only", None) is not None:
            base.speed_only = args.speed_only
        if getattr(args, "max_det", None) is not None:
            base.max_det = args.max_det
        if getattr(args, "output_dir", None) is not None:
            base.output_dir = Path(args.output_dir)
        if getattr(args, "plot", None) is not None:
            base.plot = args.plot
        if getattr(args, "quiet", None):
            base.verbose = False
        if getattr(args, "trt", None) is not None:
            base.trt = bool(args.trt)

        # In TRT mode, resolve candidate models to .engine files if present
        if getattr(base, "trt", False):
            for spec in base.models:
                p = Path(spec.path)
                if p.suffix.lower() not in (".engine", ".plan"):
                    candidates = [
                        p.with_suffix(".engine"),
                        p.parent / f"{p.stem}.engine",
                        p.parent / "export" / f"{p.stem}.engine",
                        Path("runs/export") / f"{p.stem}.engine",
                    ]
                    found = False
                    for cand in candidates:
                        if cand.exists():
                            spec.path = str(cand)
                            spec.model_type = "tensorrt"
                            print(f"[TensorRT Mode] Auto-resolved '{spec.name}' to compiled engine: {cand}")
                            found = True
                            break
                    if not found:
                        print(
                            f"[TensorRT Mode] Note: Precompiled engine not found for '{spec.name}' ({p.name}). "
                            f"Profiling will evaluate via native framework."
                        )

        return base
