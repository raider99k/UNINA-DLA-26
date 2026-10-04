#!/usr/bin/env python3
"""Run UNINA inference through Ultralytics-style API for .pt/.engine models."""

from __future__ import annotations

import argparse
from pathlib import Path

from unina_dla.ultralytics_wrapper import UNINAYOLO


def parse_imgsz(value: str) -> int | tuple[int, int]:
    value = value.strip().lower()
    delimiter = "," if "," in value else ("x" if "x" in value else None)
    if delimiter:
        parts = [int(x.strip()) for x in value.split(delimiter) if x.strip()]
        if len(parts) != 2:
            raise ValueError("--imgsz must be an int (640) or 'w,h' (768,480).")
        if parts[0] > parts[1]:
            return (parts[1], parts[0])
        return (parts[0], parts[1])
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="UNINA Ultralytics Inference")
    parser.add_argument("--model", required=True, help="Path to .pt or .engine")
    parser.add_argument("--source", required=True, help="Image/video/folder/webcam source")
    parser.add_argument("--device", default="0", help="Device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", default="640", help="Inference size: 640 or h,w")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold (used if nms=True)")
    parser.add_argument(
        "--nms", action="store_true", help="Enable IoU NMS instead of O2O filtering"
    )
    parser.add_argument("--save", action="store_true", help="Save rendered predictions")
    parser.add_argument("--show", action="store_true", help="Show rendered predictions")
    parser.add_argument("--project", default=None, help="Optional output project dir")
    parser.add_argument("--name", default=None, help="Optional output run name")
    args = parser.parse_args()

    imgsz = parse_imgsz(args.imgsz)
    model = UNINAYOLO(args.model)

    predict_kwargs = {
        "source": args.source,
        "device": args.device,
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "nms": args.nms,
        "save": args.save,
        "show": args.show,
    }
    if args.project:
        predict_kwargs["project"] = args.project
    if args.name:
        predict_kwargs["name"] = args.name

    results = model.predict(**predict_kwargs)

    print(f"Model: {Path(args.model).resolve()}")
    print(f"Source: {args.source}")
    print(f"Images: {len(results)}")
    for i, r in enumerate(results):
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"  [{i}] detections={n} path={r.path}")


if __name__ == "__main__":
    main()
