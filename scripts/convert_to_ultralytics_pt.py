#!/usr/bin/env python3
"""Convert UNINA training checkpoint (.pth) to Ultralytics-loadable .pt."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from unina_dla.ultralytics_wrapper import convert_checkpoint_to_ultralytics_pt


def parse_names(value: str | None):
    if not value:
        return None
    p = Path(value)
    if p.exists() and p.suffix.lower() in {".yaml", ".yml"}:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "names" in data:
            return data["names"]
        return data
    return [x.strip() for x in value.split(",") if x.strip()]


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
    parser = argparse.ArgumentParser(description="UNINA checkpoint -> Ultralytics .pt converter")
    parser.add_argument("--checkpoint", required=True, help="Input UNINA checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--names", default=None, help="Class names CSV or dataset YAML path")
    parser.add_argument("--imgsz", default="640", help="Metadata imgsz: 640 or h,w")
    parser.add_argument("--conf", type=float, default=0.25, help="Default metadata confidence")
    parser.add_argument("--iou", type=float, default=0.6, help="Default metadata IoU")
    args = parser.parse_args()

    names = parse_names(args.names)
    imgsz = parse_imgsz(args.imgsz)

    out = convert_checkpoint_to_ultralytics_pt(
        checkpoint=args.checkpoint,
        output_pt=args.output,
        num_classes=args.num_classes,
        names=names,
        imgsz=imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    print(f"Saved Ultralytics-compatible UNINA .pt: {out}")


if __name__ == "__main__":
    main()
