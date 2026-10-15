import argparse
import json
import os
import random
import subprocess

import numpy as np
import torch
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version, detect_upsample_mode
from unina_dla.utils.validator import validate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="unina_dla/config/unina_dla_data.yaml")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Treat checkpoint as QAT model (auto-detected if keys match)",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Image size: single int (640) or WxH (1024,640) for rectangular",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        choices=["2.6", "2.5", "3.1", "2", "3"],
        help="Architecture generation (default: auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--upsample_mode",
        type=str,
        default=None,
        choices=["nearest", "deconv", "bilinear"],
        help="Upsampling mode in Rep-PAN neck (default: auto-detected from checkpoint)",
    )
    args = parser.parse_args()

    # Set deterministic seed
    set_seed(42)  # Fixed seed for deterministic validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and validate dataset configuration
    data_cfg = check_det_dataset(args.data)
    nc = data_cfg.get("nc", 4)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        exit(1)

    print(f"Loading checkpoint {args.checkpoint}...")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
    up_mode = getattr(args, "upsample_mode", None) or detect_upsample_mode(state_dict)
    is_fused = any("rbr_reparam" in k for k in state_dict.keys())
    is_qat_ckpt = args.qat

    # Auto-detect QAT if not flagged
    if not is_qat_ckpt:
        for k in state_dict.keys():
            if "_amax" in k or "quant" in k:
                print("Auto-detected QAT checkpoint. preparing model...")
                is_qat_ckpt = True
                break

    model = UNINA_DLA(num_classes=nc, deploy=is_fused, version=arch_version, upsample_mode=up_mode).to(device)
    print(f"Initialized UNINA_DLA (nc={nc}, Gen-{arch_version}, upsample={up_mode}, deploy={is_fused}).")

    if is_qat_ckpt:
        try:
            from unina_dla.utils.qat_utils import enable_quantization, prepare_qat_model

            model = prepare_qat_model(model)
            print("Model prepared for QAT Validation.")
        except ImportError:
            print("Error: QAT utils not found but QAT checkpoint detected.")
            exit(1)

    model.load_state_dict(state_dict, strict=False)

    if is_qat_ckpt:
        enable_quantization(model)

    # Build validation dataset and dataloader (matching train_student.py pattern)
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)

    # Parse image size (supports both square and rectangular)
    imgsz_parts = [int(x.strip()) for x in args.imgsz.split(",")]
    if len(imgsz_parts) == 1:
        imgsz_hw = (imgsz_parts[0], imgsz_parts[0])
    else:
        imgsz_hw = (imgsz_parts[1], imgsz_parts[0])  # (H, W)

    # 1. Load defaults from data config
    for k, v in data_cfg.items():
        if k in dataset_cfg.__dict__:
            setattr(dataset_cfg, k, v)

    # 2. Overwrite with CLI argument/calculated size
    # Use max dimension for Ultralytics dataloader compatibility
    dataset_cfg.imgsz = max(imgsz_hw)
    print(
        f"Validation resolution: {imgsz_hw[1]}x{imgsz_hw[0]} (using imgsz={dataset_cfg.imgsz} for dataloader)"
    )

    val_set = build_yolo_dataset(
        dataset_cfg, data_cfg["val"], args.batch, data_cfg, mode="val", rect=False
    )
    val_loader = build_dataloader(val_set, batch=args.batch, workers=4, shuffle=False)

    results, _ = validate(
        model, val_loader, device, num_classes=data_cfg.get("nc", 4), names=data_cfg.get("names")
    )
    print(f"Validation Results: mAP50: {results['mAP50']:.4f}, mAP50-95: {results['mAP50-95']:.4f}")

    # Generate Industry-Standard Validation Report
    report = {
        "metrics": {k: float(v) for k, v in results.items()},  # Ensure float for JSON serialization
        "checkpoint": args.checkpoint,
        "qat_mode": is_qat_ckpt,
        "batch_size": args.batch,
        "dataset": args.data,
    }

    # Add Git Hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        report["git_hash"] = git_hash
    except Exception:
        report["git_hash"] = "unknown"

    # Save to JSON
    report_file = f"val_results_{os.path.basename(args.checkpoint)}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Validation results saved to {report_file}")
