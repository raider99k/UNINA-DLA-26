import argparse
import os
import sys

# Ensure project root is in Python path if running independently
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unina_dla.engines.phase3_export import run_export  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone UNINA-DLA Export Script")
    parser.set_defaults(verify_dla=True)
    parser.add_argument(
        "--checkpoint", type=str, default="unina-dla.pth", help="Path to .pth checkpoint"
    )
    parser.add_argument("--output", type=str, default=None, help="Output .onnx file path")

    # Export Flags
    parser.add_argument("--qat", action="store_true", help="Export QAT model")
    parser.add_argument(
        "--qat-clean-deploy",
        action="store_true",
        help="Export a clean deploy ONNX from a QAT checkpoint by stripping fake-quant/QDQ state",
    )
    parser.add_argument("--sparse", action="store_true", help="Model was trained with 2:4 sparsity")
    parser.add_argument(
        "--distilled", action="store_true", help="Model was trained with teacher distillation"
    )
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Image size: single int (640) or WxH (1024,640) for rectangular",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store ONNX weights in a separate .onnx.data file instead of embedding them in one .onnx",
    )
    parser.add_argument(
        "--verify-dla",
        action="store_true",
        dest="verify_dla",
        help="Run strict DLA ONNX verification after export (default: enabled)",
    )
    parser.add_argument(
        "--no-verify-dla",
        action="store_false",
        dest="verify_dla",
        help="Skip strict DLA ONNX verification after export",
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

    # Parse image size (supports both square and rectangular)
    imgsz_parts = [int(x.strip()) for x in args.imgsz.split(",")]
    if len(imgsz_parts) == 1:
        imgsz_hw = (imgsz_parts[0], imgsz_parts[0])  # Square
    else:
        imgsz_hw = (imgsz_parts[1], imgsz_parts[0])  # (H, W) (H=640, W=1024)
    args.imgsz_hw = imgsz_hw

    # Run Export
    run_export(args)
