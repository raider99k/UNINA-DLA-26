"""
DLA Utilities for UNINA-DLA-26
==============================
Helper tools for DLA deployment and verification.
"""

import argparse
import os
import subprocess


def generate_calibration_cache(
    onnx_path, output_cache, trtexec_path="trtexec", imgsz_hw=(640, 640)
):
    """
    Run trtexec to generate a calibration cache (PTQ Cleanup).

    Why: QAT models sometimes have scale factors that the DLA compiler dislikes.
         Running a quick PTQ pass generates a 'calib.cache' file that serves as a
         sanitized fallback for scaling factors.

    Args:
        onnx_path: Path to input ONNX model (QAT or FP32).
        output_cache: Path to save the calibration cache (e.g., 'calib.cache').
        trtexec_path: Path to trtexec binary.
    """
    print(f"Generating Calibration Cache for {onnx_path}...")

    # Check inputs
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        "--int8",
        "--fp16",  # Allow mixed precision
        f"--calib={output_cache}",
        "--buildOnly",  # Only execute the calibration step
        # Fixed input shape for calibration profile
        f"--shapes=images:1x3x{imgsz_hw[0]}x{imgsz_hw[1]}",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"SUCCESS: Calibration cache saved to {output_cache}")
        print("Tip: Use this cache when building your engine: --calib=calib.cache")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: trtexec failed with error code {e.returncode}")
        print("Ensure 'trtexec' is in your PATH or provide full path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLA Utilities")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="calib.cache", help="Output cache file")
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Image size: single int (640) or WxH (1024,640) for rectangular",
    )

    args = parser.parse_args()

    # Parse image size (supports both square and rectangular)
    imgsz_parts = [int(x.strip()) for x in args.imgsz.split(",")]
    if len(imgsz_parts) == 1:
        imgsz_hw = (imgsz_parts[0], imgsz_parts[0])
    else:
        imgsz_hw = (imgsz_parts[1], imgsz_parts[0])  # (H, W)

    generate_calibration_cache(args.onnx, args.output, imgsz_hw=imgsz_hw)
