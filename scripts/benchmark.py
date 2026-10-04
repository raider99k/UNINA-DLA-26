#!/usr/bin/env python3
"""
UNINA-DLA Benchmarking Suite CLI.

Benchmark multiple object detection models across:
- Intelligence (Accuracy): mAP@0.50, mAP@0.50:0.95, Precision, Recall, F1
- Speed (Latency & Throughput): Preprocess, Inference, Postprocess/Decode, End-to-End Latency, P95, and FPS
- Efficiency (Footprint): Parameters (M), GFLOPs, Disk Size (MB), Peak VRAM (MB)

Supports:
- UNINA-DLA PyTorch Checkpoints (.pth: FP32, Distilled, 2:4 Sparse, INT8 QAT)
- Ultralytics UNINA Converted Models (.pt)
- Standard Ultralytics YOLO & Teacher Baselines (.pt)
- ONNX Models (.onnx) via ONNX Runtime
- TensorRT Engines (.engine / .plan) targeting Jetson Orin DLA Core 0/1 or GPU

Usage Examples:
    # 1. Compare multiple model checkpoints on an NVIDIA Jetson / GPU:
    python scripts/benchmark.py \\
        --models runs/best_qat.pth runs/unina_dla.engine yolo26x.pt \\
        --data configs/formula_student.yaml \\
        --imgsz 608,960 --device 0

    # 2. Pure speed & latency profiling (no dataset required):
    python scripts/benchmark.py \\
        --models "UNINA-QAT=runs/best_qat.pth" "DLA-Engine=runs/unina_dla.engine" \\
        --speed-only --imgsz 640 --device 0 --iterations 300

    # 3. Using a benchmark configuration file:
    python scripts/benchmark.py --config configs/benchmark_fsd.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unina_dla.benchmark.config import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UNINA-DLA Multi-Model Benchmark Suite (Intelligence, Speed, Footprint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --models runs/best_qat.pth runs/unina_dla.engine --data configs/formula_student.yaml
  python scripts/benchmark.py --models "QAT=best_qat.pth" "Teacher=best.pt" --speed-only --device 0
  python scripts/benchmark.py --config configs/benchmark_fsd.yaml
        """,
    )

    # Model specification
    group_model = parser.add_argument_group("Model Selection")
    group_model.add_argument(
        "--models",
        nargs="+",
        help="One or more model paths or labeled specs (e.g. 'runs/best_qat.pth' or 'Name=path.engine')",
    )
    group_model.add_argument(
        "--config",
        type=str,
        help="Path to YAML benchmark suite configuration file",
    )
    group_model.add_argument(
        "--model-dir",
        type=str,
        help="Scan a directory for candidate models to benchmark",
    )
    group_model.add_argument(
        "--filter",
        type=str,
        default="*.pth,*.pt,*.onnx,*.engine",
        help="Comma-separated glob filters when using --model-dir",
    )

    # Dataset & task
    group_data = parser.add_argument_group("Dataset & Evaluation")
    group_data.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML config (e.g. configs/formula_student.yaml) for accuracy benchmarking",
    )
    group_data.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override dataset root directory path (e.g. /content/fsoco_dataset)",
    )
    group_data.add_argument(
        "--speed-only",
        action="store_true",
        help="Benchmark latency, throughput, and memory only (skips dataset evaluation)",
    )
    group_data.add_argument(
        "--imgsz",
        type=str,
        default=None,
        help="Input resolution: single int (640) or 'H,W' (608,960)",
    )
    group_data.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1 for edge latency profiling)",
    )
    group_data.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for inference postprocessing (default: 0.25)",
    )
    group_data.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS matching (default: 0.6)",
    )
    group_data.add_argument(
        "--nms",
        action="store_true",
        help="Force IoU NMS postprocessing instead of O2O Hungarian head (for A/B testing)",
    )
    group_data.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image (default: 300)",
    )

    # Hardware & performance profiling
    group_perf = parser.add_argument_group("Hardware & Profiling")
    group_perf.add_argument(
        "--device",
        type=str,
        default="0",
        help="Target device: '0', 'cuda:0', or 'cpu' (default: 0)",
    )
    group_perf.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 half precision for PyTorch models",
    )
    group_perf.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of unmeasured warm-up iterations (default: 50)",
    )
    group_perf.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of timed iterations for latency profiling (default: 200)",
    )
    group_perf.add_argument(
        "--trt",
        action="store_true",
        help="Target TensorRT engines for all candidate models (auto-resolves matching .engine files where available)",
    )

    # Output & reporting
    group_out = parser.add_argument_group("Reporting & Output")
    group_out.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmark",
        help="Output directory for reports and charts (default: runs/benchmark)",
    )
    group_out.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable chart generation (Pareto curve, latency breakdown)",
    )
    group_out.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress information",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Validation: Need either models, model-dir, or config
    if not args.models and not args.model_dir and not args.config:
        parser.print_help()
        print("\n[ERROR] You must specify candidate models via --models, --model-dir, or --config.")
        return 1

    config = BenchmarkConfig.from_args(args)
    if args.no_plot:
        config.plot = False

    try:
        from unina_dla.benchmark import run_benchmark
        run_benchmark(config)
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
