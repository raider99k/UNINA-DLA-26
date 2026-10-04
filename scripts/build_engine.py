#!/usr/bin/env python3
"""
UNINA-DLA Engine Builder
========================
Flexible wrapper around `trtexec` to compile ONNX models into TensorRT engines for GPU or DLA.
Handling:
- Device Targeting (GPU vs DLA)
- Precision (INT8, FP16)
- Sparsity (2:4 Structured) - ENABLED by default.
- DLA-Specific Constraints (Strict Mode, Input Formats)

Usage:
    python scripts/build_engine.py --onnx model.onnx --target dla
    python scripts/build_engine.py --onnx model.onnx --target gpu
    python scripts/build_engine.py --onnx model.onnx --target gpu --no-sparse
"""

import argparse
import os
import subprocess
import sys

# Ensure project root is importable when running this file directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _parse_names_arg(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None

    if os.path.exists(value) and value.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to parse --names from YAML. Install with: pip install pyyaml"
            ) from exc

        with open(value, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "names" in data:
            return data["names"]
        return data

    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_imgsz_arg(value):
    value = str(value).strip().lower()
    delimiter = "," if "," in value else ("x" if "x" in value else None)
    if delimiter:
        parts = [int(x.strip()) for x in value.split(delimiter) if x.strip()]
        if len(parts) != 2:
            raise ValueError("--meta-imgsz must be an int (640) or 'w,h' (768,480).")
        # In widescreen convention (W >= H), return canonical (Height, Width)
        if parts[0] > parts[1]:
            return (parts[1], parts[0])
        return (parts[0], parts[1])
    side = int(value)
    return (side, side)


def _maybe_embed_ultralytics_metadata(engine_path, args):
    if not args.embed_metadata:
        return 0

    try:
        from unina_dla.ultralytics_wrapper import (
            build_ultralytics_metadata,
            embed_metadata_in_engine,
        )
    except Exception as e:
        print(f"\n[Error] --embed-metadata requested, but wrapper import failed: {e}")
        return 2

    if not args.names:
        print("\n[Error] --embed-metadata requires --names (CSV or dataset YAML path).")
        return 2

    try:
        names = _parse_names_arg(args.names)
        if not names:
            raise ValueError("parsed names are empty")
        imgsz = _parse_imgsz_arg(args.meta_imgsz)
        dla_meta = (
            args.meta_dla
            if args.meta_dla is not None
            else (args.dla_core if args.target == "dla" else None)
        )
        metadata = build_ultralytics_metadata(
            names=names,
            imgsz=imgsz,
            stride=args.meta_stride,
            end2end=not args.meta_no_end2end,
            dynamic=args.meta_dynamic,
            batch=args.meta_batch,
            channels=3,
            nms_enabled=args.meta_nms,
            dla=dla_meta,
        )
        out = embed_metadata_in_engine(engine_path, metadata, output_path=args.meta_output)
        print(f"[Success] Ultralytics metadata embedded in engine: {out}")
        return 0
    except Exception as e:
        print(f"\n[Error] Failed to embed Ultralytics metadata: {e}")
        return 2


def _build_engine_via_python_api(
    onnx_path,
    output_path,
    enable_fp16=True,
    enable_int8=False,
    enable_sparsity=False,
    target="gpu",
    dla_core=0,
    gpu_fallback=False,
):
    """Fallback engine builder using the TensorRT Python API when trtexec is not available."""
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("-1", ""):
        del os.environ["CUDA_VISIBLE_DEVICES"]

    try:
        import tensorrt as trt
    except ImportError:
        print("\n[Error] Neither 'trtexec' binary nor 'tensorrt' Python module found.")
        print("Please ensure TensorRT is installed (`pip install tensorrt`).")
        return 1

    print("\n[Info] 'trtexec' binary not found. Falling back to TensorRT Python API builder...")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = 4 * 1024 * 1024 * 1024

    if target == "dla":
        if hasattr(trt, "DeviceType") and hasattr(config, "default_device_type"):
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            if gpu_fallback and hasattr(trt.BuilderFlag, "GPU_FALLBACK"):
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    if enable_fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    if enable_int8 and hasattr(trt.BuilderFlag, "INT8"):
        config.set_flag(trt.BuilderFlag.INT8)
    if enable_sparsity and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(
        trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"
    ):
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    else:
        network = builder.create_network(0)

    parser = trt.OnnxParser(network, logger)

    print(f"[TRT Python API] Parsing ONNX model: {onnx_path}...")
    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = parser.parse_from_file(onnx_path)
    if not parsed:
        with open(onnx_path, "rb") as f:
            parsed = parser.parse(f.read())

    if not parsed:
        for error_idx in range(parser.num_errors):
            print(f"[TRT Parser Error] {parser.get_error(error_idx)}")
        return 1

    print(f"[TRT Python API] Building and serializing engine -> {output_path}...")
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("[TRT Python API Error] Failed to build serialized network.")
            return 1
        with open(output_path, "wb") as f:
            f.write(plan)
    else:
        engine = builder.build_cuda_engine(network)
        if engine is None:
            print("[TRT Python API Error] Failed to build CUDA engine.")
            return 1
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

    print(f"\n[Success] Engine built and saved to: {output_path}")
    return 0


def build_engine(args):
    """Constructs and executes the trtexec command based on arguments."""

    # 1. Validate Input
    if not os.path.exists(args.onnx):
        print(f"[Error] ONNX file not found: {args.onnx}")
        return 1

    # 2. Determine Output Filename
    # Pattern: {name}_{target}.engine
    base_name = os.path.splitext(os.path.basename(args.onnx))[0]
    output_dir = os.path.dirname(args.onnx)

    if getattr(args, "output", None):
        output_path = args.output
    elif args.target == "dla":
        engine_name = f"{base_name}_{args.target}{args.dla_core}.engine"
        output_path = os.path.join(output_dir, engine_name)
    else:
        engine_name = f"{base_name}_{args.target}.engine"
        output_path = os.path.join(output_dir, engine_name)

    # Standardize precision flags
    # User requirement: "int8 and fp16 are both active because the model will run mixed"
    # We default to both being True unless explicitly disabled.
    enable_int8 = not args.no_int8
    enable_fp16 = not args.no_fp16

    # Sparsity Logic (Default ON)
    enable_sparsity = not args.no_sparse

    print("\n" + "=" * 50)
    print("UNINA-DLA Engine Builder")
    print("=" * 50)
    print(f"Target:   {args.target.upper()}")
    print(f"Input:    {args.onnx}")
    print(f"Output:   {output_path}")
    print(f"Precision: INT8={enable_int8}, FP16={enable_fp16}")
    print(f"Sparsity: {enable_sparsity}")

    # 3. Construct Command
    import shutil
    trtexec_bin = shutil.which("trtexec")
    if not trtexec_bin:
        for cand in [
            "/usr/src/tensorrt/bin/trtexec",
            "/usr/local/tensorrt/bin/trtexec",
            "/usr/bin/trtexec",
        ]:
            if os.path.exists(cand):
                trtexec_bin = cand
                break

    if not trtexec_bin:
        ret = _build_engine_via_python_api(
            args.onnx,
            output_path,
            enable_fp16=enable_fp16,
            enable_int8=enable_int8,
            enable_sparsity=enable_sparsity,
            target=args.target,
            dla_core=args.dla_core,
            gpu_fallback=args.gpu_fallback,
        )
        if ret == 0:
            meta_rc = _maybe_embed_ultralytics_metadata(output_path, args)
            return meta_rc
        return ret

    cmd = [trtexec_bin]

    # Common Flags
    cmd.append(f"--onnx={args.onnx}")
    cmd.append(f"--saveEngine={output_path}")

    # Precision
    if enable_fp16:
        cmd.append("--fp16")
    if enable_int8:
        cmd.append("--int8")

    # Use --best if both precisions are enabled (Default) for maximum tactic auto-tuning
    if enable_fp16 and enable_int8:
        cmd.append("--best")

    # Sparsity
    if enable_sparsity:
        # Check if trtexec version supports it (User is on Orin/JP6, so yes)
        cmd.append("--sparsity=enable")

    # Profiling
    if args.verbose:
        cmd.append("--profilingVerbosity=detailed")
        cmd.append("--exportLayerInfo=layer_info.json")  # Helpful for debugging DLA placement
    else:
        # Default verbosity is usually sufficient, maybe add --verbose if requested
        pass

    # 4. Device Logic
    if args.target == "dla":
        # DLA Specifics
        cmd.append(f"--useDLACore={args.dla_core}")

        # Memory / Performance
        # TensorRT 10.x exposes DLA memory tuning through --memPoolSize.
        cmd.append("--noDataTransfers")  # Important for zero-copy latency optimization

        # Input Format (DLA HWC4 optimization)
        # Using the user-provided format or defaults from README
        if args.input_format:
            cmd.append(f"--inputIOFormats={args.input_format}")
        else:
            # Default recommended for DLA: fp16:dla_hwc4
            # This avoids GPU reformatting kernels
            cmd.append("--inputIOFormats=fp16:dla_hwc4")
            print("[Tip] Using DLA-optimized input format (fp16:dla_hwc4).")
            print("      - Ideal for ZED X deployment.")
            print(
                "      - If testing offline (jpg/mp4), use --input_format=fp16:chw to avoid mismatches."
            )

        # Default to native DLA output storage. Override with fp16:chw for
        # legacy packed-CHW consumers.
        if args.output_format:
            cmd.append(f"--outputIOFormats={args.output_format}")
        else:
            cmd.append("--outputIOFormats=fp16:chw16")
            print("[Tip] Using native DLA output format (fp16:chw16).")
            print("      - Best for pure DLA output residency.")
            print("      - Use --output_format=fp16:chw for legacy packed-CHW consumers.")

        # Fallback Strictness
        if args.gpu_fallback:
            cmd.append("--allowGPUFallback")
            print("[Config] GPU Fallback ALLOWED (High Latency Risk!)")
        else:
            # Strict mode (implied by absence of --allowGPUFallback)
            print("[Config] DLA Strict Mode: ON (Fail on unsupported layers)")

    elif args.target == "gpu":
        # GPU Specifics
        # --best is now handled globally if precisions are enabled
        # We generally stick to standard layout (fp16:chw) for GPU unless specified
        if args.input_format:
            cmd.append(f"--inputIOFormats={args.input_format}")

    # Optional memory pool limits (TensorRT 10.x+, including DLA pools)
    if args.mem_pool_size:
        cmd.append(f"--memPoolSize={args.mem_pool_size}")

    # Calibration Cache (for QAT backup if needed)
    if args.calib and os.path.exists(args.calib):
        cmd.append(f"--calib={args.calib}")

    # 5. Execute
    # Print clean command for user verification
    print("\n[Executing Command]:")
    print(" ".join(cmd))
    print("-" * 50)

    if args.dry_run:
        print("[Dry Run] Execution skipped.")
        return 0

    try:
        # Stream output
        # Using Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"\n[Success] Engine saved to: {output_path}")
            meta_rc = _maybe_embed_ultralytics_metadata(output_path, args)
            if meta_rc != 0:
                return meta_rc
        else:
            print(f"\n[Failure] trtexec failed with return code {process.returncode}")

        return process.returncode

    except FileNotFoundError:
        ret = _build_engine_via_python_api(
            args.onnx,
            output_path,
            enable_fp16=enable_fp16,
            enable_int8=enable_int8,
            enable_sparsity=enable_sparsity,
            target=args.target,
            dla_core=args.dla_core,
            gpu_fallback=args.gpu_fallback,
        )
        if ret == 0:
            meta_rc = _maybe_embed_ultralytics_metadata(output_path, args)
            return meta_rc
        return ret
    except KeyboardInterrupt:
        print("\n[Aborted] Compilation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNINA-DLA Flexible Engine Builder")

    # Inputs
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model")
    parser.add_argument(
        "--output", type=str, default=None, help="Explicit output path for the .engine file"
    )

    # Target Selection
    parser.add_argument(
        "--target", choices=["gpu", "dla"], default="gpu", help="Target device (default: gpu)"
    )

    # DLA Specifics
    parser.add_argument("--dla_core", type=int, default=0, help="DLA Core Index (0 or 1)")
    parser.add_argument(
        "--gpu_fallback",
        action="store_true",
        help="Allow GPU fallback for DLA layers (WARNING: High latency)",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default=None,
        help="Override default input format (Default DLA: fp16:dla_hwc4)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default=None,
        help="Override output format (Default DLA: fp16:chw16, use fp16:chw for legacy consumers)",
    )

    # Model/Optimization Properties
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable 2:4 Structured Sparsity support (Default: Sparsity ENABLED)",
    )
    parser.add_argument("--no_int8", action="store_true", help="Disable INT8 precision")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 precision")
    parser.add_argument("--calib", type=str, default=None, help="Path to INT8 calibration cache")
    parser.add_argument(
        "--mem_pool_size",
        type=str,
        default=None,
        help="Optional trtexec --memPoolSize spec, e.g. 'workspace:1024' or 'dlaSRAM:1M,dlaLocalDRAM:1M,dlaGlobalDRAM:512M'",
    )

    # Execution Flags
    parser.add_argument("--verbose", action="store_true", help="Enable verbose profiling info")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")

    # Optional: Ultralytics metadata embedding for wrapper compatibility
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        help="Embed Ultralytics metadata header into output .engine after successful build",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Class names CSV or dataset YAML path (required with --embed-metadata)",
    )
    parser.add_argument(
        "--meta-imgsz",
        type=str,
        default="640",
        help="Metadata image size: 640 or h,w (e.g. 768,480)",
    )
    parser.add_argument("--meta-stride", type=int, default=32, help="Metadata stride value")
    parser.add_argument("--meta-batch", type=int, default=1, help="Metadata batch value")
    parser.add_argument(
        "--meta-dynamic", action="store_true", help="Set metadata args.dynamic=true"
    )
    parser.add_argument("--meta-nms", action="store_true", help="Set metadata args.nms=true")
    parser.add_argument("--meta-no-end2end", action="store_true", help="Set metadata end2end=false")
    parser.add_argument(
        "--meta-dla", type=int, default=None, help="Override metadata DLA core (0/1)"
    )
    parser.add_argument(
        "--meta-output",
        type=str,
        default=None,
        help="Optional output path for metadata-prefixed engine (default: in-place)",
    )

    args = parser.parse_args()
    sys.exit(build_engine(args))
