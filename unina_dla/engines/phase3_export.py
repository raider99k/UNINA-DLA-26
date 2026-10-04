import inspect
import json
import os
import subprocess

import torch

from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version
from unina_dla.utils.model_utils import fuse_conv_bn_recursive
from unina_dla.utils.qat_utils import prepare_qat_model

try:
    from pytorch_quantization import nn as quant_nn
except ModuleNotFoundError:
    quant_nn = None


def build_o2o_output_names(head):
    """
    Return export output names for O2O deploy branch only.
    """
    if hasattr(head, "strides"):
        n_levels = len(head.strides)
    elif hasattr(head, "stride"):
        stride_attr = head.stride
        n_levels = len(stride_attr) if hasattr(stride_attr, "__len__") else 1
    else:
        n_levels = 3
    return [f"reg{3 + i}" for i in range(n_levels)] + [f"cls{3 + i}" for i in range(n_levels)]


def _prepare_clean_deploy_state_dict(state_dict):
    """
    Strip quantizer-only tensors from a QAT checkpoint and remap wrapped deconv keys
    back onto the plain deploy model.
    """
    filtered = {}
    dropped = []
    remapped = []
    quantizer_tokens = (
        "_input_quantizer",
        "_weight_quantizer",
        "_output_quantizer",
        "_pre_quant_scale",
        "_amax",
        "_calibrator",
    )

    for key, value in state_dict.items():
        if any(token in key for token in quantizer_tokens):
            dropped.append(key)
            continue

        new_key = key.replace("._quant_conv_transpose.", ".")
        if new_key != key:
            remapped.append((key, new_key))
        filtered[new_key] = value

    return filtered, dropped, remapped


def run_export(args):
    """
    Phase 3: Export to ONNX (DLA-Optimized).
    Consolidates logic from scripts/export_onnx.py and train_student.py.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: EXPORT TO ONNX")
    print("=" * 60)

    # 1. Determine Parameters
    # Support both training args (SimpleNamespace) and standalone args (Namespace)
    checkpoint = getattr(args, "checkpoint", None)  # from export script
    output_file = getattr(args, "output", None)  # from export script

    # Fallback to training args
    if checkpoint is None:
        # If running from training, find the best checkpoint based on phases
        # Logic: QAT > Sparse > Distill > Standalone
        if hasattr(args, "enable_qat") and args.enable_qat:
            checkpoint = os.path.join(args.output_dir, "best_qat.pth")
        elif hasattr(args, "enable_sparsity") and args.enable_sparsity:
            checkpoint = os.path.join(args.output_dir, "best_sparse.pth")
        elif hasattr(args, "enable_distillation") and args.enable_distillation:
            checkpoint = os.path.join(args.output_dir, "best_distill.pth")
        else:
            checkpoint = os.path.join(args.output_dir, "best_standalone.pth")

    if output_file is None:
        # Auto-name based on checkpoint name
        base_dir = os.path.dirname(checkpoint)
        base_name = os.path.splitext(os.path.basename(checkpoint))[0]

        # Determine properties for naming (prioritize args)
        is_distilled = getattr(args, "distilled", False) or getattr(
            args, "enable_distillation", False
        )
        is_sparse = getattr(args, "sparse", False) or getattr(args, "enable_sparsity", False)
        is_qat = (
            getattr(args, "qat", False)
            or getattr(args, "enable_qat", False)
            or getattr(args, "qat_clean_deploy", False)
        )

        # Build suffix
        suffix_parts = []

        # 1. Distillation
        if (
            is_distilled
            and "_distilled" not in base_name
            and "distill" not in base_name
            and "standalone" not in base_name
        ):
            suffix_parts.append("distilled")

        # 2. Sparsity
        if (
            is_sparse
            and "_sparse" not in base_name
            and "sparse" not in base_name
            and "standalone" not in base_name
        ):
            suffix_parts.append("sparse")

        # 3. QAT (Usually in base_name, but add if missing and active)
        if is_qat and "standalone" not in base_name:
            if getattr(args, "qat_clean_deploy", False):
                if "qatclean" not in base_name and "clean" not in base_name:
                    suffix_parts.append("qatclean")
            elif "_qat" not in base_name and "qat" not in base_name:
                suffix_parts.append("qat")

        # 4. Resolution (Last)
        suffix_parts.append(f"{args.imgsz_hw[0]}x{args.imgsz_hw[1]}")

        suffix = "_".join(suffix_parts)
        output_file = os.path.join(base_dir, f"unina_dla_{base_name}_{suffix}.onnx")

    print(f"Exporting model from {checkpoint} to {output_file}...")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # 2. Load Checkpoint & Inspect
    ckpt = torch.load(checkpoint, map_location="cpu")

    # Check if checkpoint is from QAT
    sample_keys = []
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sample_keys = list(ckpt["model"].keys())
        elif "ema" in ckpt and isinstance(ckpt["ema"], dict):
            sample_keys = list(ckpt["ema"].keys())
        else:
            sample_keys = list(ckpt.keys())
    has_quant_keys = any(("_amax" in k or "quant" in k) for k in sample_keys)

    # Determine model properties from checkpoint/args
    source_is_qat = (
        getattr(args, "qat", False)
        or getattr(args, "enable_qat", False)
        or getattr(args, "qat_clean_deploy", False)
        or has_quant_keys
    )
    is_sparse = (
        getattr(args, "sparse", False)
        or getattr(args, "enable_sparsity", False)
        or (isinstance(ckpt, dict) and ckpt.get("sparsity_enabled", False))
    )
    clean_qat_deploy = bool(getattr(args, "qat_clean_deploy", False)) and source_is_qat
    export_with_qat_graph = source_is_qat and not clean_qat_deploy

    # Select weights: prefer EMA for non-QAT models if available (matches validation)
    raw_sd = ckpt
    if not source_is_qat and isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
            raw_sd = ckpt["ema"]
            print("[Export] Using EMA weights from checkpoint (matches best validation).")
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            raw_sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        raw_sd = ckpt["model"]

    # Clean state_dict: remove compiler wrappers (_orig_mod.) and profiling artifacts
    state_dict = {}
    for k, v in raw_sd.items():
        if k.endswith(("total_ops", "total_params")):
            continue
        k_clean = k.replace("_orig_mod.", "")
        state_dict[k_clean] = v

    # Get Config
    num_classes = getattr(args, "num_classes", 4)
    # Check if we can infer num_classes from state_dict (head.cv3)
    # This is tricky without instantiating, so rely on args for now.

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if export_with_qat_graph:
        if quant_nn is None:
            raise ImportError("Cannot export QAT model: pytorch_quantization not installed.")
        print("[Export] Mode: QAT (FP32 weights with FakeQuant Nodes)")
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

    is_fused_checkpoint = any("rbr_reparam" in k for k in state_dict.keys())

    if export_with_qat_graph:
        if quant_nn is None:
            raise ImportError("Cannot export QAT model: pytorch_quantization not installed.")
        print("[Export] Mode: QAT (FP32 weights with FakeQuant Nodes)")
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"

        if is_fused_checkpoint:
            print(f"[Export] Detected FUSED QAT checkpoint (Gen-{arch_version}). Initializing fused architecture...")
            base_model = UNINA_DLA(num_classes=num_classes, deploy=True, version=arch_version).to(device)
        else:
            print(f"[Export] Detected UNFUSED QAT checkpoint (Gen-{arch_version}). Initializing unfused architecture...")
            base_model = UNINA_DLA(num_classes=num_classes, deploy=False, version=arch_version).to(device)

        export_model = prepare_qat_model(
            base_model, verification_input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1])
        )

        # Set deploy=True flag on UNINA_DLA wrapper so forward() returns flat coordinate tuple
        if isinstance(export_model, UNINA_DLA):
            export_model.deploy = True
            if hasattr(export_model, "head") and hasattr(export_model.head, "deploy"):
                export_model.head.deploy = True
        else:
            # If prepare_qat_model wrapped it, try to find the flag
            if hasattr(export_model, "deploy"):
                export_model.deploy = True
            if hasattr(export_model, "head") and hasattr(export_model.head, "deploy"):
                export_model.head.deploy = True
    elif clean_qat_deploy:
        arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
        print(f"[Export] Mode: Clean Deploy from QAT checkpoint (QDQ stripped, Gen-{arch_version})")
        export_model = UNINA_DLA(num_classes=num_classes, deploy=is_fused_checkpoint, version=arch_version).to(device)
    else:
        arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
        print(f"[Export] Mode: Standard FP32/FP16 (Fused RepVGG, Gen-{arch_version})")
        if is_fused_checkpoint:
            print(f"[Export] Detected FUSED checkpoint (Gen-{arch_version}). Initializing fused deploy architecture...")
            export_model = UNINA_DLA(num_classes=num_classes, deploy=True, version=arch_version).to(device)
        else:
            print(f"[Export] Detected UNFUSED checkpoint (Gen-{arch_version}). Initializing multi-branch architecture for weight loading...")
            export_model = UNINA_DLA(num_classes=num_classes, deploy=False, version=arch_version).to(device)

    # 4. Load Weights
    # Strict=False allows missing aux buffers in non-QAT or vice versa
    state_dict_to_load = state_dict
    dropped_quant_keys = []
    remapped_keys = []
    if clean_qat_deploy:
        state_dict_to_load, dropped_quant_keys, remapped_keys = _prepare_clean_deploy_state_dict(
            state_dict
        )
        print(
            f"[Export] Clean-deploy checkpoint normalization: "
            f"dropped {len(dropped_quant_keys)} quantizer keys, remapped {len(remapped_keys)} wrapped deconv keys."
        )
    try:
        missing_keys, unexpected_keys = export_model.load_state_dict(
            state_dict_to_load, strict=False
        )
        filtered_missing = [
            k
            for k in missing_keys
            if not (k.startswith("head.o2m_") or k.endswith(("total_ops", "total_params")))
        ]
        filtered_unexpected = [
            k
            for k in unexpected_keys
            if not (k.startswith("head.o2m_") or k.endswith(("total_ops", "total_params")))
        ]
        if filtered_unexpected:
            print(
                f"[Export] Ignored {len(filtered_unexpected)} unexpected checkpoint keys during load: {filtered_unexpected[:5]}..."
            )
        if filtered_missing:
            print(
                f"[Export] WARNING: Missing {len(filtered_missing)} keys during load: {filtered_missing[:5]}..."
            )
    except RuntimeError as e:
        print(f"Warning: Strict processing failed ({e}). Loading non-strict.")

    export_model.eval()

    # 5. Optimization (Fusion)
    fused_count = 0
    if not export_with_qat_graph:
        # Fuse RepVGG blocks (if not already deploy mode)
        if hasattr(export_model, "switch_to_deploy"):
            export_model.switch_to_deploy()

        print("Running exhaustive recursive Conv+BN fusion for DLA...")
        fused_count = fuse_conv_bn_recursive(export_model)
        print(f"[Export] Fused {fused_count} additional Conv+BN pairs.")
    else:
        print("[Export] Skipping fusion for QAT model (preserves QuantNodes).")

    export_model.eval()
    export_model.to(device)

    # 6. Dummy Input & Precision
    dummy_input = torch.randn(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]).to(device)

    if clean_qat_deploy:
        print(
            "[Export] Precision: FP32 (Clean deploy export for TensorRT calibration / DLA checks)"
        )
        export_model.float()
        dummy_input = dummy_input.float()
    elif device.type == "cuda" and not export_with_qat_graph:
        print("[Export] Precision: FP16 (Half) for DLA Input Optimization")
        export_model.half()
        dummy_input = dummy_input.half()
    else:
        print("[Export] Precision: FP32 (Required for QAT/CPU)")
        export_model.float()
        dummy_input = dummy_input.float()

    # 7. ONNX Export
    # Hard guard: export path must expose O2O-only deploy interface.
    with torch.no_grad():
        probe_out = export_model(dummy_input)
    if not isinstance(probe_out, tuple) or len(probe_out) != 6:
        raise RuntimeError(
            "Export model did not produce the expected O2O-only 6-tensor interface "
            "(reg3..regN + cls3..clsN). Refusing export to prevent O2M leakage."
        )

    # Generate names dynamically (reg3..regN, cls3..clsN), O2M excluded by design.
    output_names = build_o2o_output_names(export_model.head)
    n_levels = len(output_names) // 2
    print(f"[Export] Dynamic output names ({n_levels} levels): {output_names}")

    # Target ONNX opset 17 (supported natively by TensorRT 8.6+ and DLA)
    target_opset = 17

    # Dynamic axes configuration (static shapes enforced by default for DLA residency)
    dynamic_axes = None
    if getattr(args, "dynamic", False):
        print("[Export] Dynamic Axes Enabled (Batch Size)")
        # Map dynamic batch size for all inputs/outputs
        dynamic_axes = {"images": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    export_kwargs = {
        "model": export_model,
        "args": dummy_input,
        "f": output_file,
        "verbose": False,
        "opset_version": target_opset,
        "input_names": ["images"],
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }

    # QAT + torch.export path can fail on data-dependent fake-quant scales.
    # Prefer legacy ONNX tracer for QAT when available.
    supports_dynamo = False
    supports_external_data = False
    try:
        export_sig = inspect.signature(torch.onnx.export)
        supports_dynamo = "dynamo" in export_sig.parameters
        supports_external_data = "external_data" in export_sig.parameters
    except (TypeError, ValueError):
        supports_dynamo = False
        supports_external_data = False

    use_external_data = bool(getattr(args, "external_data", False))
    if supports_external_data:
        export_kwargs["external_data"] = use_external_data
        if use_external_data:
            print("[Export] External ONNX data enabled (.onnx + .onnx.data).")
        else:
            print("[Export] Exporting single-file ONNX (embedded weights).")

    if export_with_qat_graph and supports_dynamo:
        print(
            "[Export] QAT export forcing legacy ONNX tracer (dynamo=False) to avoid torch.export symbolic guard issues."
        )
        export_kwargs["dynamo"] = False
    elif export_with_qat_graph:
        print(
            "[Export] QAT export using default ONNX tracer (torch.onnx.export has no dynamo parameter in this torch build)."
        )

    try:
        torch.onnx.export(**export_kwargs)
    except Exception as e:
        if not export_with_qat_graph:
            raise

        err_msg = str(e)
        is_torch_export_error = (
            "TorchExportError" in err_msg
            or "GuardOnDataDependentSymNode" in err_msg
            or "torch.export" in err_msg
            or "symbolic_shapes" in err_msg
        )

        if not is_torch_export_error:
            raise

        print(
            "[Export] QAT export hit torch.export-style symbolic guard error; retrying with legacy-compatible ONNX export settings..."
        )
        retry_kwargs = dict(export_kwargs)
        retry_kwargs.pop("dynamo", None)
        if supports_dynamo:
            retry_kwargs["dynamo"] = False

        try:
            torch.onnx.export(**retry_kwargs)
        except Exception as retry_err:
            raise RuntimeError(
                f"QAT ONNX export failed after legacy-tracer retry for checkpoint '{checkpoint}'. "
                "Recommendation: keep QAT fake-quant enabled and ensure export runs with legacy ONNX tracer "
                "(dynamo=False) in this environment."
            ) from retry_err

    print(f"Export Success: {output_file}")
    if not use_external_data:
        external_data_file = output_file + ".data"
        if os.path.exists(external_data_file):
            os.remove(external_data_file)

    # 8. Generate Manifest & Export INT8 Calibration Scales
    scale_file = None
    if source_is_qat:
        calib_scales = {}
        for k, v in raw_sd.items():
            if ("_amax" in k or "quant" in k) and torch.is_tensor(v):
                v_clean = v.detach().cpu()
                if v_clean.numel() == 1:
                    calib_scales[k] = {
                        "amax": float(v_clean.item()),
                        "scale": float(v_clean.item()) / 127.0,
                    }
                else:
                    calib_scales[k] = {
                        "amax": [float(x) for x in v_clean.flatten()],
                        "scale": [float(x) / 127.0 for x in v_clean.flatten()],
                    }
        if calib_scales:
            scale_file = os.path.splitext(output_file)[0] + "_int8_scales.json"
            with open(scale_file, "w") as f:
                json.dump(calib_scales, f, indent=4)
            print(f"[Export] Preserved {len(calib_scales)} INT8 quantization scale factors in {scale_file} for TensorRT DLA.")

    manifest = {
        "export_type": "QAT_CLEAN_DEPLOY"
        if clean_qat_deploy
        else ("QAT" if export_with_qat_graph else "FP16_Half"),
        "opset_version": target_opset,
        "input_shape": [1, 3, args.imgsz_hw[0], args.imgsz_hw[1]],
        "input_dtype": "float32" if (export_with_qat_graph or clean_qat_deploy) else "float16",
        "num_classes": num_classes,
        "checkpoint": checkpoint,
        "output_file": output_file,
        "fused_conv_bn_count": fused_count,
        "is_sparse": is_sparse,
        "source_is_qat": source_is_qat,
        "clean_qat_deploy": clean_qat_deploy,
        "external_data": use_external_data,
        "int8_scales_file": scale_file,
        "dla_optimized": True,
        "architecture_variant": "dual_head_tal_o2o",
        "runtime_policy": "pure_o2o_no_nms",
    }

    # Add Git Hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        manifest["git_hash"] = git_hash
    except Exception:
        manifest["git_hash"] = "unknown"

    manifest_file = os.path.splitext(output_file)[0] + "_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=4)

    if getattr(args, "verify_dla", True):
        print("[Export] Running strict DLA ONNX verification...")
        from unina_dla.utils.verify_dla_onnx import verify_dla_onnx

        verify_dla_onnx(output_file, run_inference=False, raise_on_failure=True, verbose=True)

    return output_file
