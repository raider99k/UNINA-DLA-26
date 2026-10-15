"""
ONNX/DLA Deep Verification Utility.

Consolidates all ONNX verification checks for Jetson Orin DLA 2.0 compatibility.
This module is called automatically at the end of train_student.py after ONNX export.

Usage (standalone):
    python -m unina_dla.utils.verify_dla_onnx path/to/model.onnx
"""

import collections
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import onnx
except ImportError:
    onnx = None

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    ort = None
    np = None


@dataclass
class VerificationResult:
    """Structured result of ONNX/DLA verification."""

    model_path: str
    passed: bool = True
    opset_version: int = 0
    producer: str = ""
    input_shape: List[int] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)
    total_nodes: int = 0
    op_counts: Dict[str, int] = field(default_factory=dict)

    # DLA Checks
    forbidden_ops_found: List[str] = field(default_factory=list)
    dilation_violations: List[str] = field(default_factory=list)
    deconv_violations: List[str] = field(default_factory=list)
    o2m_leakage_found: List[str] = field(default_factory=list)
    repvgg_fusion_ratio: float = 0.0
    activation_counts: Dict[str, int] = field(default_factory=dict)

    # Runtime
    runtime_passed: Optional[bool] = None
    runtime_error: Optional[str] = None

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# DLA 2.0 (Orin) Forbidden/Problematic Operators
# These either cause GPU fallback or are unsupported entirely
FORBIDDEN_OPS = {
    "Softmax",  # DLA does not support Softmax (use Sigmoid or raw logits)
    "Gather",  # Dynamic indexing not supported
    "NonZero",  # Dynamic shape operator
    "TopK",  # Dynamic shape operator
    "QuantizeLinear",  # Explicit Q/DQ graphs do not stay on DLA
    "DequantizeLinear",  # Explicit Q/DQ graphs do not stay on DLA
    "Erf",  # Not in DLA op set
    "ReduceSum",  # Can cause fallback depending on axes
    "ScatterND",  # Dynamic shape
    "NonMaxSuppression",  # NMS is explicitly not DLA-compatible
    "Where",  # Conditional ops are problematic
}


def verify_dla_onnx(
    model_path: str,
    run_inference: bool = False,
    raise_on_failure: bool = False,
    verbose: bool = True,
) -> VerificationResult:
    """
    Perform deep verification of ONNX model for DLA 2.0 compatibility.

    Args:
        model_path: Path to ONNX file.
        run_inference: If True, run a dummy inference with onnxruntime.
        raise_on_failure: If True, raise exception on critical failures.
        verbose: If True, print human-readable summary.

    Returns:
        VerificationResult dataclass with all check results.
    """
    result = VerificationResult(model_path=model_path)

    if onnx is None:
        result.passed = False
        result.errors.append("onnx module not installed. Run: pip install onnx")
        if verbose:
            print("[FATAL] onnx module not installed.")
        if raise_on_failure:
            raise ImportError("onnx module required for verification")
        return result

    if not os.path.exists(model_path):
        result.passed = False
        result.errors.append(f"File not found: {model_path}")
        if verbose:
            print(f"[FATAL] File not found: {model_path}")
        if raise_on_failure:
            raise FileNotFoundError(model_path)
        return result

    # =========================================================================
    # 1. Load and Validate Model Schema
    # =========================================================================
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as e:
        result.passed = False
        result.errors.append(f"onnx.load/check failed: {e}")
        if verbose:
            print(f"[FATAL] onnx.load/check failed: {e}")
        if raise_on_failure:
            raise
        return result

    # Basic Info
    result.opset_version = model.opset_import[0].version if model.opset_import else 0
    result.producer = f"{model.producer_name} {model.producer_version}"

    # Inputs
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        result.input_shape = shape

    # Outputs
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        result.output_names.append(out.name)
        result.output_shapes[out.name] = shape
        out_name_l = out.name.lower()
        if "o2m" in out_name_l:
            result.o2m_leakage_found.append(f"Output name contains O2M marker: {out.name}")
        if not (out_name_l.startswith("reg") or out_name_l.startswith("cls")):
            result.o2m_leakage_found.append(
                f"Unexpected output name (must be reg*/cls*): {out.name}"
            )

    # =========================================================================
    # 2. Operator Analysis
    # =========================================================================
    op_counts = collections.Counter()
    activation_counts = collections.Counter()

    for node in model.graph.node:
        op_counts[node.op_type] += 1

        # Track activations
        if node.op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Mul", "HardSwish", "Silu"]:
            activation_counts[node.op_type] += 1

        # Check forbidden ops
        if node.op_type in FORBIDDEN_OPS:
            result.forbidden_ops_found.append(f"{node.op_type} (node: {node.name})")

        # O2M leakage check: export graph must not carry auxiliary branch symbols.
        node_name_l = (node.name or "").lower()
        if "o2m" in node_name_l:
            result.o2m_leakage_found.append(f"O2M marker in node name: {node.name}")
        for tensor_name in list(node.input) + list(node.output):
            if tensor_name and "o2m" in tensor_name.lower():
                result.o2m_leakage_found.append(f"O2M marker in tensor name: {tensor_name}")

        # Check Conv dilation/padding rule
        if node.op_type == "Conv":
            dilations = [1, 1]
            pads = [0, 0, 0, 0]
            kernel_shape = [1, 1]

            for attr in node.attribute:
                if attr.name == "dilations":
                    dilations = list(attr.ints) if attr.ints else [1, 1]
                elif attr.name == "pads":
                    pads = list(attr.ints) if attr.ints else [0, 0, 0, 0]
                elif attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints) if attr.ints else [1, 1]

            # DLA Rule: If dilation > 1, padding must be < kernel_size
            if len(dilations) >= 2 and len(pads) >= 4 and len(kernel_shape) >= 2:
                if dilations[0] > 1:
                    if pads[0] >= kernel_shape[0] or pads[2] >= kernel_shape[0]:
                        result.dilation_violations.append(
                            f"Conv {node.name}: dilation_h={dilations[0]}, "
                            f"pad_h={pads[0]}/{pads[2]} >= kernel_h={kernel_shape[0]}"
                        )
                if dilations[1] > 1:
                    if pads[1] >= kernel_shape[1] or pads[3] >= kernel_shape[1]:
                        result.dilation_violations.append(
                            f"Conv {node.name}: dilation_w={dilations[1]}, "
                            f"pad_w={pads[1]}/{pads[3]} >= kernel_w={kernel_shape[1]}"
                        )
        elif node.op_type == "ConvTranspose":
            dilations = [1, 1]
            pads = [0, 0, 0, 0]
            kernel_shape = [1, 1]
            groups = 1

            for attr in node.attribute:
                if attr.name == "dilations":
                    dilations = list(attr.ints) if attr.ints else [1, 1]
                elif attr.name == "pads":
                    pads = list(attr.ints) if attr.ints else [0, 0, 0, 0]
                elif attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints) if attr.ints else [1, 1]
                elif attr.name == "group":
                    groups = int(attr.i)

            if groups != 1:
                result.deconv_violations.append(
                    f"ConvTranspose {node.name}: groups={groups} (DLA requires group=1)"
                )

            if any(pad != 0 for pad in pads):
                result.deconv_violations.append(
                    f"ConvTranspose {node.name}: pads={pads} (DLA requires zero padding)"
                )

            if any(dilation != 1 for dilation in dilations):
                result.deconv_violations.append(
                    f"ConvTranspose {node.name}: dilations={dilations} (DLA requires dilation=1)"
                )

            if any(k < 1 or k > 32 for k in kernel_shape):
                result.deconv_violations.append(
                    f"ConvTranspose {node.name}: kernel_shape={kernel_shape} outside DLA-supported range [1, 32]"
                )
        elif node.op_type == "Resize":
            mode = "nearest"
            for attr in node.attribute:
                if attr.name == "mode":
                    mode = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else str(attr.s)
            if mode not in ["nearest", "linear"]:
                result.warnings.append(
                    f"Resize {node.name}: mode='{mode}' may trigger GPU fallback on DLA (recommended: 'nearest')"
                )

    result.total_nodes = len(model.graph.node)
    result.op_counts = dict(op_counts)
    result.activation_counts = dict(activation_counts)

    # =========================================================================
    # 3. RepVGG Fusion Check
    # =========================================================================
    conv_count = op_counts.get("Conv", 0)
    add_count = op_counts.get("Add", 0)

    if conv_count > 0:
        result.repvgg_fusion_ratio = add_count / conv_count
        if result.repvgg_fusion_ratio > 0.5:
            result.warnings.append(
                f"High Add/Conv ratio ({add_count}/{conv_count} = {result.repvgg_fusion_ratio:.2f}). "
                "RepVGG fusion may be incomplete."
            )

    # =========================================================================
    # 4. Activation Check (SiLU detection)
    # =========================================================================
    relu_count = op_counts.get("Relu", 0)

    # Build a map of output_name -> [consuming_node_op_types]
    # to detect if Sigmoid outputs feed into Mul (SiLU pattern)
    output_consumers = collections.defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            output_consumers[inp].append(node.op_type)

    # Check each Sigmoid: is it a final output (head) or consumed by Mul (SiLU)?
    silu_pattern_count = 0
    head_sigmoid_count = 0
    output_names_set = set(result.output_names)

    for node in model.graph.node:
        if node.op_type == "Sigmoid":
            sigmoid_output = node.output[0] if node.output else None
            if sigmoid_output:
                if sigmoid_output in output_names_set:
                    # This Sigmoid is a graph output (classification head) - EXPECTED
                    head_sigmoid_count += 1
                elif "Mul" in output_consumers.get(sigmoid_output, []):
                    # This Sigmoid feeds into Mul -> SiLU pattern - PROBLEM
                    silu_pattern_count += 1

    if silu_pattern_count > 0:
        result.warnings.append(
            f"Detected {silu_pattern_count} SiLU activations (Sigmoid->Mul pattern). "
            "SiLU is not optimal for DLA. These should be in the head only, not backbone."
        )

    if head_sigmoid_count > 0:
        # This is INFO, not a warning - Sigmoid at output is expected for classification
        pass  # Silently acknowledge (or could add to result.info if desired)

    if relu_count == 0:
        result.warnings.append(
            "No ReLU activations found. DLA is optimized for ReLU. "
            "Ensure the model uses ReLU (not SiLU/GELU/HardSwish) in backbone."
        )

    # =========================================================================
    # 5. Runtime Inference Test (Optional)
    # =========================================================================
    if run_inference:
        if ort is None or np is None:
            result.warnings.append("onnxruntime/numpy not installed. Skipping runtime test.")
        else:
            try:
                # Build input shape, handling dynamic dims
                test_shape = []
                for d in result.input_shape:
                    if isinstance(d, int) and d > 0:
                        test_shape.append(d)
                    else:
                        test_shape.append(1)

                dummy_input = np.random.randn(*test_shape).astype(np.float32)
                sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                input_name = sess.get_inputs()[0].name
                _ = sess.run(None, {input_name: dummy_input})
                result.runtime_passed = True
            except Exception as e:
                result.runtime_passed = False
                result.runtime_error = str(e)
                result.errors.append(f"Runtime inference failed: {e}")

    # =========================================================================
    # 6. Final Verdict
    # =========================================================================
    if result.forbidden_ops_found:
        result.passed = False
        result.errors.append(f"Found {len(result.forbidden_ops_found)} forbidden DLA operators.")

    if result.dilation_violations:
        result.passed = False
        result.errors.append(
            f"Found {len(result.dilation_violations)} Conv dilation/padding violations."
        )

    if result.deconv_violations:
        result.passed = False
        result.errors.append(f"Found {len(result.deconv_violations)} ConvTranspose DLA violations.")

    if result.o2m_leakage_found:
        result.passed = False
        result.errors.append(
            f"Found {len(result.o2m_leakage_found)} O2M leakage indicators in exported ONNX graph."
        )

    # =========================================================================
    # 7. Print Summary
    # =========================================================================
    if verbose:
        _print_summary(result)

    if raise_on_failure and not result.passed:
        raise ValueError(f"ONNX/DLA Verification FAILED for {model_path}")

    return result


def _print_summary(result: VerificationResult):
    """Print human-readable verification summary."""
    print("\n" + "=" * 60)
    print("DEEP ONNX/DLA VERIFICATION REPORT")
    print("=" * 60)
    print(f"Model: {os.path.basename(result.model_path)}")
    print(f"Opset: {result.opset_version} | Producer: {result.producer}")
    print(f"Input Shape: {result.input_shape}")
    print(f"Outputs: {result.output_names}")
    print(f"Total Nodes: {result.total_nodes}")

    print("\n--- DLA Compatibility Checks ---")

    # Schema Check
    if not result.errors or not any("onnx.load" in e for e in result.errors):
        print("[PASS] Model schema validated (onnx.checker)")
    else:
        print("[FAIL] Model schema invalid")

    # Forbidden Ops
    if result.forbidden_ops_found:
        print(f"[FAIL] Forbidden DLA Ops: {result.forbidden_ops_found}")
    else:
        print("[PASS] No forbidden DLA operators")

    # Dilation Violations
    if result.dilation_violations:
        print("[FAIL] Conv Dilation Violations (will cause GPU fallback):")
        for v in result.dilation_violations:
            print(f"       - {v}")
    else:
        print("[PASS] All Conv dilation/padding rules satisfied")

    # Deconvolution Violations
    if result.deconv_violations:
        print("[FAIL] ConvTranspose Violations (will prevent strict DLA residency):")
        for v in result.deconv_violations:
            print(f"       - {v}")
    else:
        print("[PASS] All ConvTranspose rules satisfied")

    # O2M leakage
    if result.o2m_leakage_found:
        print("[FAIL] O2M branch leakage detected in export:")
        for leak in result.o2m_leakage_found[:10]:
            print(f"       - {leak}")
    else:
        print("[PASS] Export is O2O-only (no O2M branch leakage)")

    # RepVGG Fusion
    if result.repvgg_fusion_ratio > 0.5:
        print(
            f"[WARN] High Add/Conv ratio ({result.repvgg_fusion_ratio:.2f}). Check RepVGG fusion."
        )
    else:
        print(f"[PASS] Add/Conv ratio ({result.repvgg_fusion_ratio:.2f}) indicates proper fusion")

    # Activations
    print(f"\nActivations: {result.activation_counts}")
    relu_count = result.activation_counts.get("Relu", 0)
    if relu_count > 0:
        print(f"[INFO] ReLU detected ({relu_count} nodes). Good for DLA.")

    # Runtime
    if result.runtime_passed is not None:
        if result.runtime_passed:
            print("[PASS] Runtime inference test passed")
        else:
            print(f"[FAIL] Runtime inference failed: {result.runtime_error}")

    # Warnings
    if result.warnings:
        print("\n--- Warnings ---")
        for w in result.warnings:
            print(f"[WARN] {w}")

    # Final Verdict
    print("\n" + "-" * 60)
    if result.passed:
        print("VERDICT: ✅ PASS - Model is DLA-compatible")
    else:
        print("VERDICT: ❌ FAIL - Model has DLA compatibility issues")
        for e in result.errors:
            print(f"  ERROR: {e}")
    print("=" * 60 + "\n")


# CLI Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m unina_dla.utils.verify_dla_onnx <model.onnx>")
        sys.exit(1)

    result = verify_dla_onnx(sys.argv[1], run_inference=True, verbose=True)
    sys.exit(0 if result.passed else 1)
