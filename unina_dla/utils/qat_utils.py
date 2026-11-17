import copy
import logging

import torch
import torch.nn as nn

from unina_dla.utils.model_utils import fuse_conv_bn_recursive

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor
except ImportError:
    quant_nn = None
    QuantDescriptor = None


def _silence_pytorch_quantization_logs():
    """Silence verbose routine warnings emitted by pytorch_quantization via absl.logging."""
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
    logging.getLogger("absl").setLevel(logging.ERROR)


class PatchedQuantConvTranspose2d(nn.Module):
    """
    Compatibility wrapper for QuantConvTranspose2d.

    Fixes PyTorch >= 1.12 compatibility issue where _output_padding() signature
    changed to require 'num_spatial_dims' as an additional argument, which the
    pytorch_quantization library's QuantConvTranspose2d doesn't properly handle.

    Instead of inheriting (which causes MRO conflicts), we wrap the QuantConvTranspose2d
    and override forward to compute output_size explicitly.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        super().__init__()

        if quant_nn is None:
            raise ImportError("pytorch_quantization is required for PatchedQuantConvTranspose2d")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
        )
        self.groups = groups
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Create the underlying QuantConvTranspose2d
        self._quant_conv_transpose = quant_nn.QuantConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
        )

    @property
    def weight(self):
        return self._quant_conv_transpose.weight

    @property
    def bias(self):
        return self._quant_conv_transpose.bias

    def forward(self, x):
        """
        Forward pass that bypasses QuantConvTranspose2d._output_padding signature issues.
        """
        # Use functional API with fixed output_padding to avoid the problematic
        # internal _output_padding() signature mismatch in some library versions.
        return torch.nn.functional.conv_transpose2d(
            self._quant_conv_transpose._input_quantizer(x),
            self._quant_conv_transpose._weight_quantizer(self._quant_conv_transpose.weight),
            self._quant_conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


def prepare_qat_model(
    model,
    per_channel_weights=True,
    ignore_modules=None,
    verification_input_size=(1, 3, 640, 640),
    verbose=False,
):
    """
    Prepare model for QAT:
    1. Fuse RepVGG blocks (MUST be done before QAT)
    2. Replace standard Conv2d with QuantConv2d
    3. Exclude sensitive layers (final prediction heads)

    Args:
        model: The model to prepare.
        per_channel_weights: If True, use per-channel quantization for weights.
        ignore_modules: List of substring names to exclude from quantization.
        verification_input_size: Tuple for input shape to verify fusion.
        verbose: If True, print every quantized/skipped layer individually.
    """
    if quant_nn is None or QuantDescriptor is None:
        raise ImportError("pytorch_quantization is not installed. Cannot perform QAT.")

    _silence_pytorch_quantization_logs()
    target_device = _infer_model_device(model)

    # 0. Set default ignore list if None (preds heads and bilinear deconvolution upsampling)
    if ignore_modules is None:
        ignore_modules = ["preds", "upsample"]

    # 1. Fuse RepVGG branches (with Verification)
    # Check if model is already fused (has rbr_reparam instead of rbr_dense/rbr_1x1)
    already_fused = False
    for name, module in model.named_modules():
        if hasattr(module, "rbr_reparam") and module.rbr_reparam is not None:
            already_fused = True
            break

    if already_fused:
        print("[QAT] Model already fused (rbr_reparam detected). Skipping fusion.")
    elif hasattr(model, "switch_to_deploy"):
        print("Fusing RepVGG blocks for QAT...")

        # Verify fusion if input size is provided
        if verification_input_size is not None:
            print(f"Verifying fusion with input shape {verification_input_size}...")
            model_orig = copy.deepcopy(model)
            model_orig.eval()
            model.eval()  # Set to eval before fusion to freeze BN running stats
            model.switch_to_deploy()
            # Reset deploy=False to keep outputs comparable for verification
            model.deploy = False
            if hasattr(model, "head") and hasattr(model.head, "deploy"):
                model.head.deploy = False
            model.eval()

            # Generate random input
            device = next(model.parameters()).device
            dummy_input = torch.randn(verification_input_size).to(device)

            def get_max_diff(obj1, obj2):
                if isinstance(obj1, torch.Tensor):
                    return torch.abs(obj1 - obj2).max().item()
                elif isinstance(obj1, (list, tuple)):
                    max_d = 0.0
                    for o1, o2 in zip(obj1, obj2):
                        max_d = max(max_d, get_max_diff(o1, o2))
                    return max_d
                elif isinstance(obj1, dict):
                    max_d = 0.0
                    for k in obj1:
                        max_d = max(max_d, get_max_diff(obj1[k], obj2[k]))
                    return max_d
                return 0.0

            with torch.no_grad():
                out_orig = model_orig(dummy_input)
                out_fused = model(dummy_input)

            diff = get_max_diff(out_orig, out_fused)
            print(f"Fusion verification: Max Difference = {diff:.6f}")
            if diff > 1e-3:  # Tolerance for float32 fusion
                print("[WARNING] Fusion difference is high! Check RepVGG implementation.")
            else:
                print("[INFO] Fusion verified successfully.")

            del model_orig
        else:
            model.switch_to_deploy()

        # Fuse standard Conv+BN pairs before quantization
        print("Fusing standard Conv+BN pairs recursively for QAT...")
        fused_count = fuse_conv_bn_recursive(model)
        print(f"Fused {fused_count} standard Conv+BN pairs.")

    # 4. Initialize Quant Descriptors
    # DLA supports INT8.
    # ACTIVATIONS: Percentile Calibration (via high-bin Histogram)
    # Standard MinMax is fragile for SiLU-to-ReLU distilled networks because the activation
    # distributions change drastically and often contain outliers from RepVGG spikes.
    # Histogram calibration with high bins (default in pytorch_quantization) effectively
    # acts as Percentile Calibration (99.99%), clipping outliers to preserve precision.
    quant_desc_input = QuantDescriptor(calib_method="histogram", unsigned=False)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    if hasattr(quant_nn, "QuantConvTranspose2d") and hasattr(
        quant_nn.QuantConvTranspose2d, "set_default_quant_desc_input"
    ):
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)

    # WEIGHTS: Max Calibration
    # For DLA-optimized per-channel weights (axis=0), use 'max' calibration method.
    # This ensures representable ranges for cross-channel magnitude differences.
    axis = 0 if per_channel_weights else None
    weight_calib_method = "max" if per_channel_weights else "histogram"
    quant_desc_weight = QuantDescriptor(calib_method=weight_calib_method, axis=axis)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
    if hasattr(quant_nn, "QuantConvTranspose2d") and hasattr(
        quant_nn.QuantConvTranspose2d, "set_default_quant_desc_weight"
    ):
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)

    print(
        f"Quantization Configuration: Input=Signed Histogram (Percentile), Weights={'Per-Channel Max' if per_channel_weights else 'Per-Tensor Histogram'}"
    )

    # 5. Replace modules
    if verbose:
        print(f"Replacing modules with QuantConv2d (Ignoring: {ignore_modules})...")
    replace_modules_with_quant(model, ignore_modules=ignore_modules, verbose=verbose)
    sync_quant_device(model, target_device)

    return model


def replace_modules_with_quant(module, parent_name="", ignore_modules=None, verbose=False):
    """
    Recursively replace Conv2d with QuantConv2d.
    Excludes layers matching any string in ignore_modules to maintain FP precision.
    """
    if quant_nn is None:
        raise ImportError("pytorch_quantization is not installed. Cannot perform QAT replacement.")

    if ignore_modules is None:
        ignore_modules = ["preds", "upsample"]  # Default fallback

    counts = {"conv": 0, "deconv": 0, "skipped": 0, "shared": 0}

    def _recurse(mod, p_name):
        for name, child in mod.named_children():
            full_name = f"{p_name}.{name}" if p_name else name

            # Strict checking: if any ignore string is a substring of full_name
            should_skip = any(ign in full_name for ign in ignore_modules)

            if should_skip:
                if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                    counts["skipped"] += 1
                    if verbose:
                        print(f"[QAT] Skipping Layer (FP32): {full_name}")
                continue

            if isinstance(child, nn.Conv2d):
                counts["conv"] += 1
                if verbose:
                    print(f"[QAT] Quantizing: {full_name}")
                child_device = child.weight.device
                new_module = quant_nn.QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups,
                    child.bias is not None,
                ).to(child_device)
                new_module.weight.data = child.weight.data
                if child.bias is not None:
                    new_module.bias.data = child.bias.data
                setattr(mod, name, new_module)
            elif isinstance(child, nn.ConvTranspose2d):
                counts["deconv"] += 1
                if verbose:
                    print(f"[QAT] Quantizing Deconv: {full_name}")
                child_device = child.weight.device
                # Use PatchedQuantConvTranspose2d to fix PyTorch >= 1.12 compatibility
                new_module = PatchedQuantConvTranspose2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.output_padding,
                    child.groups,
                    child.bias is not None,
                    child.dilation,
                ).to(child_device)
                new_module._quant_conv_transpose.weight.data = child.weight.data
                if child.bias is not None:
                    new_module._quant_conv_transpose.bias.data = child.bias.data
                setattr(mod, name, new_module)
            else:
                if hasattr(child, "shared_quantizer") and getattr(child, "shared_quantizer", None) is None:
                    quant_desc = getattr(quant_nn.QuantConv2d, "default_quant_desc_input", None)
                    sq = (
                        quant_nn.TensorQuantizer(quant_desc)
                        if quant_desc is not None
                        else quant_nn.TensorQuantizer()
                    )
                    child_device = _infer_model_device(child)
                    child.shared_quantizer = sq.to(child_device) if child_device is not None else sq
                    counts["shared"] += 1
                    if verbose:
                        print(f"[QAT] Attached shared_quantizer to Concat-invariant module: {full_name}")
                _recurse(child, full_name)

    _recurse(module, parent_name)
    if not verbose and parent_name == "":
        print(
            f"[QAT] Quantized {counts['conv']} Conv2d + {counts['deconv']} ConvTranspose2d layers "
            f"(skipped {counts['skipped']} FP32 layers, {counts['shared']} shared_quantizer)."
        )


def _infer_model_device(model):
    """Return the device of a model's first parameter or buffer, or None."""
    if model is None:
        return None
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return None


def sync_quant_device(model, device=None):
    """Move model parameters, buffers, and quantizer `_amax` tensors onto the target device."""
    _silence_pytorch_quantization_logs()
    if model is None:
        return model
    target_device = torch.device(device) if device is not None else _infer_model_device(model)
    if target_device is None:
        return model

    model.to(target_device)
    for mod in model.modules():
        amax = getattr(mod, "_amax", None)
        if torch.is_tensor(amax) and amax.device != target_device:
            buffers = getattr(mod, "_buffers", None)
            if isinstance(buffers, dict) and "_amax" in buffers:
                buffers["_amax"] = amax.to(target_device)
            else:
                mod._amax = amax.to(target_device)
    return model


def enable_calibration(model):
    """Enable calibration mode for all quantized modules."""
    if quant_nn is None:
        return
    _silence_pytorch_quantization_logs()
    model._quant_observers_frozen = False
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant()


def enable_quantization(model, device=None):
    """Enable proper quantization (disable calibration) and align quantizer device."""
    _silence_pytorch_quantization_logs()
    if quant_nn is not None:
        model._quant_observers_frozen = False
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module.disable_calib()
                module.enable_quant()
    sync_quant_device(model, device)


@torch.no_grad()
def update_weight_quantizers(model):
    """
    Dynamically update per-channel/per-tensor weight quantizer amax buffers to track
    weight evolution during QAT fine-tuning until freeze_quantization_observers() is called.
    """
    if getattr(model, "_quant_observers_frozen", False):
        return
    for module in model.modules():
        wq = getattr(module, "_weight_quantizer", None)
        w = getattr(module, "weight", None)
        if wq is not None and w is not None and torch.is_tensor(w):
            amax = getattr(wq, "_amax", None)
            if amax is not None and torch.is_tensor(amax) and amax.numel() > 0:
                w_abs = w.detach().abs()
                if amax.ndim == 0 or amax.numel() == 1:
                    new_amax = w_abs.max().reshape_as(amax)
                else:
                    reduce_dims = tuple(range(1, w_abs.ndim))
                    new_amax = w_abs.amax(dim=reduce_dims).reshape_as(amax)
                amax.copy_(new_amax.clamp(min=1e-7).to(device=amax.device, dtype=amax.dtype))


def freeze_quantization_observers(model):
    """
    Freeze the min/max ranges of all quantizers (Oscillation Dampening).

    Call this near the end of QAT training (last 10-20% of epochs).
    The weights can still update, but the scale factors (S) are locked.
    This prevents "scale jitter" in the final epochs and helps the loss converge.
    """
    already_frozen = getattr(model, "_quant_observers_frozen", False)
    model._quant_observers_frozen = True
    if quant_nn is None:
        return
    _silence_pytorch_quantization_logs()
    frozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_calib()  # Stops updating min/max
            module.enable_quant()  # Keeps applying fake-quantization
            frozen_count += 1
    if not already_frozen:
        print(f"[QAT] Froze {frozen_count} quantizer observers (Oscillation Dampening).")

