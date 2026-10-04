import copy

import torch
import torch.nn as nn

from unina_dla.utils.model_utils import fuse_conv_bn_recursive

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor
except ImportError:
    quant_nn = None
    QuantDescriptor = None


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
    model, per_channel_weights=True, ignore_modules=None, verification_input_size=(1, 3, 640, 640)
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
    """
    if quant_nn is None or QuantDescriptor is None:
        raise ImportError("pytorch_quantization is not installed. Cannot perform QAT.")

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
    print(f"Replacing modules with QuantConv2d (Ignoring: {ignore_modules})...")
    replace_modules_with_quant(model, ignore_modules=ignore_modules)

    return model


def replace_modules_with_quant(module, parent_name="", ignore_modules=None):
    """
    Recursively replace Conv2d with QuantConv2d.
    Excludes layers matching any string in ignore_modules to maintain FP precision.
    """
    if quant_nn is None:
        raise ImportError("pytorch_quantization is not installed. Cannot perform QAT replacement.")

    if ignore_modules is None:
        ignore_modules = ["preds", "upsample"]  # Default fallback

    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # Check if we should skip this module
        # Strict checking: if any ignore string is a substring of full_name
        should_skip = any(ign in full_name for ign in ignore_modules)

        if should_skip:
            # Check if it IS a conv layer we are effectively skipping
            if isinstance(child, nn.Conv2d):
                print(f"[QAT] Skipping Layer (FP32): {full_name}")
            continue

        if isinstance(child, nn.Conv2d):
            print(f"[QAT] Quantizing: {full_name}")
            new_module = quant_nn.QuantConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None,
            )
            new_module.weight.data = child.weight.data
            if child.bias is not None:
                new_module.bias.data = child.bias.data
            setattr(module, name, new_module)
        elif isinstance(child, nn.ConvTranspose2d):
            print(f"[QAT] Quantizing Deconv: {full_name}")
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
            )
            new_module._quant_conv_transpose.weight.data = child.weight.data
            if child.bias is not None:
                new_module._quant_conv_transpose.bias.data = child.bias.data
            setattr(module, name, new_module)
        # Recurse for other modules (containers etc)
        else:
            replace_modules_with_quant(child, full_name, ignore_modules=ignore_modules)


def enable_calibration(model):
    """Enable calibration mode for all quantized modules."""
    if quant_nn is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant()


def enable_quantization(model):
    """Enable proper quantization (disable calibration)."""
    if quant_nn is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_calib()
            module.enable_quant()


def freeze_quantization_observers(model):
    """
    Freeze the min/max ranges of all quantizers (Oscillation Dampening).

    Call this near the end of QAT training (last 10-20% of epochs).
    The weights can still update, but the scale factors (S) are locked.
    This prevents "scale jitter" in the final epochs and helps the loss converge.
    """
    if quant_nn is None:
        return
    frozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_calib()  # Stops updating min/max
            module.enable_quant()  # Keeps applying fake-quantization
            frozen_count += 1
    print(f"[QAT] Froze {frozen_count} quantizer observers (Oscillation Dampening).")
