from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
    verbose: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    Compute and print model summary: layers, parameters, and GFLOPs.

    Mimics Ultralytics-style output:
        Model summary: 123 layers, 1,234,567 parameters, 1,234,567 gradients, 12.3 GFLOPs

    Args:
        model: The PyTorch model to analyze.
        input_size: Input tensor shape (B, C, H, W). Default: (1, 3, 640, 640) for YOLO.
        verbose: If True, prints the summary to stdout.
        device: Device for dummy input (default None preserves current model device).

    Returns:
        dict with keys: 'layers', 'params', 'gradients', 'gflops'
    """
    try:
        orig_device = next(model.parameters()).device
    except (StopIteration, Exception):
        orig_device = torch.device(device or "cpu")

    target_device = orig_device if device is None else torch.device(device)

    try:
        model = model.to(target_device)
        model.eval()

        # Count layers (all modules, not just named)
        n_layers = sum(1 for _ in model.modules())

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Compute GFLOPs (Multiply-Accumulate operations / GMACs, consistent with Ultralytics conventions)
        gflops = 0.0
        dummy = torch.zeros(input_size, device=target_device)
        try:
            from thop import profile

            custom_ops = {}
            try:
                from pytorch_quantization.nn.modules.quant_conv import (
                    QuantConv2d,
                    QuantConvTranspose2d,
                )

                def count_quant_conv2d(m, x, y):
                    x_in = x[0]
                    cin = m.in_channels
                    cout = m.out_channels
                    kh, kw = (
                        m.kernel_size
                        if isinstance(m.kernel_size, tuple)
                        else (m.kernel_size, m.kernel_size)
                    )
                    groups = getattr(m, "groups", 1)
                    batch_size = x_in.size(0)
                    out_h = y.size(2)
                    out_w = y.size(3)
                    kernel_ops = (cin // groups) * kh * kw
                    bias_ops = 1 if getattr(m, "bias", None) is not None else 0
                    ops_per_element = kernel_ops + bias_ops
                    total_ops = batch_size * cout * out_h * out_w * ops_per_element
                    m.total_ops += torch.DoubleTensor([int(total_ops)])

                custom_ops[QuantConv2d] = count_quant_conv2d
                custom_ops[QuantConvTranspose2d] = count_quant_conv2d
            except Exception:
                pass

            try:
                from unina_dla.utils.qat_utils import PatchedQuantConvTranspose2d

                custom_ops[PatchedQuantConvTranspose2d] = count_quant_conv2d
            except Exception:
                pass

            flops, _ = profile(model, inputs=(dummy,), custom_ops=custom_ops, verbose=False)
            gflops = flops / 1e9
        except Exception:
            gflops = 0.0

        # Fallback 1: Direct module forward-hook MACs counter (robust for any custom or quantized conv)
        if gflops == 0.0:
            try:
                macs_accum = [0]
                hooks = []

                def make_hook(layer):
                    def hook(module, inp, outp):
                        if hasattr(layer, "weight") and layer.weight is not None:
                            cin = getattr(layer, "in_channels", layer.weight.shape[1] if layer.weight.dim() > 1 else 1)
                            groups = getattr(layer, "groups", 1)
                            kh = layer.weight.shape[2] if layer.weight.dim() > 2 else 1
                            kw = layer.weight.shape[3] if layer.weight.dim() > 3 else 1
                            t_out = outp[0] if isinstance(outp, (tuple, list)) else outp
                            if hasattr(t_out, "numel") and hasattr(t_out, "shape") and t_out.dim() > 0:
                                out_numel = t_out.numel() // t_out.shape[0]
                                macs_accum[0] += out_numel * (cin // groups) * kh * kw
                    return hook

                for mod in model.modules():
                    mod_cls = mod.__class__.__name__
                    if "Conv" in mod_cls or isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                        hooks.append(mod.register_forward_hook(make_hook(mod)))

                with torch.no_grad():
                    model(dummy)
                for h in hooks:
                    h.remove()

                if macs_accum[0] > 0:
                    gflops = macs_accum[0] / 1e9
            except Exception:
                pass

        # Fallback 2: FlopCounterMode (divided by 2 for GMACs)
        if gflops == 0.0:
            try:
                from torch.utils.flop_counter import FlopCounterMode

                with FlopCounterMode(model, display=False) as fcm:
                    model(dummy)
                gflops = (fcm.get_total_flops() / 2.0) / 1e9
            except Exception:
                pass

        # Final fallback: Rough estimate based on parameter count
        if gflops == 0.0:
            avg_spatial = (input_size[2] / 16) * (input_size[3] / 16)
            gflops = 2 * n_params * avg_spatial / 1e9
            if verbose:
                print("[WARNING] Could not compute exact GFLOPs. Using rough estimate.")

        result = {"layers": n_layers, "params": n_params, "gradients": n_grads, "gflops": gflops}

        if verbose:
            print(
                f"Model summary: {n_layers} layers, {n_params:,} parameters, "
                f"{n_grads:,} gradients, {gflops:.1f} GFLOPs"
            )

        return result
    finally:
        try:
            model.to(orig_device)
        except Exception:
            pass


def fuse_conv_bn_recursive(model):
    """
    Exhaustively and recursively search for Conv2d + BatchNorm2d pairs in a model
    and fuse them into a single Conv2d layer.

    This is critical for DLA 2.0 residency, as standalone BatchNorm layers
    trigger GPU fallbacks.

    Args:
        model (nn.Module): The model to fuse.

    Returns:
        int: Total number of fused layers.
    """
    fused_count = 0

    # Standard torch modules to ignore for recursion as they don't contain children
    # we're interested in fusing directly, but we still recurse into containers.

    def _fuse_recursive(module):
        nonlocal fused_count

        # 1. Handle nn.Sequential blocks (Common pattern)
        if isinstance(module, nn.Sequential):
            i = 0
            while i < len(module) - 1:
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    # Fuse
                    fused = torch.nn.utils.fuse_conv_bn_eval(module[i], module[i + 1])
                    module[i] = fused
                    module[i + 1] = nn.Identity()
                    fused_count += 1
                    i += 2  # Skip the identity we just placed
                else:
                    _fuse_recursive(module[i])
                    i += 1
            if i < len(module):  # Handle last element if not checked
                _fuse_recursive(module[i])

        # 2. Handle generic modules with child attributes (Custom modules, Heads, etc.)
        else:
            # We look for siblings in the named_children list
            # Note: This handles cases where Conv and BN are attributes but not in a Sequential
            # For example: self.conv = Conv2d(...), self.bn = BatchNorm2d(...)

            # First, check children recursively
            for name, child in module.named_children():
                _fuse_recursive(child)

            # Then, look for sibling combinations within this module's attributes
            # This is harder as attributes don't have a fixed order unless they are in Sequential.
            # However, most architectures follow a naming convention or are Sequential.
            # We already covered Sequential. For custom modules, if they aren't Sequential,
            # we rely on the fact that if they are strictly Conv->BN, they SHOULD be fused.
            # 2a. Check for strict naming convention (e.g. self.cv1 followed by self.bn1)
            # This is a heuristic for DLA optimization where manual fusion is needed.
            children = list(module.named_children())
            i = 0
            while i < len(children) - 1:
                name1, child1 = children[i]
                name2, child2 = children[i + 1]

                # Check for Conv2d + BatchNorm2d pair
                if isinstance(child1, nn.Conv2d) and isinstance(child2, nn.BatchNorm2d):
                    # Heuristic: Check if they are likely related
                    # 1. Names are similar (e.g. cv1/bn1, conv1/bn1) OR
                    # 2. Just strictly consecutive in definition order (risky but needed for some blocks)

                    # We proceed with fusion.
                    fused = torch.nn.utils.fuse_conv_bn_eval(child1, child2)

                    # Replace in the parent module
                    setattr(module, name1, fused)
                    setattr(module, name2, nn.Identity())

                    fused_count += 1
                    i += 2  # Skip the BN we just killed
                else:
                    # Recurse if no match
                    _fuse_recursive(child1)
                    i += 1

            # Handle last child
            if i < len(children):
                _fuse_recursive(children[i][1])

    # Start recursion
    model.eval()
    _fuse_recursive(model)
    return fused_count
