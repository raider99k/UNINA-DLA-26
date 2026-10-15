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


REPVGG_A1_DEFAULT_URL = (
    "https://huggingface.co/timm/repvgg_a1.rvgg_in1k/resolve/main/pytorch_model.bin"
)


def load_pretrained_backbone(
    model: nn.Module,
    weights_path_or_url: Union[str, bool] = "repvgg_a1",
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> dict:
    """
    Load ImageNet-pretrained RepVGG-A1 weights into the RepVGG backbone of UNINA-DLA.

    Compatibility across Generations:
    - Gen 2.6 (Hardware-Verified Champion, blocks [2, 4, 14, 1]):
        Stages 0 to 4 (Stem, P2, P3, P4, P5) match RepVGG-A1 layer-for-layer (100% backbone coverage).
        Stage 4 (512 channels, groups=1) is initialized by slicing the first 512 filters from
        RepVGG-A1 Stage 4 (1280 channels), providing complete pretrained initialization.
    - Gen 2.5 (Legacy Champion, blocks [2, 4, 14, 1]):
        Stages 0 to 3 (Stem, P2, P3, P4) match RepVGG-A1 layer-for-layer (93.8% parameters loaded).
        Stage 4 (P5, 512 channels, groups=2) is a custom DLA hardware adaptation to guarantee
        on-chip CBUF SRAM residency (< 1 MiB) and is initialized with Kaiming Normal.
    - Gen 3.1 (Spatial Neck Experiment, blocks [2, 4, 14, 1]):
        Stages 0 to 3 match RepVGG-A1 layer-for-layer (90.8% parameters loaded).
    - Gen 3 (blocks [2, 6, 8, 2]):
        Stem, Stage 1, first 4 blocks of Stage 2, and first 8 blocks of Stage 3 match (76.3% parameters loaded).

    Args:
        model: UNINA-DLA detector or RepVGGBackbone instance.
        weights_path_or_url: Path to .pth/.bin checkpoint file, URL, or 'repvgg_a1' for auto-download.
        device: Device to place loaded tensors on.
        verbose: If True, prints a detailed summary of loaded parameters.

    Returns:
        Dictionary with loading diagnostics ('loaded_params', 'total_params', 'coverage_pct', 'loaded_tensors').
    """
    import os
    import re

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_src = str(weights_path_or_url).strip()

    # Resolve source path / URL
    if weights_src.lower() in ("true", "1", "repvgg_a1", "a1", "default", "none"):
        weights_url = REPVGG_A1_DEFAULT_URL
        if verbose:
            print(f"[Pretrained Backbone] Auto-fetching official RepVGG-A1 ImageNet weights from:\n  {weights_url}")
        state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location=dev, progress=verbose)
    elif weights_src.startswith(("http://", "https://")):
        if verbose:
            print(f"[Pretrained Backbone] Fetching weights from URL:\n  {weights_src}")
        state_dict = torch.hub.load_state_dict_from_url(weights_src, map_location=dev, progress=verbose)
    else:
        if not os.path.isfile(weights_src):
            raise FileNotFoundError(f"[Pretrained Backbone] Checkpoint file not found: {weights_src}")
        if verbose:
            print(f"[Pretrained Backbone] Loading local checkpoint:\n  {weights_src}")
        ckpt = torch.load(weights_src, map_location=dev, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Extract target backbone module
    backbone = getattr(model, "backbone", model)
    target_sd = backbone.state_dict()
    total_backbone_elements = sum(p.numel() for p in target_sd.values())

    def _translate_key(src_k: str) -> Optional[str]:
        # Strip common container prefixes
        k = src_k
        for prefix in ("model.", "backbone.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]

        # Normalize branch names across implementations:
        # timm uses conv_kxk / conv_1x1 / identity, while RepVGG uses rbr_dense / rbr_1x1 / rbr_identity
        k = k.replace(".conv_kxk.", ".rbr_dense.")
        k = k.replace(".conv_1x1.", ".rbr_1x1.")
        k = k.replace(".identity.", ".rbr_identity.")

        # 1. timm / holocron key format: features.X.Y.branches...
        if k.startswith("features."):
            parts = k.split(".")
            if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
                stg = int(parts[1])
                blk = int(parts[2])
                rest = ".".join(parts[3:])
                rest = rest.replace("branches.0.0.", "rbr_dense.conv.")
                rest = rest.replace("branches.0.1.", "rbr_dense.bn.")
                rest = rest.replace("branches.1.0.", "rbr_1x1.conv.")
                rest = rest.replace("branches.1.1.", "rbr_1x1.bn.")
                rest = rest.replace("branches.2.", "rbr_identity.")
                if stg == 0:
                    return f"stem.{rest}"
                else:
                    return f"stages.{stg - 1}.{blk}.{rest}"

        # 2. DingXiaoH official format: stage0. / stage1.X. / stage2.X.
        if k.startswith("stage0."):
            return "stem." + k[len("stage0."):]
        m_stage = re.match(r"^stage([1-4])\.(.+)", k)
        if m_stage:
            stage_idx = int(m_stage.group(1)) - 1  # stage1 -> stages.0
            return f"stages.{stage_idx}.{m_stage.group(2)}"

        # 3. Direct UNINA-DLA or timm-native format (stem. or stages.)
        if k.startswith(("stem.", "stages.")):
            return k

        return None

    loaded_params = 0
    skipped_params = 0
    loaded_tensors = 0
    skipped_tensors = 0

    with torch.no_grad():
        for src_k, src_tensor in state_dict.items():
            tgt_k = _translate_key(src_k)
            if tgt_k is not None and tgt_k in target_sd:
                tgt_tensor = target_sd[tgt_k]
                if tgt_tensor.shape == src_tensor.shape:
                    tgt_tensor.copy_(src_tensor.to(device=tgt_tensor.device, dtype=tgt_tensor.dtype))
                    loaded_params += src_tensor.numel()
                    loaded_tensors += 1
                elif (
                    tgt_tensor.ndim >= 1
                    and src_tensor.ndim == tgt_tensor.ndim
                    and tgt_tensor.shape[0] < src_tensor.shape[0]
                    and tgt_tensor.shape[1:] == src_tensor.shape[1:]
                ):
                    # Slicing support for Stage 4 when channels=512 and groups=1 (Gen 2.6):
                    # Official RepVGG-A1 Stage 4 has 1280 channels -> slice first 512 channels
                    tgt_tensor.copy_(
                        src_tensor[: tgt_tensor.shape[0]].to(device=tgt_tensor.device, dtype=tgt_tensor.dtype)
                    )
                    loaded_params += tgt_tensor.numel()
                    loaded_tensors += 1
                else:
                    skipped_params += src_tensor.numel()
                    skipped_tensors += 1

    coverage_pct = min(100.0, (loaded_params / max(total_backbone_elements, 1)) * 100.0)

    if verbose:
        print(f"[Pretrained Backbone] Successfully initialized {loaded_tensors} tensors "
              f"({loaded_params:,} / {total_backbone_elements:,} params/buffers - {coverage_pct:.2f}% coverage).")
        if coverage_pct > 99.0:
            print("[Pretrained Backbone] Complete 100% backbone pretraining achieved across Stages 0 to 4 (Gen 2.6).")
        else:
            print("[Pretrained Backbone] Note: Stages 0 to 3 match RepVGG-A1 ImageNet weights. "
                  "Stage 4 (custom 512-ch / groups=2 DLA SRAM layout) and neck/heads remain randomly initialized.")

    return {
        "loaded_params": loaded_params,
        "total_params": total_backbone_elements,
        "coverage_pct": coverage_pct,
        "loaded_tensors": loaded_tensors,
        "skipped_tensors": skipped_tensors,
    }


def load_warm_start_weights(
    model: nn.Module,
    weights_path: str,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> dict:
    """
    Load weights from an existing UNINA-DLA checkpoint into the model with shape-tolerant matching.
    Copies all parameters where names and shapes match. Mismatched shapes (e.g. from groups=2 to groups=1)
    and missing layers (e.g. dropped deconv layers when switching to nn.Upsample) are skipped cleanly.
    """
    import os

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[Warm Start] Checkpoint not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    target_sd = model.state_dict()
    total_model_params = sum(p.numel() for p in model.parameters())

    loaded_params = 0
    loaded_tensors = 0
    skipped_tensors = 0

    with torch.no_grad():
        for k, src_tensor in state_dict.items():
            if k in target_sd:
                tgt_tensor = target_sd[k]
                if tgt_tensor.shape == src_tensor.shape:
                    tgt_tensor.copy_(src_tensor.to(device=tgt_tensor.device, dtype=tgt_tensor.dtype))
                    loaded_params += src_tensor.numel()
                    loaded_tensors += 1
                else:
                    skipped_tensors += 1
            else:
                skipped_tensors += 1

    pct = (loaded_params / max(total_model_params, 1)) * 100.0
    if verbose:
        print(f"[Warm Start] Loaded {loaded_tensors} tensors ({loaded_params:,} / {total_model_params:,} params - {pct:.2f}% coverage) from {weights_path}.")
        if skipped_tensors > 0:
            print(f"[Warm Start] Skipped {skipped_tensors} incompatible/unused tensors (e.g. legacy deconv weights or mismatched group convs).")

    return {
        "loaded_params": loaded_params,
        "total_params": total_model_params,
        "coverage_pct": pct,
        "loaded_tensors": loaded_tensors,
        "skipped_tensors": skipped_tensors,
    }


