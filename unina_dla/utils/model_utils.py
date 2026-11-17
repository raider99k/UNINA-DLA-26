import json
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

    def _fuse_recursive(module):
        nonlocal fused_count

        # 1. Handle nn.Sequential blocks
        if isinstance(module, nn.Sequential):
            i = 0
            while i < len(module) - 1:
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    fused = torch.nn.utils.fuse_conv_bn_eval(module[i], module[i + 1])
                    module[i] = fused
                    module[i + 1] = nn.Identity()
                    fused_count += 1
                    i += 2
                else:
                    _fuse_recursive(module[i])
                    i += 1
            if i < len(module):
                _fuse_recursive(module[i])

        # 2. Handle custom containers with sibling Conv2d + BatchNorm2d attributes
        else:
            for _, child in module.named_children():
                _fuse_recursive(child)

            children = list(module.named_children())
            i = 0
            while i < len(children) - 1:
                name1, child1 = children[i]
                name2, child2 = children[i + 1]

                if isinstance(child1, nn.Conv2d) and isinstance(child2, nn.BatchNorm2d):
                    fused = torch.nn.utils.fuse_conv_bn_eval(child1, child2)
                    setattr(module, name1, fused)
                    setattr(module, name2, nn.Identity())
                    fused_count += 1
                    i += 2
                else:
                    i += 1

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

    translated_sd = {}
    for src_k, src_tensor in state_dict.items():
        tgt_k = _translate_key(src_k)
        if tgt_k is not None:
            translated_sd[tgt_k] = src_tensor

    # Precompute consistent top-k channel indices by L1-norm importance for any stage block
    # where pretrained output channels exceed target channels (e.g. Stage 4: 1280 -> 512 in Gen 2.6 / Gen 3.2).
    channel_slice_indices = {}
    with torch.no_grad():
        for tgt_k, src_tensor in translated_sd.items():
            if tgt_k.endswith(".rbr_dense.conv.weight") and tgt_k in target_sd:
                tgt_tensor = target_sd[tgt_k]
                if (
                    src_tensor.ndim == 4
                    and tgt_tensor.ndim == 4
                    and tgt_tensor.shape[0] < src_tensor.shape[0]
                    and tgt_tensor.shape[1:] == src_tensor.shape[1:]
                ):
                    block_prefix = tgt_k[: -len("rbr_dense.conv.weight")]
                    k_out = tgt_tensor.shape[0]
                    # Compute L1 importance across (C_in, kH, kW)
                    importance = src_tensor.detach().float().abs().mean(dim=(1, 2, 3))
                    bn_w_key = block_prefix + "rbr_dense.bn.weight"
                    bn_v_key = block_prefix + "rbr_dense.bn.running_var"
                    if bn_w_key in translated_sd and bn_v_key in translated_sd:
                        bn_w = translated_sd[bn_w_key].detach().float().abs()
                        bn_v = translated_sd[bn_v_key].detach().float().clamp(min=0.0)
                        if bn_w.shape[0] == importance.shape[0] and bn_v.shape[0] == importance.shape[0]:
                            importance = importance * (bn_w / torch.sqrt(bn_v + 1e-5))
                    top_idx = torch.topk(importance, k=k_out, largest=True).indices.sort().values
                    channel_slice_indices[block_prefix] = top_idx

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
                    # Importance-ranked channel selection for Stage 4 when channels=512 and groups=1 (Gen 2.6 / Gen 3.2)
                    matched_idx = None
                    for prefix, idx_tensor in channel_slice_indices.items():
                        if tgt_k.startswith(prefix) and idx_tensor.numel() == tgt_tensor.shape[0]:
                            matched_idx = idx_tensor.to(device=src_tensor.device)
                            break
                    if matched_idx is not None:
                        sliced = src_tensor.index_select(0, matched_idx)
                    else:
                        sliced = src_tensor[: tgt_tensor.shape[0]]
                    tgt_tensor.copy_(sliced.to(device=tgt_tensor.device, dtype=tgt_tensor.dtype))
                    loaded_params += tgt_tensor.numel()
                    loaded_tensors += 1
                else:
                    skipped_params += src_tensor.numel()
                    skipped_tensors += 1

    coverage_pct = min(100.0, (loaded_params / max(total_backbone_elements, 1)) * 100.0)

    if verbose:
        print(f"[Pretrained Backbone] Successfully initialized {loaded_tensors} tensors "
              f"({loaded_params:,} / {total_backbone_elements:,} params/buffers - {coverage_pct:.2f}% coverage).")
        if coverage_pct > 99.0 or len(channel_slice_indices) > 0:
            print("[Pretrained Backbone] Complete Stages 0 to 4 RepVGG-A1 ImageNet pretraining loaded (L1-norm filter selection on Stage 4).")
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
    Prefers EMA weights if present in checkpoint. Copies all parameters where names and shapes match
    (including cross-generation translation of 1x1 neck fusion blocks to spatial fusion block 0).
    Mismatched shapes (e.g. from groups=2 to groups=1) and missing layers are skipped cleanly.
    """
    import os

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[Warm Start] Checkpoint not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
            state_dict = ckpt["ema"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    target_sd = model.state_dict()
    total_model_params = sum(p.numel() for p in model.parameters())

    loaded_params = 0
    loaded_tensors = 0
    skipped_tensors = 0

    with torch.no_grad():
        for raw_k, src_tensor in state_dict.items():
            k = raw_k.replace("_orig_mod.", "")
            tgt_k = k if k in target_sd else None
            if tgt_k is None and "_fusion.rbr_" in k:
                k_alt = k.replace("_fusion.rbr_", "_fusion.0.rbr_")
                if k_alt in target_sd:
                    tgt_k = k_alt
            elif tgt_k is None and "_fusion.0.rbr_" in k:
                k_alt = k.replace("_fusion.0.rbr_", "_fusion.rbr_")
                if k_alt in target_sd:
                    tgt_k = k_alt

            if tgt_k is not None:
                tgt_tensor = target_sd[tgt_k]
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


def strip_engine_metadata(raw: bytes) -> bytes:
    """Strip Ultralytics' leading [4-byte little-endian len][JSON dict] prefix if present."""
    if len(raw) < 8:
        return raw
    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
    if meta_len <= 0 or (4 + meta_len) >= len(raw):
        return raw
    try:
        meta = json.loads(raw[4 : 4 + meta_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    if not isinstance(meta, dict):
        return raw
    return raw[4 + meta_len :]



