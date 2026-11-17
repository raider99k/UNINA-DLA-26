"""
2:4 Structured Sparsity Utilities for UNINA-DLA.

This module provides utilities to enable NVIDIA's 2:4 structured sparsity pattern
for accelerated inference on Jetson Orin DLA 2.0.

The 2:4 pattern enforces exactly 2 zeros in every group of 4 contiguous weights,
enabling ~2x theoretical speedup on Sparse Tensor Cores and DLA.

Supports two backends:
1. NVIDIA APEX ASP (preferred, if available)
2. Pure PyTorch fallback (works on any PyTorch 2.0+ system)

Usage:
    from unina_dla.utils.sparsity_utils import enable_sparsity, verify_sparsity

    # After distillation training completes:
    enable_sparsity(model, optimizer)

    # Fine-tune with distillation for recovery
    # ...

    # Before export, verify sparsity pattern:
    is_valid, report = verify_sparsity(model)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Try to import APEX ASP
try:
    from apex.contrib.sparsity import ASP

    HAS_ASP = True
except ImportError:
    HAS_ASP = False
    ASP = None

# Public capability flag consumed by training engines.
# This module provides sparsity utilities even without APEX (PyTorch fallback).
HAS_SPARSITY = True

# Accuracy-sensitive and structural layers to exclude from 2:4 pruning by default:
# - Backbone stem has C_in=3 (cannot form 2:4 pattern without destroying RGB input channels)
# - DLAUpsample uses ConvTranspose2d with bilinear kernels that must not be pruned
# - Final O2O/O2M predictors are precision-critical
DEFAULT_SPARSE_IGNORE_PATTERNS = (
    "backbone.stem",
    "stem",
    "upsample",
    "DLAUpsample",
    "head.o2o_cls_preds",
    "head.o2o_reg_preds",
    "head.o2m_cls_preds",
    "head.o2m_reg_preds",
    "head.cls_preds",
    "head.reg_preds",
    "cls_preds",
    "reg_preds",
)


def _is_excluded_param(param_name: str, ignore_param_patterns: Optional[List[str]]) -> bool:
    """Return True when a parameter should be excluded from sparsification."""
    if not ignore_param_patterns:
        return False
    return any(pattern in param_name for pattern in ignore_param_patterns)


def check_asp_available() -> bool:
    """Check if NVIDIA APEX ASP is available."""
    return HAS_ASP


def _apply_2_4_mask_to_tensor(weight: torch.Tensor) -> torch.Tensor:
    """
    Apply 2:4 structured sparsity mask to a weight tensor.

    For Conv2d: weight shape is (C_out, C_in, H, W)
    For Linear: weight shape is (Out, In)

    The 2:4 pattern is applied along the last dimension after reshaping.
    NVIDIA applies it along C_in for Conv2d, so we permute accordingly.

    Args:
        weight: The weight tensor to sparsify

    Returns:
        The sparsified weight tensor (same shape, 50% zeros in 2:4 pattern)
    """
    original_dtype = weight.dtype

    # Work in float32 for numerical stability
    weight = weight.float()

    if weight.dim() == 4:  # Conv2d: (C_out, C_in, H, W)
        # NVIDIA 2:4 applies along C_in dimension
        # Permute to (C_out, H, W, C_in), apply mask, permute back
        c_out, c_in, h, w = weight.shape

        # Layers with fewer than 4 input channels (e.g. 3-channel RGB stem) cannot be 2:4 pruned
        if c_in < 4:
            return weight.to(original_dtype)

        # Handle case where C_in is not divisible by 4
        if c_in % 4 != 0:
            # Pad C_in to next multiple of 4, then truncate after masking
            pad_size = (4 - c_in % 4) % 4
            weight_padded = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_size))
            c_in_padded = c_in + pad_size
        else:
            weight_padded = weight
            c_in_padded = c_in
            pad_size = 0

        # Reshape for 2:4 masking: (C_out * H * W, C_in / 4, 4)
        weight_reshaped = weight_padded.permute(0, 2, 3, 1).contiguous()
        weight_reshaped = weight_reshaped.view(-1, c_in_padded // 4, 4)

    elif weight.dim() == 2:  # Linear: (Out, In)
        out_features, in_features = weight.shape

        if in_features % 4 != 0:
            pad_size = (4 - in_features % 4) % 4
            weight_padded = torch.nn.functional.pad(weight, (0, pad_size))
            in_features_padded = in_features + pad_size
        else:
            weight_padded = weight
            in_features_padded = in_features
            pad_size = 0

        weight_reshaped = weight_padded.view(out_features, in_features_padded // 4, 4)

    else:
        # Return unchanged for other dimensions
        return weight.to(original_dtype)

    # Apply 2:4 mask: keep top 2 magnitude values per group of 4
    abs_weights = weight_reshaped.abs()

    # Get indices of top 2 values in each group
    _, top2_indices = abs_weights.topk(2, dim=-1)

    # Create mask
    mask = torch.zeros_like(weight_reshaped)
    mask.scatter_(-1, top2_indices, 1.0)

    # Apply mask
    weight_masked = weight_reshaped * mask

    # Reshape back to original layout
    if weight.dim() == 4:
        weight_out = weight_masked.view(c_out, h, w, c_in_padded)
        weight_out = weight_out.permute(0, 3, 1, 2).contiguous()
        if pad_size > 0:
            weight_out = weight_out[:, :c_in, :, :]
    else:
        weight_out = weight_masked.view(out_features, in_features_padded)
        if pad_size > 0:
            weight_out = weight_out[:, :in_features]

    return weight_out.to(original_dtype)


def _prune_model_pytorch(
    model: nn.Module, verbosity: int = 1, ignore_param_patterns: Optional[List[str]] = None
) -> None:
    """
    Apply 2:4 structured sparsity using pure PyTorch (no APEX required).

    This function modifies weights in-place to enforce 2:4 sparsity pattern.
    """
    layer_stats = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2 or "weight" not in name:
                continue
            if _is_excluded_param(name, ignore_param_patterns):
                continue

            # Skip small layers (not worth pruning)
            if param.numel() < 16:
                continue

            # Apply 2:4 mask
            sparse_weight = _apply_2_4_mask_to_tensor(param.data)
            param.data.copy_(sparse_weight)

            # Stats
            zeros = (param.data == 0).sum().item()
            total = param.numel()
            sparsity = zeros / total * 100

            if verbosity >= 2:
                layer_stats.append((name, total, total - zeros, sparsity))

    if verbosity >= 2 and layer_stats:
        print("\nPer-Layer Sparsity:")
        print(f"{'Layer':<50} {'Total':>10} {'NonZero':>10} {'Sparsity':>8}")
        print("-" * 80)
        for name, total, nonzero, sparsity in layer_stats:
            short_name = name[-48:] if len(name) > 48 else name
            print(f"{short_name:<50} {total:>10,} {nonzero:>10,} {sparsity:>7.1f}%")


def enable_sparsity(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    verbosity: int = 1,
    ignore_param_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """
    Enable 2:4 structured sparsity on a model.

    Uses NVIDIA APEX ASP if available, otherwise falls back to pure PyTorch.

    This function:
    1. Analyzes the model to find prunable layers (Conv2d, Linear)
    2. Computes importance scores for each weight
    3. Prunes 2 out of every 4 weights (lowest magnitude)
    4. For APEX: Registers forward hooks to maintain sparse masks during training
    5. For PyTorch: Returns masks for manual enforcement via create_zero_clamp_hook()

    Args:
        model: The PyTorch model to sparsify
        optimizer: The optimizer (APEX needs this to register mask hooks)
        verbosity: 0=silent, 1=summary, 2=per-layer details

    Returns:
        The sparsified model (same object, modified in-place)
    """
    if ignore_param_patterns is None:
        ignore_param_patterns = list(DEFAULT_SPARSE_IGNORE_PATTERNS)

    # Count params before sparsity
    total_params_before = sum(p.numel() for p in model.parameters())
    conv_params_before = 0
    for n, p in model.named_parameters():
        if "weight" in n and p.dim() >= 2 and not _is_excluded_param(n, ignore_param_patterns):
            conv_params_before += p.numel()

    if verbosity >= 1:
        print("[Sparsity] Enabling 2:4 structured sparsity...")
        print(f"  Total Parameters: {total_params_before:,} | Prunable (Conv/Linear): {conv_params_before:,}")
        if ignore_param_patterns:
            if verbosity >= 2:
                print(f"  Ignored Parameters (name contains): {ignore_param_patterns}")
            else:
                print(f"  Ignored Patterns: {len(ignore_param_patterns)} rules (stem, upsample, final pred heads)")

    # Choose backend.
    # Name-based layer exclusions are not guaranteed in ASP, so we switch to the
    # PyTorch path when exclusions are requested to preserve O2O-sensitive heads.
    using_asp = HAS_ASP and not ignore_param_patterns
    if using_asp:
        if verbosity >= 1:
            print("  Backend: NVIDIA APEX ASP (optimized)")
        ASP.prune_trained_model(model, optimizer)
    else:
        if verbosity >= 1:
            if HAS_ASP and ignore_param_patterns:
                print("  Backend: Pure PyTorch (enforcing layer-level exclusions over APEX ASP)")
            else:
                print("  Backend: Pure PyTorch (2:4 mask enforced via zero-clamp hook)")
        _prune_model_pytorch(model, verbosity, ignore_param_patterns)

    # Count effective (non-zero) params after sparsity
    nonzero_params = 0
    total_pruned = 0
    layer_stats = []

    for name, param in model.named_parameters():
        is_prunable = (
            param.dim() >= 2
            and "weight" in name
            and not _is_excluded_param(name, ignore_param_patterns)
        )
        if is_prunable:
            zeros = (param == 0).sum().item()
            total = param.numel()
            nonzero = total - zeros
            sparsity = zeros / total * 100
            nonzero_params += nonzero
            total_pruned += zeros

            if verbosity >= 2 and using_asp:  # PyTorch backend already printed
                layer_stats.append((name, total, nonzero, sparsity))
        else:
            if param.dim() >= 2:
                nonzero_params += (param != 0).sum().item()
            else:
                nonzero_params += param.numel()

    if verbosity >= 2 and using_asp and layer_stats:
        print("\nPer-Layer Sparsity:")
        print(f"{'Layer':<50} {'Total':>10} {'NonZero':>10} {'Sparsity':>8}")
        print("-" * 80)
        for name, total, nonzero, sparsity in layer_stats:
            short_name = name[-48:] if len(name) > 48 else name
            print(f"{short_name:<50} {total:>10,} {nonzero:>10,} {sparsity:>7.1f}%")

    if verbosity >= 1:
        effective_sparsity = (
            total_pruned / conv_params_before * 100 if conv_params_before > 0 else 0
        )
        print("\n[Sparsity Summary]")
        print(f"  > Pruned Weights: {total_pruned:,} ({effective_sparsity:.1f}%)")
        print(
            f"  > Effective Parameters: {nonzero_params:,} ({nonzero_params / total_params_before * 100:.1f}% of original)"
        )
        print(f"  > Expected DLA Speedup: ~{100 / (100 - effective_sparsity / 2):.1f}x theoretical")
        if not using_asp and verbosity >= 2:
            print(
                "  > [!] CRITICAL: Call create_zero_clamp_hook() and invoke after each optimizer.step()"
            )

    return model


def verify_sparsity(
    model: nn.Module, strict: bool = True, ignore_param_patterns: Optional[List[str]] = None
) -> Tuple[bool, Dict]:
    """
    Verify that all prunable layers maintain the 2:4 sparsity pattern.

    This should be called before ONNX export to ensure the sparse pattern
    wasn't corrupted during fine-tuning (e.g., by optimizer momentum drift).

    IMPORTANT: NVIDIA 2:4 sparsity is applied along the Input Channel (C_in)
    dimension for Conv2d layers. For verification, we must permute the weight
    to (C_out, H, W, C_in) before grouping by 4.

    Args:
        model: The model to verify
        strict: If True, requires EXACT 2:4 pattern (2 zeros per 4 weights).
                If False, allows >= 2 zeros per 4 (more sparse is OK).

    Returns:
        Tuple of (is_valid, report_dict)
        - is_valid: True if all layers pass verification
        - report_dict: Per-layer verification results
    """
    if ignore_param_patterns is None:
        ignore_param_patterns = list(DEFAULT_SPARSE_IGNORE_PATTERNS)

    report = {
        "valid": True,
        "layers": {},
        "summary": {"total_groups": 0, "valid_groups": 0, "invalid_groups": 0},
    }

    for name, param in model.named_parameters():
        if param.dim() < 2 or "weight" not in name:
            continue
        if _is_excluded_param(name, ignore_param_patterns):
            report["layers"][name] = {"status": "skipped", "reason": "excluded_by_policy"}
            continue

        weight = param.data.detach()

        # NVIDIA 2:4 Sparsity applies to the Input Channel (Dim 1)
        # For Conv2d: (C_out, C_in, H, W) -> permute to (C_out, H, W, C_in)
        if weight.dim() == 4:  # Conv2d
            # Check if C_in (dim 1) is divisible by 4
            if weight.shape[1] % 4 != 0:
                report["layers"][name] = {
                    "status": "skipped",
                    "reason": f"C_in={weight.shape[1]} not divisible by 4",
                }
                continue
            # Permute to (Out, H, W, In) so groups of 4 are taken from C_in
            weight_t = weight.permute(0, 2, 3, 1).contiguous()
        elif weight.dim() == 2:  # Linear: (Out, In)
            # Check if In (dim 1) is divisible by 4
            if weight.shape[1] % 4 != 0:
                report["layers"][name] = {
                    "status": "skipped",
                    "reason": f"In={weight.shape[1]} not divisible by 4",
                }
                continue
            weight_t = weight
        else:
            # Skip other dimensions
            report["layers"][name] = {
                "status": "skipped",
                "reason": f"unsupported dim={weight.dim()}",
            }
            continue

        flat = weight_t.view(-1, 4)
        num_groups = flat.shape[0]

        # Count zeros per group
        zeros_per_group = (flat == 0).sum(dim=1)

        if strict:
            # Exactly 2 zeros per group
            valid_groups = (zeros_per_group == 2).sum().item()
        else:
            # At least 2 zeros per group
            valid_groups = (zeros_per_group >= 2).sum().item()

        invalid_groups = num_groups - valid_groups
        layer_valid = invalid_groups == 0

        report["layers"][name] = {
            "valid": layer_valid,
            "total_groups": num_groups,
            "valid_groups": valid_groups,
            "invalid_groups": invalid_groups,
            "sparsity": (flat == 0).float().mean().item() * 100,
        }

        # Check DLA 32-byte channel alignment
        c_in = weight.shape[1]
        if c_in % 32 != 0:
            report["layers"][name]["dla_warning"] = f"C_in={c_in} not 32-aligned (DLA may pad)"

        report["summary"]["total_groups"] += num_groups
        report["summary"]["valid_groups"] += valid_groups
        report["summary"]["invalid_groups"] += invalid_groups

        if not layer_valid:
            report["valid"] = False

    return report["valid"], report


def create_zero_clamp_hook(
    model: nn.Module, ignore_param_patterns: Optional[List[str]] = None
) -> callable:
    """
    Create an optimizer hook that re-enforces zeros after each update.

    During training with sparsity, optimizer momentum can cause pruned (zero)
    weights to drift to small non-zero values. This hook clamps them back to zero
    after each optimizer.step().

    CRITICAL for PyTorch fallback: APEX ASP handles this automatically via its
    optimizer wrapper. When using Pure PyTorch sparsity, you MUST call this hook
    after every optimizer.step() to maintain the 2:4 pattern.

    Usage:
        hook = create_zero_clamp_hook(model)
        # After each optimizer.step():
        optimizer.step()
        hook()  # Re-enforce zeros

    Returns:
        A callable hook function
    """
    if ignore_param_patterns is None:
        ignore_param_patterns = list(DEFAULT_SPARSE_IGNORE_PATTERNS)

    # Capture sparse masks only for parameters subject to pruning.
    masks = {}
    for name, param in model.named_parameters():
        if (
            param.dim() >= 2
            and "weight" in name
            and not _is_excluded_param(name, ignore_param_patterns)
        ):
            masks[name] = (param.data != 0).float()

    def zero_clamp_hook(optimizer=None) -> None:
        """Clamp pruned weights back to zero."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data.mul_(masks[name].to(param.device))

    return zero_clamp_hook


def create_zero_grad_hook(
    model: nn.Module, ignore_param_patterns: Optional[List[str]] = None
) -> callable:
    """
    Create an optimizer hook that zeroes GRADIENTS of pruned weights before update.

    This is critical to prevent optimizer momentum/velocity buffers from accumulating
    non-zero values for pruned weights, which would cause them to drift away from zero
    in subsequent steps even if the weights themselves are clamped.

    Usage:
        grad_hook = create_zero_grad_hook(model)
        # Before optimizer.step():
        grad_hook()
        optimizer.step()

    Returns:
        A callable hook function
    """
    if ignore_param_patterns is None:
        ignore_param_patterns = list(DEFAULT_SPARSE_IGNORE_PATTERNS)

    # Capture the sparse mask at hook creation time
    masks = {}
    for name, param in model.named_parameters():
        if (
            param.dim() >= 2
            and "weight" in name
            and not _is_excluded_param(name, ignore_param_patterns)
        ):
            # We want to identify the ZEROS in the current weight
            # The mask should be 1 where weight is != 0, and 0 where weight is 0
            masks[name] = (param.data != 0).float()

    def zero_grad_hook() -> None:
        """Clamp gradients of pruned weights to zero."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks and param.grad is not None:
                    # Multiply grad by mask (0 where weight should be zero)
                    param.grad.mul_(masks[name].to(param.device))

    return zero_grad_hook


def get_sparse_model_summary(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 640, 640), device: str = "cpu"
) -> Dict:
    """
    Get detailed model summary including sparsity statistics.

    Args:
        model: The model to analyze
        input_size: Input tensor shape for FLOP calculation
        device: Device to run analysis on

    Returns:
        Dictionary with params, GFLOPs, sparsity info
    """
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count non-zero parameters (effective params after sparsity)
    nonzero_params = 0
    conv_total = 0
    conv_nonzero = 0

    for name, param in model.named_parameters():
        if param.dim() >= 2:
            nz = (param != 0).sum().item()
            nonzero_params += nz
            conv_total += param.numel()
            conv_nonzero += nz
        else:
            nonzero_params += param.numel()

    # Estimate GFLOPs (approximate via forward hooks)
    # This is a simplified estimation; for accurate count use thop or fvcore
    try:
        from thop import profile

        dummy = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = None  # thop not installed

    sparsity = (1 - conv_nonzero / conv_total) * 100 if conv_total > 0 else 0

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "effective_params": nonzero_params,
        "sparsity_percent": sparsity,
        "gflops": gflops,
        "conv_params": {"total": conv_total, "nonzero": conv_nonzero},
    }

    return summary


def print_sparse_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cpu",
    title: str = "Model Summary",
) -> None:
    """
    Print a formatted model summary with sparsity statistics.

    Args:
        model: The model to analyze
        input_size: Input tensor shape
        device: Device to run analysis on
        title: Header title for the summary
    """
    summary = get_sparse_model_summary(model, input_size, device)

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Input Size: {input_size}")
    print(f"Total Parameters:      {summary['total_params']:>12,}")
    print(f"Trainable Parameters:  {summary['trainable_params']:>12,}")
    print(f"Effective Parameters:  {summary['effective_params']:>12,} (non-zero)")
    print(f"Weight Sparsity:       {summary['sparsity_percent']:>11.1f}%")

    if summary["gflops"] is not None:
        # For sparse models, effective GFLOPs is reduced by sparsity ratio
        effective_gflops = summary["gflops"] * (1 - summary["sparsity_percent"] / 100)
        print(f"GFLOPs (dense):        {summary['gflops']:>12.2f}")
        print(f"GFLOPs (effective):    {effective_gflops:>12.2f}")
    else:
        print("GFLOPs: (install 'thop' for FLOP counting)")

    print(f"{'=' * 60}\n")
