import numpy as np
import torch
import torch.nn as nn


def fuse_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple:
    """
    Fold BatchNorm parameters into convolution.

    Mathematical formulation:
    W' = (γ/σ) * W
    b' = β - (μ*γ/σ)

    Args:
        conv (nn.Conv2d): Convolution layer
        bn (nn.BatchNorm2d): BatchNorm layer

    Returns:
        tuple: (fused_weight, fused_bias)
    """
    check_bn_training_mode(bn)
    w = conv.weight
    mean = bn.running_mean
    running_var = bn.running_var
    if mean is None or running_var is None:
        raise ValueError("BatchNorm layer must have running_mean and running_var for fusion.")
    var_sqrt = torch.sqrt(running_var + bn.eps)

    beta = bn.bias
    gamma = bn.weight

    if beta is None:
        beta = torch.zeros_like(mean)
    if gamma is None:
        gamma = torch.ones_like(mean)

    # Calculate fusion factor (γ/σ)
    factor = gamma / var_sqrt

    # Reshape factor for broadcasting: [C_out, 1, 1, 1]
    fused_weight = w * factor.view(-1, 1, 1, 1)

    # Calculate fused bias: β - (μ*γ/σ)
    # If conv has bias, add it scaled by factor: (b - μ) * (γ/σ) + β  => b*(γ/σ) + β - μ*(γ/σ)
    if conv.bias is not None:
        fused_bias = (conv.bias - mean) * factor + beta
    else:
        fused_bias = beta - mean * factor

    return fused_weight, fused_bias


def pad_1x1_to_nxn(kernel_1x1: torch.Tensor, n: int) -> torch.Tensor:
    """
    Zero-pad 1x1 kernel to nxn (value at center).

    Args:
        kernel_1x1: Tensor of shape [C_out, C_in, 1, 1]
        n: Target kernel size

    Returns:
        Tensor of shape [C_out, C_in, n, n]
    """
    if kernel_1x1 is None:
        return 0
    if n == 1:
        return kernel_1x1

    # Pad pixels on all sides
    p = (n - 1) // 2
    return torch.nn.functional.pad(kernel_1x1, [p, p, p, p])


def create_identity_kernel(channels: int, groups: int, kernel_size: int) -> torch.Tensor:
    """
    Create nxn identity kernel for residual connection.

    The kernel has 1 at the center for i==j (respecting groups) and 0 elsewhere.

    Args:
        channels (int): Number of input/output channels
        groups (int): Number of groups
        kernel_size (int): Size of the kernel (n)

    Returns:
        Tensor of shape [channels, channels/groups, n, n]
    """
    kernel_value = np.zeros(
        (channels, channels // groups, kernel_size, kernel_size), dtype=np.float32
    )
    center = kernel_size // 2

    for i in range(channels):
        input_idx = i % (channels // groups)
        kernel_value[i, input_idx, center, center] = 1

    return torch.from_numpy(kernel_value)


def merge_repvgg_branches(conv_dense, bn_dense, conv1x1, bn1x1, identity_bn=None):
    """
    Merge all RepVGG branches into a single equivalent convolution.

    Args:
        conv_dense, bn_dense: Dense branch (3x3 or 1x1)
        conv1x1, bn1x1: 1x1 branch
        identity_bn: BatchNorm from identity branch (optional)

    Returns:
        tuple: (merged_weight, merged_bias)
    """
    kernel_size = conv_dense.kernel_size[0]

    # 1. Get fused weights/bias for dense branch
    w_dense, b_dense = fuse_bn_into_conv(conv_dense, bn_dense)

    # 2. Get fused weights/bias for 1x1 branch
    w1, b1 = fuse_bn_into_conv(conv1x1, bn1x1)

    # Pad 1x1 weight to match dense branch size
    w1 = pad_1x1_to_nxn(w1, kernel_size)

    # Initialize total weight/bias
    w_final = w_dense + w1
    b_final = b_dense + b1

    # 3. Handle Identity branch
    if identity_bn is not None:
        check_bn_training_mode(identity_bn)
        groups = conv_dense.groups
        channels = conv_dense.in_channels

        # Create identity kernel matching the dense branch size
        id_kernel = create_identity_kernel(channels, groups, kernel_size).to(
            conv_dense.weight.device
        )

        mean = identity_bn.running_mean
        running_var = identity_bn.running_var
        if mean is None or running_var is None:
            raise ValueError(
                "Identity BatchNorm must have running_mean and running_var for fusion."
            )
        var_sqrt = torch.sqrt(running_var + identity_bn.eps)
        beta = identity_bn.bias
        gamma = identity_bn.weight

        if beta is None:
            beta = torch.zeros_like(mean)
        if gamma is None:
            gamma = torch.ones_like(mean)

        factor = gamma / var_sqrt
        b_factor = factor.view(-1, 1, 1, 1)

        w_id_fused = id_kernel * b_factor
        b_id_fused = beta - mean * factor

        # Add to total
        w_final += w_id_fused
        b_final += b_id_fused

    return w_final, b_final


def check_bn_training_mode(module: nn.Module):
    """
    Verify that BN layers are in eval mode and have valid stats before fusion.
    """
    if module.training:
        print(
            "[WARNING] Module is in training mode during fusion. "
            "Running stats (mean/var) might be unstable or untracked batches might be ignored. "
            "Call model.eval() before fusion for best results."
        )

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.running_var is not None:
            # Check if running_var is effectively uninitialized (allzeros)
            # Using allclose against zeros is more robust than sum() == 0
            if torch.allclose(module.running_var, torch.zeros_like(module.running_var)):
                print(
                    f"[WARNING] BN layer {module} appears to have uninitialized stats (all zeros). Fusion may result in identity/invalid weights."
                )
            elif module.running_var.mean() < 1e-6:
                print(
                    f"[WARNING] BN layer {module} has extremely low variance stats ({module.running_var.mean():.2e}). Check if model is trained."
                )
