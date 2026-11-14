import torch
import torch.nn as nn

from unina_dla.model.backbone.repvgg_block import RepVGGBlock


class DLAUpsample(nn.Module):
    """
    DLA-friendly Upsampling using ConvTranspose2d (Deconvolution).
    Generic Resize/Upsample layers often trigger GPU fallbacks on DLA.
    Use an ungrouped deconvolution because DLA does not accept grouped/depthwise
    ConvTranspose in strict residency mode.
    """

    def __init__(self, channels, scale=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=scale, stride=scale, groups=1, bias=False
        )
        # Initialize weights to an identity-preserving bilinear upsample.
        with torch.no_grad():
            self.upsample.weight.copy_(self.get_bilinear_kernel(scale))

    def get_bilinear_kernel(self, scale):
        """
        Generate an amplitude-preserving diagonal kernel for ConvTranspose2d(kernel_size=scale, stride=scale).
        Because kernel_size == stride with zero padding (required for strict DLA 2.0 residency),
        output tiles are non-overlapping: each output pixel receives signal from exactly one input pixel.
        Initializing each kH x kW spatial entry on the channel diagonal to 1.0 preserves exact 1.0x
        feature amplitude at step 0 (preventing the 4x signal attenuation of overlapping bilinear formulas)
        while allowing all 4 sub-pixel quadrants and cross-channel weights to be learned during training.
        """
        filt = torch.ones((scale, scale), dtype=torch.float32)

        # ConvTranspose2d stores weights as [in_channels, out_channels / groups, kH, kW].
        # Filling only the diagonal preserves per-channel identity behavior at initialization
        # while keeping groups=1 for DLA compatibility.
        weight = torch.zeros((self.upsample.in_channels, self.upsample.out_channels, scale, scale))
        for i in range(min(self.upsample.in_channels, self.upsample.out_channels)):
            weight[i, i, :, :] = filt
        return weight

    def forward(self, x):
        return self.upsample(x)


class RepPAN(nn.Module):
    """
    Re-parameterizable Path Aggregation Network (Rep-PAN).

    DLA Optimizations:
    - Uses concatenation instead of element-wise addition (more efficient on DLA)
    - Deconvolution-based upsampling (ensures DLA residency)
    - RepVGG blocks for fusion at inference

    Args:
        in_channels (list): [P3, P4, P5] input channels from backbone
        out_channels (int or list): Output channel count. Can be:
            - int: Uniform channels for all levels (e.g., 256)
            - list: Per-level channels [P3, P4, P5] (e.g., [128, 256, 256] for Slim P3)
        deploy (bool): Enable inference mode
    """

    def __init__(
        self,
        in_channels=[128, 256, 256],
        out_channels=[256, 192, 128],
        deploy=False,
        spatial_fusion=True,
        p5_groups=None,
        upsample_mode="nearest",
    ):
        super(RepPAN, self).__init__()
        self.deploy = deploy
        self.spatial_fusion = spatial_fusion
        self.p5_groups = p5_groups
        self.upsample_mode = str(upsample_mode).lower() if upsample_mode is not None else "nearest"

        # P3, P4, P5 input channels from backbone
        c3, c4, c5 = in_channels

        # Normalize out_channels to list format (backward compatible)
        if isinstance(out_channels, int):
            out_channels = [out_channels, out_channels, out_channels]
        c3_out, c4_out, c5_out = out_channels

        # Store for reference
        self.out_channels = out_channels

        def make_fusion(in_c, out_c):
            if spatial_fusion:
                return nn.Sequential(
                    RepVGGBlock(in_c, out_c, kernel_size=1, deploy=deploy),
                    RepVGGBlock(out_c, out_c, kernel_size=3, deploy=deploy),
                )
            return RepVGGBlock(in_c, out_c, kernel_size=1, groups=1, deploy=deploy)

        # ============================================================
        # TOP-DOWN PATHWAY (Large → Small, P5 → P4 → P3)
        # ============================================================

        # Reduce P5 to c5_out channels
        # Gen 2.6 uses p5_groups=1 (hardware-verified CBUF residency: wtBanks=1, minDataBanks=2 <= 16)
        # Gen 2.5 uses p5_groups=2 for legacy 1MB theoretical weight limit compliance
        if p5_groups is not None:
            eff_p5_groups = p5_groups
        else:
            eff_p5_groups = 2 if (c5 >= 512 and c5 % 2 == 0) else 1
        self.reduce_p5 = RepVGGBlock(c5, c5_out, groups=eff_p5_groups, deploy=deploy)

        # Reduce P4 to c4_out channels
        self.reduce_p4 = RepVGGBlock(c4, c4_out, deploy=deploy)

        # Upsample P5 (c5_out channels) for concat with P4
        # Gen 2.6 uses nn.Upsample(nearest) which runs natively on Orin DLA 2.0 with zero parameters
        # Gen 2.5 and legacy models use DLAUpsample (ConvTranspose2d with bilinear kernel)
        if self.upsample_mode == "nearest":
            self.upsample_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        elif self.upsample_mode == "bilinear":
            self.upsample_p5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.upsample_p5 = DLAUpsample(c5_out)

        # Fuse P4: concat(P4_reduced, P5_upsampled) → c4_out
        # Input: c4_out + c5_out, Output: c4_out
        self.p4_fusion = make_fusion(c4_out + c5_out, c4_out)

        # Reduce P3 to c3_out channels
        self.reduce_p3 = RepVGGBlock(c3, c3_out, deploy=deploy)

        # Upsample P4_fused (c4_out channels) for concat with P3
        if self.upsample_mode == "nearest":
            self.upsample_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        elif self.upsample_mode == "bilinear":
            self.upsample_p4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.upsample_p4 = DLAUpsample(c4_out)

        # Fuse P3: concat(P3_reduced, P4_upsampled) → c3_out
        # Input: c3_out + c4_out, Output: c3_out
        self.p3_fusion = make_fusion(c3_out + c4_out, c3_out)

        # ============================================================
        # BOTTOM-UP PATHWAY (Small → Large, P3 → P4 → P5)
        # ============================================================

        # Downsample P3_out (c3_out) for concat with P4_fused
        self.downsample_p3 = RepVGGBlock(c3_out, c3_out, stride=2, deploy=deploy)

        # Fuse N3 (becomes final P4): concat(P3_down, P4_fused) → c4_out
        # Input: c3_out + c4_out, Output: c4_out
        self.n3_fusion = make_fusion(c3_out + c4_out, c4_out)

        # Downsample N3_out (c4_out) for concat with P5_reduced
        self.downsample_p4 = RepVGGBlock(c4_out, c4_out, stride=2, deploy=deploy)

        # Fuse N4 (becomes final P5): concat(N3_down, P5_reduced) → c5_out
        # Input: c4_out + c5_out, Output: c5_out
        self.n4_fusion = make_fusion(c4_out + c5_out, c5_out)

    def forward(self, inputs):
        # inputs: [P3, P4, P5]
        p3, p4, p5 = inputs

        # --- Top-Down ---
        p5_red = self.reduce_p5(p5)
        p5_up = self.upsample_p5(p5_red)
        p4_red = self.reduce_p4(p4)

        # Concat P4 and P5_up
        p4_cat = torch.cat([p4_red, p5_up], dim=1)
        p4_fused = self.p4_fusion(p4_cat)

        p4_up = self.upsample_p4(p4_fused)
        p3_red = self.reduce_p3(p3)

        # Concat P3 and P4_up
        p3_cat = torch.cat([p3_red, p4_up], dim=1)
        p3_out = self.p3_fusion(p3_cat)  # This is the final P3 output

        # --- Bottom-Up ---
        p3_down = self.downsample_p3(p3_out)

        # Concat P3_down and P4_fused (from top-down phase)
        # Note: Standard PANet uses the "laterals" from top-down.
        n3_cat = torch.cat([p3_down, p4_fused], dim=1)
        n3_out = self.n3_fusion(n3_cat)  # This is the final P4 output

        n3_down = self.downsample_p4(n3_out)

        # Concat n3_down and p5_red
        n4_cat = torch.cat([n3_down, p5_red], dim=1)
        n4_out = self.n4_fusion(n4_cat)  # This is the final P5 output

        # Return [P3, P4, P5]
        return (p3_out, n3_out, n4_out)

    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        if self.deploy:
            return

        for module in self.modules():
            if module is not self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True
