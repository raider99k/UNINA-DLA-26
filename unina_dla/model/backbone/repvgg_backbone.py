import torch
import torch.nn as nn

from .repvgg_block import RepVGGBlock


class DLASPPF(nn.Module):
    """
    DLA-Native Spatial Pyramid Pooling Fast (SPPF).

    Hardware Compliance for Jetson Orin DLA 2.0:
    - MaxPool2d(5, stride=1, padding=2) maps 100% to DLA Planar Data Processor (PDP).
    - 1x1 convolutions map to DLA SDP/MAC arrays.
    - Concat INT8 Scale Invariance: All branches (x, y1, y2, y3) share identical
      quantization dynamic range to satisfy NVDLA DMA zero-requantization Concat rule.
    """

    def __init__(self, c1, c2, k=5, deploy=False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = RepVGGBlock(c1, c_, kernel_size=1, deploy=deploy)
        self.cv2 = RepVGGBlock(c_ * 4, c2, kernel_size=1, deploy=deploy)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.deploy = deploy
        self.shared_quantizer = None

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        if self.shared_quantizer is not None:
            # Force identical INT8 scale across all branches to guarantee zero DLA fallback
            x = self.shared_quantizer(x)
            y1 = self.shared_quantizer(y1)
            y2 = self.shared_quantizer(y2)
            y3 = self.shared_quantizer(y3)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

    def switch_to_deploy(self):
        self.deploy = True
        self.cv1.switch_to_deploy()
        self.cv2.switch_to_deploy()


class RepVGGBackbone(nn.Module):
    """
    RepVGG Backbone optimized for DLA (ReLU only).
    Supports standard RepVGG-A1 block layout [2, 4, 14, 1] as well as rebalanced racing depths
    and optional DLA-native SPPF pooling.
    """

    def __init__(
        self,
        deploy=False,
        out_indices=(1, 2, 3),
        num_blocks=(2, 4, 14, 1),
        stage_channels=(64, 128, 256, 256),
        use_sppf=True,
        stage4_groups=None,
    ):
        super().__init__()
        self.deploy = deploy
        self.out_indices = out_indices
        self.use_sppf = use_sppf
        self.num_blocks = num_blocks
        self.stage_channels = stage_channels

        self.stages = nn.ModuleList()
        self.in_planes = stage_channels[0]

        # Stage 0: Stem
        self.stem = RepVGGBlock(3, stage_channels[0], stride=2, deploy=deploy)

        # Stage 1: layer1 (P2)
        layer1 = self._make_layer(stage_channels[0], num_blocks[0], stride=2, deploy=deploy)
        self.stages.append(layer1)

        # Stage 2: layer2 (P3)
        layer2 = self._make_layer(stage_channels[1], num_blocks[1], stride=2, deploy=deploy)
        self.stages.append(layer2)

        # Stage 3: layer3 (P4)
        layer3 = self._make_layer(stage_channels[2], num_blocks[2], stride=2, deploy=deploy)
        self.stages.append(layer3)

        # Stage 4: layer4 (P5)
        # Gen 2.6 uses stage4_groups=1 (hardware-verified CBUF residency: wtBanks=1, minDataBanks=2 <= 16)
        # Gen 2.5 uses stage4_groups=2 for legacy 1MB theoretical weight limit compliance
        if stage4_groups is not None:
            groups = stage4_groups
        else:
            groups = 2 if stage_channels[3] >= 512 else 1
        layer4 = self._make_layer(
            stage_channels[3], num_blocks[3], stride=2, deploy=deploy, groups=groups
        )
        self.stages.append(layer4)

        # Optional DLA-native SPPF on final stage
        if use_sppf:
            self.sppf = DLASPPF(stage_channels[3], stage_channels[3], k=5, deploy=deploy)
        else:
            self.sppf = None

        # Define output channels for P3, P4, P5
        self.out_channels = [stage_channels[1], stage_channels[2], stage_channels[3]]

    def _make_layer(self, planes, num_blocks, stride, deploy, groups=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                RepVGGBlock(self.in_planes, planes, stride=s, deploy=deploy, groups=groups)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # self.stages contains [Stage1(P2), Stage2(P3), Stage3(P4), Stage4(P5)]
            # We want P3, P4, P5 -> indices 1, 2, 3
            if i in self.out_indices:
                if i == 3 and self.sppf is not None:
                    outs.append(self.sppf(x))
                else:
                    outs.append(x)
        return outs

    def switch_to_deploy(self):
        self.deploy = True
        self.stem.switch_to_deploy()
        for stage in self.stages:
            for block in stage:
                block.switch_to_deploy()
        if self.sppf is not None:
            self.sppf.switch_to_deploy()
