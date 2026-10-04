import math

import torch
import torch.nn as nn

from unina_dla.model.backbone.repvgg_block import RepVGGBlock


class YOLO26Head(nn.Module):
    """
    Dual-head YOLO26-style detection head.

    - O2O branch: deploy branch (exported and used at runtime)
    - O2M branch: auxiliary training branch (never exported)
    """

    def __init__(self, num_classes=4, in_channels=[256, 192, 128], deploy=False):
        super(YOLO26Head, self).__init__()
        self.num_classes = num_classes
        self.deploy = deploy
        self.reg_max = 0  # direct regression, no DFL

        # Explicit strides for P3/P4/P5
        self.stride = torch.tensor([8, 16, 32])

        # Shared feature stems
        self.stems = nn.ModuleList()

        # Dedicated O2O convolution towers (deploy branch)
        self.o2o_cls_convs = nn.ModuleList()
        self.o2o_reg_convs = nn.ModuleList()

        # Dedicated O2M convolution towers (auxiliary branch)
        self.o2m_cls_convs = nn.ModuleList()
        self.o2m_reg_convs = nn.ModuleList()

        # O2O prediction heads (deploy branch)
        self.o2o_cls_preds = nn.ModuleList()
        self.o2o_reg_preds = nn.ModuleList()

        # O2M prediction heads (aux branch)
        self.o2m_cls_preds = nn.ModuleList()
        self.o2m_reg_preds = nn.ModuleList()

        for ch in in_channels:
            self.stems.append(RepVGGBlock(ch, ch, deploy=deploy))

            # Dedicated O2O towers
            self.o2o_cls_convs.append(
                nn.Sequential(
                    RepVGGBlock(ch, ch, deploy=deploy),
                    RepVGGBlock(ch, ch, deploy=deploy),
                )
            )
            self.o2o_reg_convs.append(
                nn.Sequential(
                    RepVGGBlock(ch, ch, deploy=deploy),
                    RepVGGBlock(ch, ch, deploy=deploy),
                )
            )

            # Dedicated O2M towers (auxiliary supervision)
            self.o2m_cls_convs.append(
                nn.Sequential(
                    RepVGGBlock(ch, ch, deploy=deploy),
                    RepVGGBlock(ch, ch, deploy=deploy),
                )
            )
            self.o2m_reg_convs.append(
                nn.Sequential(
                    RepVGGBlock(ch, ch, deploy=deploy),
                    RepVGGBlock(ch, ch, deploy=deploy),
                )
            )

            # O2O heads
            self.o2o_cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.o2o_reg_preds.append(
                nn.Sequential(
                    nn.Conv2d(ch, 4, 1),
                    nn.ReLU(),
                )
            )

            # O2M heads
            self.o2m_cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.o2m_reg_preds.append(
                nn.Sequential(
                    nn.Conv2d(ch, 4, 1),
                    nn.ReLU(),
                )
            )

        # Backward-compatible aliases for legacy utilities
        self.cls_convs = self.o2o_cls_convs
        self.reg_convs = self.o2o_reg_convs
        self.cls_preds = self.o2o_cls_preds
        self.reg_preds = self.o2o_reg_preds

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # RetinaNet negative prior bias: b = -log((1 - pi) / pi) for pi = 0.01 (pi ~= 1%)
        # Prevents initial BCE loss explosion on 8400 mostly-background anchors
        prior_prob = 0.01
        cls_bias_init = float(-math.log((1.0 - prior_prob) / prior_prob))
        for branch in self.o2o_cls_preds:
            if hasattr(branch, "bias") and branch.bias is not None:
                nn.init.constant_(branch.bias, cls_bias_init)
        for branch in self.o2m_cls_preds:
            if hasattr(branch, "bias") and branch.bias is not None:
                nn.init.constant_(branch.bias, cls_bias_init)

        # P3/P4/P5 priors for distance-based LTRB regression
        scale_biases = [2.0, 4.0, 4.0]
        for branch, bias in zip(self.o2o_reg_preds, scale_biases):
            nn.init.constant_(branch[0].bias, bias)
        for branch, bias in zip(self.o2m_reg_preds, scale_biases):
            nn.init.constant_(branch[0].bias, bias)

    def forward(self, feats):
        o2o_reg_outputs, o2o_cls_outputs = [], []
        o2m_reg_outputs, o2m_cls_outputs = [], []

        for i, x in enumerate(feats):
            stem_feat = self.stems[i](x)

            # Dedicated O2O tower forward
            o2o_cls_feat = self.o2o_cls_convs[i](stem_feat)
            o2o_reg_feat = self.o2o_reg_convs[i](stem_feat)
            o2o_cls_outputs.append(self.o2o_cls_preds[i](o2o_cls_feat))
            o2o_reg_outputs.append(self.o2o_reg_preds[i](o2o_reg_feat))

            if not self.deploy:
                # Dedicated O2M tower forward (auxiliary branch)
                o2m_cls_feat = self.o2m_cls_convs[i](stem_feat)
                o2m_reg_feat = self.o2m_reg_convs[i](stem_feat)
                o2m_cls_outputs.append(self.o2m_cls_preds[i](o2m_cls_feat))
                o2m_reg_outputs.append(self.o2m_reg_preds[i](o2m_reg_feat))

        if self.deploy:
            return tuple(o2o_reg_outputs), tuple(o2o_cls_outputs)

        return (
            tuple(o2o_reg_outputs),
            tuple(o2o_cls_outputs),
            tuple(o2m_reg_outputs),
            tuple(o2m_cls_outputs),
        )

    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        if self.deploy:
            return

        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True
