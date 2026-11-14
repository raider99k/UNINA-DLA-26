import torch.nn as nn

from unina_dla.utils.repvgg_fusion import merge_repvgg_branches


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """
    RepVGG Block with DLA-compatible ReLU activation.

    Training: Multi-branch (3x3 + 1x1 + identity) for better gradients
    Inference: Single fused 3x3 conv for DLA efficiency

    Attributes:
        deploy (bool): If True, use fused inference mode
        groups (int): Groups for depthwise separable (default=1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size

        # Auto-calculate padding based on kernel size
        padding = kernel_size // 2

        # STRICTLY ReLU for DLA compatibility
        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            # Identity branch: only if stride is 1 and shapes match.
            # Note: For kernel_size=1, we keep it as a simple residual.
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

        self._init_weights()

    def _init_weights(self):
        # Note: nn.Conv2d already initializes with kaiming_uniform_(a=sqrt(5), mode='fan_in'),
        # which keeps per-branch variance at ~1/3 so the 3-branch sum (3x3 + 1x1 + identity)
        # preserves unit variance across deep stacks even before BN running stats warm up.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out_1x1 = self.rbr_1x1(inputs) if self.rbr_1x1 is not None else 0
        return self.nonlinearity(self.rbr_dense(inputs) + out_1x1 + id_out)

    def switch_to_deploy(self):
        """Fuse branches into single conv for inference."""
        if hasattr(self, "rbr_reparam"):
            return

        conv1x1 = self.rbr_1x1.conv if self.rbr_1x1 is not None else None
        bn1x1 = self.rbr_1x1.bn if self.rbr_1x1 is not None else None

        kernel, bias = merge_repvgg_branches(
            self.rbr_dense.conv,
            self.rbr_dense.bn,
            conv1x1,
            bn1x1,
            self.rbr_identity,
        )

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches to save memory
        self.__delattr__("rbr_dense")
        if hasattr(self, "rbr_1x1"):
            self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True
