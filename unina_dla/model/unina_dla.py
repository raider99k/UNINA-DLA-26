import torch
import torch.nn as nn

from unina_dla.model.backbone.repvgg_backbone import RepVGGBackbone
from unina_dla.model.head.yolo26_head import YOLO26Head
from unina_dla.model.neck.rep_pan import RepPAN


def detect_architecture_version(state_dict):
    """
    Auto-detect UNINA-DLA architecture generation from state_dict keys and tensor shapes.
    - Gen 3 / 3.1: contains backbone.sppf or neck.p4_fusion.1 (spatial fusion RepVGG blocks)
    - Gen 2.5: contains stage 3 with 512 channels, neck channels [256, 192, 128], no sppf
    - Gen 2: neck channels [192, 192, 128]
    """
    if any("backbone.sppf" in k for k in state_dict.keys()) or any("p4_fusion.1" in k for k in state_dict.keys()):
        # Distinguish Gen 3 (P4=8) from Gen 3.1 (P4=14)
        if any("stages.2.13" in k for k in state_dict.keys()):
            return "3.1"
        return "3"

    for k, v in state_dict.items():
        if "neck.reduce_p3" in k and hasattr(v, "shape") and len(v.shape) > 0:
            if v.shape[0] == 256:
                return "2.5"
            elif v.shape[0] == 192:
                return "2"

    return "2.5"


class UNINA_DLA(nn.Module):
    """
    UNINA-DLA Model: RepVGG-B0 + Rep-PAN + YOLO26Head.

    Hard-Constraints for Jetson Orin DLA 2.0:
    - ALL Activations: ReLU (No SiLU).
    - No Softmax (Use Sigmoid for classes if needed, or raw logits).
    - Head: YOLO26 NMS-Free (Direct Coords, No DFL).
    - Output (deploy=True): 6-tensor tuple (reg3, reg4, reg5, cls3, cls4, cls5) for zero-copy TensorRT deployment.

    Generations:
    - version="2.5" (Default, production configuration):
        Backbone [2, 4, 14, 1], widths (64, 128, 256, 512).
        Pointwise 1x1 Rep-PAN neck with wide SOD channels [256, 192, 128].
        Deploy params: 21.26M, FLOPs: 55.4 GFLOPs, 83.78% QAT O2O mAP50.
    - version="3.1":
        Backbone [2, 4, 14, 1] + DLASPPF, widths (64, 128, 256, 256).
        Spatial Rep-PAN neck (1x1 -> 3x3 RepVGG) [256, 192, 128].
        Deploy params: 22.82M, FLOPs: 60.9 GFLOPs, 83.19% QAT O2O mAP50.
    - version="3":
        Backbone [2, 6, 8, 2] + DLASPPF, Spatial Rep-PAN neck [256, 192, 128].
    - version="2":
        Backbone [2, 4, 14, 1], Pointwise 1x1 Rep-PAN neck [192, 192, 128].
    """

    def __init__(
        self,
        num_classes=4,
        deploy=False,
        version="2.5",
        neck_channels=None,
        backbone_blocks=None,
        backbone_channels=None,
        use_sppf=None,
        spatial_fusion=None,
    ):
        super().__init__()
        self.deploy = deploy
        self.version = str(version) if version is not None else "2.5"

        # Architectural Presets by Generation
        if self.version == "2.5":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [256, 192, 128]
        elif self.version == "3.1":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 256)
            cfg_sppf = True
            cfg_spatial = True
            cfg_neck = [256, 192, 128]
        elif self.version == "3":
            cfg_blocks = (2, 6, 8, 2)
            cfg_channels = (64, 128, 256, 256)
            cfg_sppf = True
            cfg_spatial = True
            cfg_neck = [256, 192, 128]
        elif self.version == "2":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [192, 192, 128]
        else:
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [256, 192, 128]

        # Explicit overrides take precedence if provided
        if backbone_blocks is not None:
            cfg_blocks = backbone_blocks
        if backbone_channels is not None:
            cfg_channels = backbone_channels
        if use_sppf is not None:
            cfg_sppf = use_sppf
        if spatial_fusion is not None:
            cfg_spatial = spatial_fusion
        if neck_channels is not None:
            cfg_neck = neck_channels

        # Backbone
        self.backbone = RepVGGBackbone(
            deploy=deploy,
            num_blocks=cfg_blocks,
            stage_channels=cfg_channels,
            use_sppf=cfg_sppf,
        )

        # Neck
        self.neck_channels = cfg_neck
        self.neck = RepPAN(
            in_channels=self.backbone.out_channels,
            out_channels=cfg_neck,
            deploy=deploy,
            spatial_fusion=cfg_spatial,
        )

        # Head
        self.head = YOLO26Head(num_classes=num_classes, in_channels=cfg_neck, deploy=deploy)

    def forward(self, x):
        # Backbone
        features = self.backbone(x)  # [P3, P4, P5]

        # Neck
        neck_features = self.neck(features)

        # Head
        outputs = self.head(neck_features)

        if self.deploy:
            # Deployment Mode: O2O-only, fixed 6-tensor interface for zero-copy runtime
            # Validation can temporarily toggle only model.deploy without toggling
            # head.deploy. In that case head may still return 4-tuples
            # (o2o_reg, o2o_cls, o2m_reg, o2m_cls). Keep deploy path robust.
            if isinstance(outputs, tuple) and len(outputs) == 4:
                o2o_reg_outs, o2o_cls_outs = outputs[0], outputs[1]
            else:
                o2o_reg_outs, o2o_cls_outs = outputs
            o2o_cls_outs = [c.sigmoid() for c in o2o_cls_outs]
            return tuple(o2o_reg_outs) + tuple(o2o_cls_outs)

        # Training Mode:
        # (o2o_box, o2o_cls, neck_features, o2m_box, o2m_cls)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            o2o_box, o2o_cls = outputs
            return o2o_box, o2o_cls, neck_features, o2o_box, o2o_cls
        o2o_box, o2o_cls, o2m_box, o2m_cls = outputs
        return o2o_box, o2o_cls, neck_features, o2m_box, o2m_cls

    def switch_to_deploy(self):
        """
        Switch entire model to inference mode (fuse RepVGG blocks).
        """
        self.eval()
        self.deploy = True
        self.backbone.switch_to_deploy()
        self.neck.switch_to_deploy()
        self.head.switch_to_deploy()


if __name__ == "__main__":
    # Smoke test
    model = UNINA_DLA(num_classes=4, deploy=False)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print("Training Output shapes:")
    for i, (o2o_cls, o2o_box, o2m_cls, o2m_box) in enumerate(
        zip(output[1], output[0], output[4], output[3])
    ):
        print(
            f"Scale {i}: O2O Cls {o2o_cls.shape}, O2O Box {o2o_box.shape}, O2M Cls {o2m_cls.shape}, O2M Box {o2m_box.shape}"
        )

    print("\nSwitching to deploy...")
    model.switch_to_deploy()
    output_deploy = model(dummy_input)
    print("Deploy successful.")
