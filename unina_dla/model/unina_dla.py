import torch
import torch.nn as nn

from unina_dla.model.backbone.repvgg_backbone import RepVGGBackbone
from unina_dla.model.head.yolo26_head import YOLO26Head
from unina_dla.model.neck.rep_pan import RepPAN


def detect_architecture_version(state_dict):
    """
    Auto-detect UNINA-DLA architecture generation from state_dict keys and tensor shapes.
    - Gen 3.2: contains backbone.sppf or neck.p4_fusion.1, Stage 4 with 512 channels, deconv/nearest upsampling, unbounded reg
    - Gen 3 / 3.1: contains backbone.sppf or neck.p4_fusion.1 (spatial fusion RepVGG blocks) with 256-ch Stage 4
    - Gen 2.6: contains stage 3 with 512 channels, neck channels [256, 192, 128], stage4 groups=1
    - Gen 2.5: contains stage 3 with 512 channels, neck channels [256, 192, 128], stage4 groups=2
    - Gen 2: neck channels [192, 192, 128]
    """
    if any("backbone.sppf" in k for k in state_dict.keys()) or any("p4_fusion.1" in k for k in state_dict.keys()):
        # Check if Stage 4 has 512 channels -> Gen 3.2
        for sk, sv in state_dict.items():
            if getattr(sv, "ndim", 0) >= 2:
                if ("stages.3.0" in sk and ("rbr_dense.conv.weight" in sk or "rbr_reparam.weight" in sk or sk.endswith("stages.3.0.weight"))) and sv.shape[0] == 512:
                    return "3.2"
                if "backbone.sppf.cv1" in sk and ("rbr_dense.conv.weight" in sk or "rbr_reparam.weight" in sk) and sv.shape[1] == 512:
                    return "3.2"
                if "neck.reduce_p5" in sk and ("rbr_dense.conv.weight" in sk or "rbr_reparam.weight" in sk) and sv.shape[1] == 512:
                    return "3.2"
        # Distinguish Gen 3 (P4=8) from Gen 3.1 (P4=14)
        if any("stages.2.13" in k for k in state_dict.keys()):
            return "3.1"
        return "3"

    for k, v in state_dict.items():
        if "neck.reduce_p3" in k and hasattr(v, "shape") and len(v.shape) > 0:
            if v.shape[0] == 256:
                # Distinguish Gen 2.6 (groups=1) from Gen 2.5 (groups=2)
                for sk, sv in state_dict.items():
                    if ("stages.3.0" in sk and ("rbr_dense.conv.weight" in sk or "rbr_reparam.weight" in sk)) or \
                       ("stages.3.0.weight" in sk):
                        if getattr(sv, "ndim", 0) >= 2:
                            if sv.shape[1] == 256:
                                return "2.6"
                            elif sv.shape[1] == 128:
                                return "2.5"
                    if "neck.reduce_p5" in sk and ("rbr_dense.conv.weight" in sk or "rbr_reparam.weight" in sk):
                        if getattr(sv, "ndim", 0) >= 2:
                            if sv.shape[1] == 512:
                                return "2.6"
                            elif sv.shape[1] == 256:
                                return "2.5"
                return "2.5"
            elif v.shape[0] == 192:
                return "2"

    return "2.5"


def detect_upsample_mode(state_dict):
    """
    Auto-detect upsample mode ('nearest' vs 'deconv') from state_dict keys.
    If deconv weights are present ('neck.upsample_p5.upsample.weight'), returns 'deconv'.
    Otherwise returns 'nearest'.
    """
    for k in state_dict.keys():
        if "neck.upsample" in k and "weight" in k:
            return "deconv"
    return "nearest"



class UNINA_DLA(nn.Module):
    """
    UNINA-DLA Model: RepVGG-A1 (adapted) + Rep-PAN + YOLO26Head.

    Hard-Constraints for Jetson Orin DLA 2.0:
    - ALL Activations: ReLU (No SiLU).
    - No Softmax (Use Sigmoid for classes if needed, or raw logits).
    - Head: YOLO26 NMS-Free (Direct Coords, No DFL).
    - Output (deploy=True): 6-tensor tuple (reg3, reg4, reg5, cls3, cls4, cls5) for zero-copy TensorRT deployment.

    Generations:
    - version="3.2" (DLA 2.0 SOTA Architecture):
        Backbone [2, 4, 14, 1] + DLASPPF, widths (64, 128, 256, 512), groups=1 throughout.
        Spatial 3x3 Rep-PAN neck with wide SOD channels [256, 192, 128] + learnable DLA deconv upsampling (compatible with nearest).
        Unbounded direct LTRB regression in YOLO26Head (no hard ReLU clipping on sub-stride cones).
        Full 100% ImageNet RepVGG-A1 pretraining coverage across all 4 stages with L1-norm filter selection.
    - version="2.6" (Hardware-Verified Champion):
        Backbone [2, 4, 14, 1], widths (64, 128, 256, 512), groups=1 throughout.
        Pointwise 1x1 Rep-PAN neck with wide SOD channels [256, 192, 128], reduce_p5 groups=1.
        Zero-parameter DLA-native nearest upsampling (nn.Upsample) replacing deconvolution.
        Full ImageNet pretraining coverage across all 4 backbone stages (no scratch Stage 4).
        Deploy params: 21.93M (nearest) / 22.14M (deconv), FLOPs: 56.0 GFLOPs.
    - version="2.5" (Legacy Production Champion):
        Backbone [2, 4, 14, 1], widths (64, 128, 256, 512), groups=2 on Stage 4.
        Pointwise 1x1 Rep-PAN neck with wide SOD channels [256, 192, 128], reduce_p5 groups=2.
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
        stage4_groups=None,
        p5_groups=None,
        upsample_mode=None,
        reg_relu=None,
    ):
        super().__init__()
        self.deploy = deploy
        self.version = str(version) if version is not None else "2.5"

        # Architectural Presets by Generation
        if self.version == "3.2":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = True
            cfg_spatial = True
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 1
            cfg_p5_groups = 1
            cfg_upsample_mode = "deconv"
            cfg_reg_relu = False
        elif self.version == "2.6":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 1
            cfg_p5_groups = 1
            cfg_upsample_mode = "nearest"
            cfg_reg_relu = True
        elif self.version == "2.5":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 2
            cfg_p5_groups = 2
            cfg_upsample_mode = "deconv"
            cfg_reg_relu = True
        elif self.version == "3.1":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 256)
            cfg_sppf = True
            cfg_spatial = True
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 1
            cfg_p5_groups = 1
            cfg_upsample_mode = "deconv"
            cfg_reg_relu = True
        elif self.version == "3":
            cfg_blocks = (2, 6, 8, 2)
            cfg_channels = (64, 128, 256, 256)
            cfg_sppf = True
            cfg_spatial = True
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 1
            cfg_p5_groups = 1
            cfg_upsample_mode = "deconv"
            cfg_reg_relu = True
        elif self.version == "2":
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [192, 192, 128]
            cfg_stage4_groups = 2
            cfg_p5_groups = 2
            cfg_upsample_mode = "deconv"
            cfg_reg_relu = True
        else:
            cfg_blocks = (2, 4, 14, 1)
            cfg_channels = (64, 128, 256, 512)
            cfg_sppf = False
            cfg_spatial = False
            cfg_neck = [256, 192, 128]
            cfg_stage4_groups = 1
            cfg_p5_groups = 1
            cfg_upsample_mode = "nearest"
            cfg_reg_relu = True

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
        if stage4_groups is not None:
            cfg_stage4_groups = stage4_groups
        if p5_groups is not None:
            cfg_p5_groups = p5_groups
        if upsample_mode is not None:
            cfg_upsample_mode = str(upsample_mode).lower()
        if reg_relu is not None:
            cfg_reg_relu = bool(reg_relu)

        self.upsample_mode = cfg_upsample_mode
        self.reg_relu = cfg_reg_relu

        # Backbone
        self.backbone = RepVGGBackbone(
            deploy=deploy,
            num_blocks=cfg_blocks,
            stage_channels=cfg_channels,
            use_sppf=cfg_sppf,
            stage4_groups=cfg_stage4_groups,
        )

        # Neck
        self.neck_channels = cfg_neck
        self.neck = RepPAN(
            in_channels=self.backbone.out_channels,
            out_channels=cfg_neck,
            deploy=deploy,
            spatial_fusion=cfg_spatial,
            p5_groups=cfg_p5_groups,
            upsample_mode=cfg_upsample_mode,
        )

        # Head
        self.head = YOLO26Head(
            num_classes=num_classes,
            in_channels=cfg_neck,
            deploy=deploy,
            reg_relu=cfg_reg_relu,
        )

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
