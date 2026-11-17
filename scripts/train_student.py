import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

from unina_dla.engines.common import validate_dual_head_state_dict
from unina_dla.engines.phase1_5_sparsity import run_sparsity_finetune
from unina_dla.engines.phase1_classic import run_classic_training
from unina_dla.engines.phase1_distill import run_distillation
from unina_dla.engines.phase2_qat import run_qat
from unina_dla.engines.phase3_export import run_export

AUGMENTATION_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "auto_augment",
    "erasing",
    "close_mosaic",
)


def apply_yaml_augmentations(
    data_cfg, data_cfg_dict=None, yaml_path=None, close_mosaic_override=None
):
    """Merge augmentation hyperparameters from dataset YAML into Ultralytics data_cfg namespace."""
    merged_yaml = {}
    if yaml_path and os.path.exists(yaml_path):
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_yaml = yaml.safe_load(f)
            if isinstance(raw_yaml, dict):
                merged_yaml.update(raw_yaml)
        except Exception:
            pass
    if isinstance(data_cfg_dict, dict):
        merged_yaml.update(data_cfg_dict)

    applied = {}
    for k in AUGMENTATION_KEYS:
        if k in merged_yaml and merged_yaml[k] is not None:
            val = merged_yaml[k]
            if k == "auto_augment" and isinstance(val, str) and val.strip().lower() == "none":
                val = None
            setattr(data_cfg, k, val)
            applied[k] = val

    # If auto_augment is disabled and erasing wasn't explicitly set in YAML, disable random erasing
    if getattr(data_cfg, "auto_augment", None) is None and "erasing" not in applied:
        data_cfg.erasing = 0.0
        applied["erasing"] = 0.0

    if close_mosaic_override is not None:
        data_cfg.close_mosaic = int(close_mosaic_override)
        applied["close_mosaic"] = int(close_mosaic_override)

    return applied


def calculate_finetune_epochs(base_epochs: int, pct: float = 0.10) -> int:
    """Proportional epoch scheduling for fine-tuning phases with min 1, max 30 guards."""
    raw_pct = int(base_epochs * pct)
    return max(1, min(raw_pct, 30))


def main():
    parser = argparse.ArgumentParser(
        description="UNINA-DLA Training Pipeline: Distillation -> Sparsity -> QAT -> Export"
    )
    parser.set_defaults(verify_dla=True)

    # Dataset & Teacher
    parser.add_argument(
        "--data",
        type=str,
        default="unina_dla/config/unina_dla_data.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--teacher", type=str, default=None, help="Path to teacher model .pt")
    parser.add_argument("--imgsz", type=str, default="640", help="Image size: 640 or 768,480")
    parser.add_argument(
        "--imgsz_hw",
        type=int,
        nargs=2,
        default=None,
        help="[DEPRECATED] Image size (h w) for rectangular training. Use --imgsz instead.",
    )
    parser.add_argument(
        "--formula_image",
        action="store_true",
        default=False,
        help="Activate static rectangular tensor padding strictly to --imgsz_hw (e.g. 480x768) with dataset shuffling enabled across epochs",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Num workers")
    parser.add_argument("--device", type=str, default="0", help="Device (0, cpu)")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        choices=["3.2", "2.6", "2.5", "3.1", "2", "3"],
        help="Architecture generation (defaults to 3.2 for fresh training, or auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--upsample_mode",
        type=str,
        default=None,
        choices=["nearest", "deconv", "bilinear"],
        help="Upsampling mode in Rep-PAN neck (defaults to architecture preset: deconv for 3.2/2.5, nearest for 2.6)",
    )

    # Phase 1: Distillation
    parser.add_argument("--epochs", type=int, default=150, help="Phase 1 Epochs")
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="Disable Mosaic/Mixup/CopyPaste for the final N epochs of Phase 1 (set 0 to disable)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Phase 1 linear warmup epochs (defaults to 3, capped by total epochs; set 0 to disable)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning Rate (defaults to 0.01 for MuSGD/SGD, 0.001 for AdamW)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="MuSGD",
        choices=["MuSGD", "musgd", "AdamW", "adamw", "SGD", "sgd"],
        help="Phase 1 optimizer: MuSGD (default), AdamW, or SGD",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay (defaults to 1e-4 for AdamW, 5e-4 for MuSGD/SGD)",
    )
    parser.add_argument(
        "--enable_distillation", action="store_true", help="Enable Knowledge Distillation"
    )
    parser.add_argument(
        "--skip_distillation", action="store_true", help="Skip Phase 1 (load existing checkpoint)"
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help='Run in "Classic" Mode (Baseline training, no Distillation, no fancy loss)',
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume Phase 1 from checkpoint")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Initial checkpoint to warm-start student weights (shape-tolerant matching)",
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=str,
        default=None,
        nargs="?",
        const="repvgg_a1",
        help="Initialize RepVGG backbone with ImageNet pretrained weights (e.g. --pretrained_backbone, --pretrained_backbone repvgg_a1, or path to checkpoint). Default: None (train from scratch).",
    )
    parser.add_argument(
        "--backbone_lr_ratio",
        type=float,
        default=1.0,
        help="Learning rate ratio for the backbone relative to base --lr (e.g. 0.1 for 10x lower backbone LR). Defaults to 1.0 (uniform full rate; empirical testing shows uniform LR outperforms 0.1x differential LR).",
    )

    # Fade Schedule (Teacher Fade)
    parser.add_argument(
        "--fade_start", type=float, default=0.8, help="Teacher Fade Start (1.0=Full Strength)"
    )
    parser.add_argument(
        "--fade_end", type=float, default=0.1, help="Teacher Fade End (Decays to this minimum)"
    )
    parser.add_argument(
        "--task_weight_start",
        type=float,
        default=0.35,
        help="Task loss weight at distillation start (teacher-dominant if < 1.0)",
    )
    parser.add_argument(
        "--task_weight_end",
        type=float,
        default=1.0,
        help="Task loss weight at distillation end (GT authority)",
    )

    # O2O Assignment Curriculum (training-only; no deployment behavior change)
    parser.add_argument(
        "--o2o_topk_start", type=int, default=3, help="Early-phase TAL top-k for denser supervision"
    )
    parser.add_argument("--o2o_topk_mid", type=int, default=2, help="Mid-phase TAL top-k")
    parser.add_argument(
        "--o2o_topk_end", type=int, default=1, help="Final-phase TAL top-k (strict O2O)"
    )
    parser.add_argument(
        "--o2o_topk_stage1_pct",
        type=float,
        default=0.15,
        help="Progress threshold for switching start->mid top-k",
    )
    parser.add_argument(
        "--o2o_topk_stage2_pct",
        type=float,
        default=0.35,
        help="Progress threshold for switching mid->end top-k",
    )
    parser.add_argument(
        "--o2m_aux_weight_start", type=float, default=0.8, help="Aux O2M task weight at phase start"
    )
    parser.add_argument(
        "--o2m_aux_weight_end", type=float, default=0.0, help="Aux O2M task weight at phase end (0.0 for pure late O2O)"
    )
    parser.add_argument(
        "--o2m_aux_weight_end_distill",
        type=float,
        default=None,
        help="Aux O2M task weight at distillation phase end (defaults to --o2m_aux_weight_end or 0.0)",
    )
    parser.add_argument(
        "--o2m_aux_weight_sparse_qat",
        type=float,
        default=0.0,
        help="Aux O2M task weight during sparsity/QAT (set 0.0 for strict late O2O)",
    )
    parser.add_argument(
        "--o2m_topk",
        type=int,
        default=10,
        help="TaskAlignedAssigner top-k for O2M auxiliary branch",
    )
    parser.add_argument(
        "--dup_loss_weight",
        type=float,
        default=0.05,
        help="Weight for O2O duplicate suppression ranking loss (set 0 to disable)",
    )
    parser.add_argument(
        "--dup_iou_thr",
        type=float,
        default=0.6,
        help="IoU threshold to mark negative anchors as local duplicates",
    )
    parser.add_argument(
        "--dup_margin",
        type=float,
        default=0.2,
        help="Ranking margin: positive logit should exceed duplicate negatives by this amount",
    )
    parser.add_argument(
        "--dup_topk",
        type=int,
        default=20,
        help="Max duplicate negatives sampled per GT for ranking loss",
    )
    parser.add_argument(
        "--hard_bg_loss_weight",
        type=float,
        default=0.02,
        help="Weight for hard background suppression (set 0 to disable)",
    )
    parser.add_argument(
        "--hard_bg_iou_thr",
        type=float,
        default=0.2,
        help="Max IoU to any GT for hard-background negatives",
    )
    parser.add_argument(
        "--hard_bg_topk", type=int, default=64, help="Top-k hardest background negatives per image"
    )
    parser.add_argument(
        "--dup_start_pct",
        type=float,
        default=0.6,
        help="Training progress threshold to start ramping duplicate suppression",
    )
    parser.add_argument(
        "--kd_mask_topk",
        type=int,
        default=1,
        help="Top-k used to build teacher KD masks (recommended: 1 for strict O2O)",
    )
    parser.add_argument(
        "--kd_assigner_parity",
        action="store_true",
        help="Use Ultralytics-style one2one assigner settings for KD masks",
    )
    parser.add_argument(
        "--kd_parity_topk",
        type=int,
        default=7,
        help="KD parity assigner top-k (Ultralytics one2one default)",
    )
    parser.add_argument(
        "--kd_parity_topk2",
        type=int,
        default=1,
        help="KD parity assigner top-k2 (Ultralytics one2one default)",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=300,
        help="Maximum detections per image in validation postprocess",
    )
    parser.add_argument(
        "--deploy_eval_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for deployment-style O2O vs NMS validation",
    )
    parser.add_argument(
        "--deploy_eval_iou",
        type=float,
        default=0.6,
        help="IoU threshold for deployment-style NMS validation",
    )
    parser.add_argument(
        "--tb_log_every",
        type=int,
        default=10,
        help="TensorBoard scalar logging interval in optimizer steps",
    )
    parser.add_argument(
        "--tb_detail_every",
        type=int,
        default=50,
        help="TensorBoard interval for detailed diagnostic scalars",
    )

    # Per-Vector KD Component Weights (relative to global teacher fade)
    parser.add_argument(
        "--kd_sdf_start", type=float, default=1.0, help="SDF KD multiplier at phase start"
    )
    parser.add_argument(
        "--kd_sdf_end", type=float, default=1.0, help="SDF KD multiplier at phase end"
    )
    parser.add_argument(
        "--kd_logit_start", type=float, default=1.0, help="Logit KD multiplier at phase start"
    )
    parser.add_argument(
        "--kd_logit_end", type=float, default=0.9, help="Logit KD multiplier at phase end"
    )
    parser.add_argument(
        "--kd_box_start", type=float, default=1.0, help="Box KD multiplier at phase start"
    )
    parser.add_argument(
        "--kd_box_end", type=float, default=0.5, help="Box KD multiplier at phase end"
    )
    parser.add_argument(
        "--kd_mech_all",
        action="store_true",
        help="Enable all KD mechanisms (cross-head + global relation)",
    )
    parser.add_argument(
        "--kd_mech_cross_head",
        action="store_true",
        help="Enable CrossKD-style routed teacher-head supervision",
    )
    parser.add_argument(
        "--kd_cross_head_weight",
        type=float,
        default=0.15,
        help="Weight for routed cross-head KD terms",
    )
    parser.add_argument(
        "--kd_mech_global_rel",
        action="store_true",
        help="Enable global relation distillation inside SDF KD",
    )
    parser.add_argument(
        "--kd_global_rel_weight",
        type=float,
        default=0.10,
        help="Weight for SDF global relation term",
    )
    parser.add_argument(
        "--kd_global_rel_tokens",
        type=int,
        default=256,
        help="Max spatial tokens used for global relation pooling",
    )

    # Phase 1.5: Sparsity
    parser.add_argument(
        "--enable_sparsity", action="store_true", help="Enable 2:4 Structured Sparsity (NVIDIA ASP)"
    )
    parser.add_argument(
        "--epochs_sparsity",
        type=int,
        default=None,
        help="Sparsity Fine-tuning Epochs (Default: 10%% of Phase 1, clipped [1, 30])",
    )
    parser.add_argument(
        "--skip_sparsity", action="store_true", help="Skip Sparsity Phase (if resuming QAT)"
    )
    parser.add_argument(
        "--resume_sparsity", type=str, default=None, help="Resume Sparsity from checkpoint"
    )
    parser.add_argument("--bn_batches", type=int, default=200, help="Batches for BN Re-estimation")

    # Phase 2: QAT
    parser.add_argument("--enable_qat", action="store_true", help="Enable QAT")
    parser.add_argument(
        "--epochs_qat",
        type=int,
        default=None,
        help="QAT Epochs (Default: 5%% of Phase 1, clipped [1, 30])",
    )
    parser.add_argument("--resume_qat", type=str, default=None, help="Resume QAT from checkpoint")
    parser.add_argument("--calib_batches", type=int, default=200, help="QAT Calibration Batches")
    parser.add_argument(
        "--qat-clean-deploy",
        action="store_true",
        help="Export a clean deploy ONNX from the QAT checkpoint by stripping fake-quant/QDQ state",
    )

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="unina_dla_student",
        help="Output project directory (can be absolute path for Colab/Drive)",
    )
    parser.add_argument(
        "--verify-dla",
        action="store_true",
        dest="verify_dla",
        help="Run strict DLA ONNX verification after export (default: enabled)",
    )
    parser.add_argument(
        "--no-verify-dla",
        action="store_false",
        dest="verify_dla",
        help="Skip strict DLA ONNX verification after export",
    )

    parser.add_argument(
        "--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)"
    )
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument(
        "--compile", action="store_true", default=True, help="Enable torch.compile (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--no-compile", action="store_false", dest="compile", help="Disable torch.compile"
    )

    # Early Stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs for Phase 1 (0 disables early stopping)",
    )
    parser.add_argument(
        "--patience_sparsity",
        type=int,
        default=None,
        help="Early stopping patience in epochs for Phase 1.5 Sparsity (defaults to disabled)",
    )
    parser.add_argument(
        "--patience_qat",
        type=int,
        default=None,
        help="Early stopping patience in epochs for Phase 2 QAT (defaults to disabled)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum validation metric improvement required to reset early stopping patience",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="mAP50",
        choices=["mAP50", "mAP50-95", "fitness"],
        help="Validation metric monitored by early stopping (fitness = 0.1*mAP50 + 0.9*mAP50-95)",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=None,
        help="Minimum Phase 1 epochs before early stopping patience begins counting (defaults to warmup epochs)",
    )

    args = parser.parse_args()
    args.patience = max(0, int(args.patience or 0))
    if args.patience_sparsity is not None:
        args.patience_sparsity = max(0, int(args.patience_sparsity))
    if args.patience_qat is not None:
        args.patience_qat = max(0, int(args.patience_qat))
    args.min_delta = max(0.0, float(args.min_delta or 0.0))
    if args.early_stop_min_epochs is not None:
        args.early_stop_min_epochs = max(0, int(args.early_stop_min_epochs))
    args._o2m_aux_weight_end_distill_explicit = args.o2m_aux_weight_end_distill is not None

    if args.kd_mech_all:
        args.kd_mech_cross_head = True
        args.kd_mech_global_rel = True

    if args.optimizer is None:
        args.optimizer = "MuSGD"
    args.optimizer = str(args.optimizer).strip()

    if args.lr is None:
        args.lr = 0.001 if args.optimizer.lower() == "adamw" else 0.01

    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.optimizer.lower() == "adamw" else 5e-4

    if args.backbone_lr_ratio is None:
        args.backbone_lr_ratio = 1.0

    # Hardware
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(f"cuda:{args.device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Resolution Handling
    # Support both --imgsz "768,480" (str) and legacy --imgsz_hw logic
    if args.imgsz_hw is None:
        if isinstance(args.imgsz, str) and ("," in args.imgsz or "x" in args.imgsz):
            delim = "," if "," in args.imgsz else "x"
            parts = [int(x.strip()) for x in args.imgsz.split(delim) if x.strip()]
            if len(parts) == 2:
                # Widescreen format (W, H), e.g. 768,480 -> canonical (H, W) is (480, 768)
                if parts[0] > parts[1]:
                    args.imgsz_hw = (parts[1], parts[0])
                else:
                    args.imgsz_hw = (parts[0], parts[1])
            else:
                raise ValueError(
                    f"Invalid format for --imgsz: {args.imgsz}. Expected 'w,h' or single int."
                )
        else:
            val = int(args.imgsz)
            args.imgsz_hw = (val, val)
    print(f"Training Resolution (H, W): {args.imgsz_hw}")

    # Proportional epoch scheduling for fine-tuning phases
    # If user doesn't specify epochs, calculate dynamically based on Phase 1 length

    if args.epochs_sparsity is None:
        args.epochs_sparsity = calculate_finetune_epochs(args.epochs, pct=0.10)
        if args.enable_sparsity and not args.skip_sparsity:
            print(
                f"[Dynamic Epochs] Sparsity set to {args.epochs_sparsity} epochs (10% of {args.epochs})"
            )
    else:
        args.epochs_sparsity = max(1, int(args.epochs_sparsity))

    if args.epochs_qat is None:
        args.epochs_qat = calculate_finetune_epochs(args.epochs, pct=0.05)
        if args.enable_qat:
            print(f"[Dynamic Epochs] QAT set to {args.epochs_qat} epochs (5% of {args.epochs})")
    else:
        args.epochs_qat = max(1, int(args.epochs_qat))

    def validate_dual_head_checkpoint(path):
        ckpt_local = torch.load(path, map_location="cpu")
        validate_dual_head_state_dict(ckpt_local, path)

    # Experiment Setup
    # Construct final output path (handles both absolute and relative paths)
    args.output_dir = os.path.join(args.project, args.name)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir) if SummaryWriter is not None else None
    args.tb_step_offset = 0
    args.tb_epoch_offset = 0

    # Data Setup
    print("Initializing Datasets...")
    data_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    data_cfg.data = args.data
    data_cfg.imgsz = max(args.imgsz_hw)  # Use max dim for build_yolo_dataset checks

    # Use Ultralytics generic dataset builder
    manual_load = False
    try:
        from ultralytics.data.utils import check_det_dataset

        data_cfg_dict = check_det_dataset(args.data)
    except Exception as e:
        print(f"Warning: check_det_dataset failed ({e}). Falling back to manual loading.")
        manual_load = True
        import yaml

        with open(args.data) as f:
            data_cfg_dict = yaml.safe_load(f)

    # Merge custom student augmentations from dataset YAML into data_cfg
    applied_augs = apply_yaml_augmentations(
        data_cfg,
        data_cfg_dict=data_cfg_dict,
        yaml_path=args.data,
        close_mosaic_override=getattr(args, "close_mosaic", None),
    )
    if applied_augs:
        print(f"[Augmentations] Applied dataset YAML overrides: {applied_augs}")

    # Expand user path '~' if present in 'path'
    if "path" in data_cfg_dict and isinstance(data_cfg_dict["path"], str):
        data_cfg_dict["path"] = os.path.expanduser(data_cfg_dict["path"])

    # Handle dataset path override OR resolve paths if manual load was used
    if args.data_path or manual_load:
        if args.data_path:
            print(f"Overriding dataset root path with: {args.data_path}")
            data_cfg_dict["path"] = args.data_path

        root_path = data_cfg_dict.get("path", "")

        # Robustly resolve paths
        for k in ["train", "val", "test"]:
            if k in data_cfg_dict:
                path_val = data_cfg_dict[k]
                if isinstance(path_val, str) and not os.path.isabs(path_val):
                    # If it's just a filename or relative path, prepend new root
                    data_cfg_dict[k] = os.path.join(root_path, path_val)

                # Check existence to warn early
                if isinstance(data_cfg_dict[k], str) and not os.path.exists(data_cfg_dict[k]):
                    # Check common directory variations if path does not exist
                    fallbacks = {
                        "val": [
                            "val/images",
                            "valid/images",
                            "validation/images",
                            "val",
                            "valid",
                            "validation",
                        ],
                        "test": ["test/images", "testing/images", "test", "testing"],
                    }
                    if k in fallbacks:
                        found = False
                        for fb in fallbacks[k]:
                            fb_path = os.path.join(root_path, fb)
                            if os.path.exists(fb_path):
                                print(f"Found {k} split at fallback: {fb_path}")
                                data_cfg_dict[k] = fb_path
                                found = True
                                break
                        if not found and k == "test":
                            print("Warning: Test set path not found. Disabling testing.")
                            del data_cfg_dict[k]
                        elif not found:
                            print(f"Warning: {k} set path not found at {data_cfg_dict[k]}")

                # Pre-verification for Train set
                if (
                    k == "train"
                    and isinstance(data_cfg_dict[k], str)
                    and not os.path.exists(data_cfg_dict[k])
                ):
                    print(f"Error: Train set not found at {data_cfg_dict[k]}!")

    # Pass 'rect=True' for rectangular training if aspect ratio is non-square
    is_rect = args.imgsz_hw[0] != args.imgsz_hw[1]
    use_formula = getattr(args, "formula_image", False)
    rect_train = True if use_formula else is_rect
    rect_val = True if use_formula else is_rect
    shuffle_mode = True if use_formula else not is_rect

    if use_formula:
        print(
            f"[Info] Formula-Image static rectangular training enabled: Canvas fixed to {args.imgsz_hw[0]}x{args.imgsz_hw[1]} with shuffle=True."
        )
    elif is_rect:
        print(
            "[Info] Rectangular training enabled: Disabling shuffle to maintain batch aspect ratios."
        )

    # Build Train Loader
    train_set = build_yolo_dataset(
        data_cfg,
        data_cfg_dict["train"],
        batch=args.batch,
        data=data_cfg_dict,
        mode="train",
        rect=rect_train,
        stride=32,
    )
    if use_formula and hasattr(train_set, "batch_shapes"):
        train_set.batch_shapes[:] = [args.imgsz_hw[0], args.imgsz_hw[1]]

    train_loader = build_dataloader(
        train_set, batch=args.batch, workers=args.workers, shuffle=shuffle_mode
    )

    # Build Val Loader
    val_set = build_yolo_dataset(
        data_cfg,
        data_cfg_dict["val"],
        batch=args.batch,
        data=data_cfg_dict,
        mode="val",
        rect=rect_val,
        stride=32,
    )
    if use_formula and hasattr(val_set, "batch_shapes"):
        val_set.batch_shapes[:] = [args.imgsz_hw[0], args.imgsz_hw[1]]

    val_loader = build_dataloader(val_set, batch=args.batch, workers=args.workers, shuffle=False)

    def _advance_tb_offsets(active_loader, phase_epochs):
        phase_epochs = max(int(phase_epochs), 0)
        if phase_epochs <= 0 or active_loader is None:
            return
        try:
            loader_len = int(len(active_loader))
        except Exception:
            loader_len = 0
        args.tb_step_offset = int(getattr(args, "tb_step_offset", 0)) + (loader_len * phase_epochs)
        args.tb_epoch_offset = int(getattr(args, "tb_epoch_offset", 0)) + phase_epochs

    # ---------------------------------------------------------
    # Phase 1: Distillation / Standalone
    # ---------------------------------------------------------
    phase1_ckpt = None
    phase1_ran = False

    if args.resume and not args.skip_distillation:
        print(f"Resuming Phase 1 from {args.resume}...")
        phase1_ckpt = run_distillation(
            args, device, data_cfg_dict, train_loader, val_loader, writer
        )
        phase1_ran = True
    elif args.skip_distillation:
        if args.resume and os.path.exists(args.resume):
            validate_dual_head_checkpoint(args.resume)
            print(f"Skipping Phase 1, using checkpoint: {args.resume}")
            phase1_ckpt = args.resume
        else:
            raise ValueError("Skip distillation requested but no resume checkpoint provided!")
    elif args.classic:
        phase1_ckpt = run_classic_training(
            args, device, data_cfg_dict, train_loader, val_loader, writer
        )
        phase1_ran = True
    else:
        # Phase 1 Engine handles both distillation and standalone training
        phase1_ckpt = run_distillation(
            args, device, data_cfg_dict, train_loader, val_loader, writer
        )
        phase1_ran = True

    if phase1_ran:
        _advance_tb_offsets(train_loader, getattr(args, "_phase_epochs_completed", args.epochs))

    # Check if checkpoint already has sparsity enabled
    try:
        ckpt_tmp = torch.load(phase1_ckpt, map_location="cpu")
        checkpoint_has_sparsity = ckpt_tmp.get("sparsity_enabled", False)
        del ckpt_tmp
    except Exception:
        checkpoint_has_sparsity = False

    # ---------------------------------------------------------
    # Phase 1.5: Sparsity Fine-Tuning
    # ---------------------------------------------------------
    sparse_ckpt = phase1_ckpt
    sparse_ran = False

    if args.enable_sparsity:
        if args.skip_sparsity:
            print("Skipping Sparsity Phase (per flag).")
        elif checkpoint_has_sparsity:
            print("Checkpoint already has sparsity enabled. Skipping Phase 1.5.")
        else:
            initial_sparse_ckpt = args.resume_sparsity if args.resume_sparsity else phase1_ckpt
            sparse_ckpt = run_sparsity_finetune(
                args, device, data_cfg_dict, train_loader, val_loader, initial_sparse_ckpt, writer
            )
            sparse_ran = True

    if sparse_ran:
        _advance_tb_offsets(
            train_loader, getattr(args, "_phase_epochs_completed", args.epochs_sparsity)
        )

    # ---------------------------------------------------------
    # Phase 2: QAT
    # ---------------------------------------------------------
    qat_ckpt = sparse_ckpt
    qat_ran = False

    if args.enable_qat:
        # QAT Dataloader Logic:
        # QAT needs CLEAN data (no Mosaic) to stabilize calibration and quantization.
        print("[QAT] Rebuilding dataloader with Mosaic=0.0 for clean fine-tuning...")

        # Distinguish between unaugmented calibration (handled separately on val/calib dataset)
        # and QAT fine-tuning augmentations (keep mild geometric & photometric transforms).
        import copy

        qat_hyp = copy.deepcopy(data_cfg)
        qat_hyp.mosaic = 0.0  # Disable heavy multi-image blending for QAT
        qat_hyp.mixup = 0.0
        qat_hyp.copy_paste = 0.0
        # Retain mild fine-tuning augmentations to prevent overfitting on quantized weights
        qat_hyp.scale = float(getattr(data_cfg, "scale", 0.5)) * 0.4  # mild scale (e.g. 0.2)
        qat_hyp.translate = float(getattr(data_cfg, "translate", 0.1)) * 0.5  # mild translation
        qat_hyp.fliplr = float(getattr(data_cfg, "fliplr", 0.5))
        qat_hyp.hsv_h = float(getattr(data_cfg, "hsv_h", 0.015))
        qat_hyp.hsv_s = float(getattr(data_cfg, "hsv_s", 0.7))
        qat_hyp.hsv_v = float(getattr(data_cfg, "hsv_v", 0.4))
        qat_hyp.degrees = 0.0
        qat_hyp.close_mosaic = 0

        # REBUILD the dataset and loader with properly updated hyp namespace
        qat_train_set = build_yolo_dataset(
            qat_hyp,
            data_cfg_dict["train"],
            batch=args.batch,
            data=data_cfg_dict,
            mode="train",
            rect=rect_train,
            stride=32,
        )
        if use_formula and hasattr(qat_train_set, "batch_shapes"):
            qat_train_set.batch_shapes[:] = [args.imgsz_hw[0], args.imgsz_hw[1]]

        # Reassign to train_loader so run_qat uses the new one
        train_loader = build_dataloader(
            qat_train_set, batch=args.batch, workers=args.workers, shuffle=shuffle_mode
        )

        initial_qat_ckpt = args.resume_qat if args.resume_qat else sparse_ckpt
        if args.resume_qat:
            validate_dual_head_checkpoint(args.resume_qat)
            print(f"Resuming QAT from {args.resume_qat}...")

        qat_ckpt = run_qat(
            args, device, data_cfg_dict, train_loader, val_loader, initial_qat_ckpt, writer
        )
        qat_ran = True

    if qat_ran:
        _advance_tb_offsets(train_loader, getattr(args, "_phase_epochs_completed", args.epochs_qat))

    # ---------------------------------------------------------
    # Phase 3: Export to ONNX (DLA-Optimized)
    # ---------------------------------------------------------
    # The run_export logic will auto-detect best checkpoint if args.checkpoint is None
    if args.classic:
        current_ckpt = os.path.join(args.output_dir, "best_classic.pth")
        args.checkpoint = current_ckpt

    args.num_classes = data_cfg_dict.get("nc", 4)
    _best_exported_onnx = run_export(args)

    # ---------------------------------------------------------
    # Phase 4: Final Test Set Evaluation (Optional)
    # ---------------------------------------------------------
    if "test" in data_cfg_dict and data_cfg_dict["test"]:
        print("\n" + "=" * 60)
        print("PHASE 4: FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        test_path = data_cfg_dict["test"]
        print(f"Building Test Loader from: {test_path}")

        try:
            test_set = build_yolo_dataset(
                data_cfg,
                test_path,
                batch=args.batch,
                data=data_cfg_dict,
                mode="val",
                rect=rect_val,
                stride=32,
            )
            if use_formula and hasattr(test_set, "batch_shapes"):
                test_set.batch_shapes[:] = [args.imgsz_hw[0], args.imgsz_hw[1]]

            test_loader = build_dataloader(
                test_set, batch=args.batch, workers=args.workers, shuffle=False
            )

            # Evaluate best PyTorch checkpoint using the custom validator/metrics pipeline.
            final_ckpt = (
                qat_ckpt
                if args.enable_qat
                else (sparse_ckpt if args.enable_sparsity else phase1_ckpt)
            )
            print(f"Loading best checkpoint for testing: {final_ckpt}")

            # Re-init model
            from unina_dla.model.unina_dla import (
                UNINA_DLA,
                detect_architecture_version,
                detect_upsample_mode,
            )

            num_classes = data_cfg_dict.get("nc", 4)

            # Detect architecture from checkpoint
            ckpt = torch.load(final_ckpt, map_location=device)
            state_dict = (
                ckpt.get("ema", ckpt["model"])
                if "ema" in ckpt
                else (ckpt["model"] if "model" in ckpt else ckpt)
            )
            is_fused = any("rbr_reparam" in k for k in state_dict.keys())
            detected_ver = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
            detected_up = getattr(args, "upsample_mode", None) or detect_upsample_mode(state_dict)

            test_model = UNINA_DLA(num_classes=num_classes, deploy=is_fused, version=detected_ver, upsample_mode=detected_up).to(device)

            # Reconstruct QAT modules before loading checkpoint to preserve quantizer state.
            if args.enable_qat:
                from unina_dla.utils.qat_utils import enable_quantization, prepare_qat_model

                test_model = prepare_qat_model(
                    test_model,
                    verification_input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]),
                )
                test_model.load_state_dict(state_dict, strict=False)
                enable_quantization(test_model, device=device)
            else:
                test_model.load_state_dict(state_dict, strict=False)

            test_model.eval()

            print("Running Test Set Validation...")
            from unina_dla.utils.validator import validate

            # Force NMS for Classic Mode (O2M Training)
            use_nms = True if args.classic else None

            # Lineage-based Prefixing
            lineage_prefix = (
                "classic_"
                if args.classic
                else ("distilled_" if args.enable_distillation else "standalone_")
            )
            if args.enable_sparsity:
                lineage_prefix += "sparse_"
            if args.enable_qat:
                lineage_prefix += "qat_"

            test_prefix = f"{lineage_prefix}TEST_"
            raw_names = data_cfg_dict.get("names", {})
            test_names = (
                {i: n for i, n in enumerate(raw_names)}
                if isinstance(raw_names, (list, tuple))
                else (raw_names or {})
            )

            results, _ = validate(
                test_model,
                test_loader,
                device=device,
                num_classes=num_classes,
                names=test_names,
                save_dir=args.output_dir,
                prefix=test_prefix,
                nms=use_nms,
                max_det=int(getattr(args, "max_det", 300)),
            )

            print(f"\nFinal Test Set (conf=0.001) mAP50: {results['mAP50']:.4f}")
            print(f"Final Test Set (conf=0.001) mAP50-95: {results['mAP50-95']:.4f}")

            deploy_conf = float(getattr(args, "deploy_eval_conf", 0.25))
            deploy_iou = float(getattr(args, "deploy_eval_iou", 0.6))
            print(
                f"\nRunning Test Set Deployment Validation (conf={deploy_conf:.3f}, iou={deploy_iou:.3f})..."
            )
            deploy_results, _ = validate(
                test_model,
                test_loader,
                device=device,
                conf_thres=deploy_conf,
                iou_thres=deploy_iou,
                num_classes=num_classes,
                names=test_names,
                save_dir=None,
                prefix=f"{test_prefix}deploy_",
                nms=use_nms,
                max_det=int(getattr(args, "max_det", 300)),
            )

            print(
                f"\nFinal Test Set (conf={deploy_conf:.2f}) mAP50: {deploy_results['mAP50']:.4f}"
            )
            print(
                f"Final Test Set (conf={deploy_conf:.2f}) mAP50-95: {deploy_results['mAP50-95']:.4f}"
            )

            # Save to file with dynamic name
            metrics_save_path = os.path.join(args.output_dir, f"{lineage_prefix}test_metrics.txt")
            with open(metrics_save_path, "w") as f:
                f.write(f"mAP50: {results['mAP50']}\n")
                f.write(f"mAP50-95: {results['mAP50-95']}\n")
                f.write(f"deploy_conf: {deploy_conf}\n")
                f.write(f"deploy_mAP50: {deploy_results['mAP50']}\n")
                f.write(f"deploy_mAP50-95: {deploy_results['mAP50-95']}\n")

        except Exception as e:
            print(f"[Warning] Failed to run test set evaluation: {e}")
            print("Skipping testing phase.")
    else:
        print("[Info] No test set defined or found. Skipping Phase 4.")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
