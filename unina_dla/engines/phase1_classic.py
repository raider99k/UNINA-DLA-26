import torch
import torch.optim as optim

try:
    from unina_dla.utils.optimizers import MuSGD
except ImportError:
    from ultralytics.engine.trainer import MuSGD
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA

from unina_dla.engines.trainer import UnifiedTrainer
from unina_dla.model.losses.adapters import ClassicLossAdapter
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.model_utils import model_summary


def run_classic_training(args, device, data_cfg, train_loader, val_loader, writer):
    """
    Classic Baseline Training.

    Features:
    - One-to-Many Assignment (topk=10)
    - Standard BCE + CIoU Loss
    - NMS Enabled during Validation
    - No Teacher/Distillation
    """
    print("\n" + "=" * 60)
    print("PHASE 1: CLASSIC MODE TRAINING")
    print("Baseline: One-to-Many Assignment, Standard Loss, NMS Enabled")
    print("=" * 60)

    nc_data = data_cfg.get("nc", 4)
    print(f"Dataset Class Count: {nc_data}")

    # Initialize Student
    arch_version = getattr(args, "version", None) or "2.5"
    up_mode = getattr(args, "upsample_mode", None)
    student = UNINA_DLA(num_classes=nc_data, deploy=False, version=arch_version, upsample_mode=up_mode).to(device)
    print(f"Student initialized with {nc_data} classes (YOLO26 Head, Gen-{arch_version}, upsample={student.upsample_mode}).")

    # Optional Warm-Start Weights Loading
    if getattr(args, "weights", None):
        from unina_dla.utils.model_utils import load_warm_start_weights
        load_warm_start_weights(student, args.weights, device=device)

    # Optional Pretrained Backbone Loading
    if getattr(args, "pretrained_backbone", None):
        from unina_dla.utils.model_utils import load_pretrained_backbone
        load_pretrained_backbone(student, args.pretrained_backbone, device=device)

    # Print Model Summary
    model_summary(
        student, input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]), device=str(device)
    )

    # Unified Trainer
    trainer = UnifiedTrainer(
        args, device, student, teacher_wrapper=None, data_cfg=data_cfg, writer=writer
    )

    # EMA
    ema = ModelEMA(student)
    print("EMA Initialized.")

    # Optimizer selection (AdamW, SGD, or MuSGD)
    opt_name = str(getattr(args, "optimizer", "MuSGD")).strip().lower()
    base_lr = float(getattr(args, "lr", 0.001 if opt_name == "adamw" else 0.01))
    weight_decay = float(getattr(args, "weight_decay", 1e-4 if opt_name == "adamw" else 1e-5))
    backbone_lr_ratio = float(getattr(args, "backbone_lr_ratio", 1.0))

    if opt_name == "adamw":
        weights = [p for p in student.parameters() if p.ndim >= 2]
        biases_and_norms = [p for p in student.parameters() if p.ndim < 2]
        param_groups = [
            {"params": weights, "lr": base_lr, "weight_decay": weight_decay},
            {"params": biases_and_norms, "lr": base_lr, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
        print(f"[Optimizer] Initialized AdamW: lr={base_lr}, weight_decay={weight_decay} (no-bias-decay)")
    elif opt_name == "sgd":
        weights = [p for p in student.parameters() if p.ndim >= 2]
        biases_and_norms = [p for p in student.parameters() if p.ndim < 2]
        param_groups = [
            {"params": weights, "lr": base_lr, "weight_decay": weight_decay, "momentum": 0.937, "nesterov": True},
            {"params": biases_and_norms, "lr": base_lr, "weight_decay": 0.0, "momentum": 0.937, "nesterov": True},
        ]
        optimizer = optim.SGD(param_groups, lr=base_lr)
        print(f"[Optimizer] Initialized SGD: lr={base_lr}, momentum=0.937, weight_decay={weight_decay}")
    else:
        # Default: MuSGD
        if backbone_lr_ratio != 1.0 and hasattr(student, "backbone"):
            pretrained_params = []
            if hasattr(student.backbone, "stem"):
                pretrained_params.extend(list(student.backbone.stem.parameters()))
            if hasattr(student.backbone, "stages"):
                stage_limit = 4 if arch_version == "2.6" else 3
                for stage in student.backbone.stages[:stage_limit]:
                    pretrained_params.extend(list(stage.parameters()))

            pretrained_param_ids = {id(p) for p in pretrained_params}
            bb_muon = [p for p in pretrained_params if p.ndim >= 2]
            bb_sgd = [p for p in pretrained_params if p.ndim < 2]

            scratch_params = [p for p in student.parameters() if id(p) not in pretrained_param_ids]
            head_muon = [p for p in scratch_params if p.ndim >= 2]
            head_sgd = [p for p in scratch_params if p.ndim < 2]

            param_groups = [
                {
                    "params": bb_muon,
                    "lr": base_lr * backbone_lr_ratio,
                    "use_muon": True,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": weight_decay,
                },
                {
                    "params": bb_sgd,
                    "lr": base_lr * backbone_lr_ratio,
                    "use_muon": False,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                },
                {
                    "params": head_muon,
                    "lr": base_lr,
                    "use_muon": True,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": weight_decay,
                },
                {
                    "params": head_sgd,
                    "lr": base_lr,
                    "use_muon": False,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                },
            ]
            stages_str = "Stem + Stages 1-4 (100% Backbone)" if arch_version == "2.6" else "Stem + Stages 1-3"
            scratch_str = "Neck + Heads" if arch_version == "2.6" else "Stage 4 + Neck + Heads"
            print(
                f"[Optimizer] Surgical Differential LR enabled:\n"
                f"  - Pretrained Stages ({stages_str}): LR = {base_lr * backbone_lr_ratio:.6f} ({backbone_lr_ratio:.2f}x ratio)\n"
                f"  - Scratch Layers ({scratch_str}): LR = {base_lr:.6f} (1.00x full rate)"
            )
        else:
            muon_params = [p for p in student.parameters() if p.ndim >= 2]
            sgd_params = [p for p in student.parameters() if p.ndim < 2]

            param_groups = [
                {
                    "params": muon_params,
                    "lr": base_lr,
                    "use_muon": True,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": weight_decay,
                },
                {
                    "params": sgd_params,
                    "lr": base_lr,
                    "use_muon": False,
                    "nesterov": True,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                },
            ]

        optimizer = MuSGD(param_groups, lr=base_lr, muon=0.1, sgd=1.0)
        print(f"[Optimizer] Initialized MuSGD: base_lr={base_lr}, weight_decay={weight_decay}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Classic Loss
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    task_loss_fn = ClassicLossAdapter(student, dataset_cfg)

    # Scaler
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP Enabled: {use_amp}")

    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, scaler, ema)

    # Optional: torch.compile
    if args.compile and hasattr(torch, "compile"):
        print("Compiling student model (torch.compile)...")
        import logging

        logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.ERROR)
        try:
            student = torch.compile(student, mode="default")
            trainer.student = student  # Update trainer ref
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    # Training Loop
    best_ckpt = None
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, train_loader, prefix="classic")
        best_ckpt = trainer.run_validation(val_loader, epoch, prefix="classic")

    print(f"Phase 1 (Classic) Complete. Best Checkpoint: {best_ckpt}")
    return best_ckpt
