import torch
import torch.optim as optim
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA

from unina_dla.engines.common import build_phase1_optimizer, compile_student_model
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
    arch_version = getattr(args, "version", None) or "3.2"
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

    optimizer = build_phase1_optimizer(student, args, arch_version)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    task_loss_fn = ClassicLossAdapter(student, dataset_cfg)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP Enabled: {use_amp}")

    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, scaler, ema)
    student = compile_student_model(student, trainer, args)

    # Training Loop
    best_ckpt = None
    epochs_completed = 0
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, train_loader, prefix="classic")
        best_ckpt = trainer.run_validation(val_loader, epoch, prefix="classic")
        epochs_completed += 1
        if trainer.should_stop:
            break

    args._phase_epochs_completed = epochs_completed
    print(f"Phase 1 (Classic) Complete. Best Checkpoint: {best_ckpt}")
    return best_ckpt
