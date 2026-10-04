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
    student = UNINA_DLA(num_classes=nc_data, deploy=False, version=arch_version).to(device)
    print(f"Student initialized with {nc_data} classes (YOLO26 Head, Gen-{arch_version}).")

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

    # Optimizer (MuSGD for consistency)
    muon_params = [p for p in student.parameters() if p.ndim >= 2]
    sgd_params = [p for p in student.parameters() if p.ndim < 2]

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "nesterov": True,
            "momentum": 0.95,
            "weight_decay": 1e-5,
        },
        {
            "params": sgd_params,
            "use_muon": False,
            "nesterov": True,
            "momentum": 0.95,
            "weight_decay": 0.0,
        },
    ]

    optimizer = MuSGD(param_groups, lr=float(getattr(args, "lr", 0.01)), muon=0.1, sgd=1.0)
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
