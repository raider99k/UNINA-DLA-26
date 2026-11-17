import os

import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.utils.torch_utils import ModelEMA

from unina_dla.engines.common import (
    build_distill_losses,
    build_dual_task_loss,
    build_phase1_optimizer,
    compile_student_model,
    extract_teacher_channels,
    restore_task_loss_state,
    validate_dual_head_state_dict,
)
from unina_dla.engines.trainer import UnifiedTrainer
from unina_dla.model.losses.distillation_losses import AdaptiveLossBalancer
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.model_utils import model_summary


def run_distillation(args, device, data_cfg, train_loader, val_loader, writer):
    phase1_title = (
        "PHASE 1: TRI-VECTOR KNOWLEDGE DISTILLATION (YOLO26)"
        if args.enable_distillation
        else "PHASE 1: STANDALONE TRAINING (YOLO26)"
    )
    print("\n" + "=" * 60)
    print(phase1_title)
    print("=" * 60)

    # Standalone or distillation mode check
    if not args.enable_distillation:
        print("[Mode] Standalone Training (No Teacher)")
    else:
        print("[Mode] Knowledge Distillation (Teacher Enabled)")

    # Load Teacher
    teacher_wrapper = None
    if args.enable_distillation:
        print(f"Loading Teacher from {args.teacher}...")
        teacher_wrapper = get_teacher_model(args.teacher)
        if teacher_wrapper is None:
            raise RuntimeError(f"Failed to load teacher model from {args.teacher}")

    nc_data = data_cfg.get("nc", 4)
    print(f"Dataset Class Count: {nc_data}")

    # Initialize Student
    arch_version = getattr(args, "version", None)
    if arch_version is None and getattr(args, "resume", None) and os.path.isfile(args.resume):
        try:
            from unina_dla.model.unina_dla import detect_architecture_version

            ckpt_probe = torch.load(args.resume, map_location="cpu", weights_only=False)
            sd_probe = ckpt_probe.get("model", ckpt_probe) if isinstance(ckpt_probe, dict) else ckpt_probe
            arch_version = detect_architecture_version(sd_probe)
        except Exception:
            arch_version = None
    arch_version = arch_version or "3.2"
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

    model_summary(
        student, input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]), device=str(device)
    )

    # Unified Trainer
    trainer = UnifiedTrainer(args, device, student, teacher_wrapper, data_cfg, writer)

    # EMA
    ema = ModelEMA(student)
    print("EMA Initialized.")

    # Setup Distillation Components
    adapters = nn.ModuleList()
    loss_balancer = None
    distill_losses = {}

    if args.enable_distillation:
        if trainer.teacher_model:
            t_channels = extract_teacher_channels(
                trainer.teacher_model, args.imgsz_hw, device
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1], device=device)
                s_neck = student.neck(student.backbone(dummy))
                s_channels = [f.shape[1] for f in s_neck]

            print(f"Teacher Channels: {t_channels}, Student Channels: {s_channels}")
            adapters = nn.ModuleList(
                [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
            )
            for m in adapters.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        m.bias.data.zero_()

        loss_balancer = AdaptiveLossBalancer(num_losses=3).to(device)
        distill_losses = build_distill_losses(args, device)
        sdf_global_rel_weight = distill_losses["sdf"].global_rel_weight
        print(
            f"[KD Mechanisms] cross_head={bool(getattr(args, 'kd_mech_cross_head', False))}"
            f" (w={float(getattr(args, 'kd_cross_head_weight', 0.0)):.3f}), "
            f"global_rel={bool(getattr(args, 'kd_mech_global_rel', False))}"
            f" (w={sdf_global_rel_weight:.3f}, tokens={int(getattr(args, 'kd_global_rel_tokens', 256))})"
        )
        trainer.setup_distillation(adapters, loss_balancer, distill_losses)

    optimizer = build_phase1_optimizer(
        student, args, arch_version, adapters=adapters, loss_balancer=loss_balancer
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    task_loss_fn = build_dual_task_loss(student, args, device)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP Enabled: {use_amp}")

    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, scaler, ema)
    student = compile_student_model(student, trainer, args)

    # Resume
    start_epoch = 0
    if args.resume and not args.skip_distillation:
        if os.path.isfile(args.resume):
            print(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            validate_dual_head_state_dict(ckpt, args.resume)
            if "model" in ckpt:
                student.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "adapters" in ckpt:
                adapters.load_state_dict(ckpt["adapters"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "map" in ckpt:
                trainer.best_map = ckpt["map"]
            if "ema" in ckpt:
                ema.ema.load_state_dict(ckpt["ema"])
            if "ema_state" in ckpt and "updates" in ckpt["ema_state"]:
                ema.updates = ckpt["ema_state"]["updates"]
            if "balancer" in ckpt and loss_balancer:
                loss_balancer.load_state_dict(ckpt["balancer"])
            restore_task_loss_state(task_loss_fn, ckpt)
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if "early_stopping" in ckpt:
                trainer.load_early_stopping_state(ckpt["early_stopping"])
            print(f"Resumed from Epoch {start_epoch}")

    # Loop
    phase_prefix = "distill" if args.enable_distillation else "standalone"
    best_ckpt = (
        trainer.get_best_checkpoint_path()
        if os.path.isfile(trainer.get_best_checkpoint_path())
        else None
    )
    epochs_completed = 0
    if not trainer.should_stop:
        for epoch in range(start_epoch, args.epochs):
            trainer.train_epoch(epoch, train_loader, prefix=phase_prefix)
            best_ckpt = trainer.run_validation(val_loader, epoch, prefix=phase_prefix)
            epochs_completed += 1
            if trainer.should_stop:
                break
    else:
        print("[EarlyStopping] Resumed checkpoint already reached stopping criterion.")

    args._phase_epochs_completed = epochs_completed
    print(f"Phase 1 Complete. Best Checkpoint: {best_ckpt}")
    return best_ckpt
