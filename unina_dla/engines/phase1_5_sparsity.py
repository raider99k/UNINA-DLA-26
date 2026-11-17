import os

import torch
import torch.nn as nn
from ultralytics.data import build_dataloader
from ultralytics.utils.torch_utils import ModelEMA

from unina_dla.engines.common import (
    build_distill_losses,
    build_dual_task_loss,
    extract_teacher_channels,
    reestimate_bn_stats,
    restore_task_loss_state,
    validate_dual_head_state_dict,
)
from unina_dla.engines.trainer import UnifiedTrainer
from unina_dla.model.losses.distillation_losses import AdaptiveLossBalancer
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version, detect_upsample_mode
from unina_dla.utils.sparsity_utils import (
    create_zero_clamp_hook,
    create_zero_grad_hook,
    enable_sparsity,
    print_sparse_model_summary,
    verify_sparsity,
)


@torch.no_grad()
def bn_reestimate(model, loader, device, num_batches=200):
    """Re-estimate BatchNorm2d running statistics after pruning."""
    reestimate_bn_stats(model, loader, device, num_batches=num_batches)


def run_sparsity_finetune(args, device, data_cfg, train_loader, val_loader, distill_ckpt, writer):
    print("\n" + "=" * 60)
    print("PHASE 1.5: 2:4 STRUCTURED SPARSITY FINE-TUNING")
    print("=" * 60)

    nc_data = data_cfg.get("nc", 4)

    # 1. Load Student Checkpoint and Detect Architecture Generation
    print(f"Loading Student from {distill_ckpt}...")
    ckpt = torch.load(distill_ckpt, map_location=device, weights_only=False)
    state_dict = (
        ckpt.get("ema", ckpt["model"])
        if "ema" in ckpt
        else (ckpt["model"] if "model" in ckpt else ckpt)
    )
    validate_dual_head_state_dict(state_dict, distill_ckpt)
    # Filter profiling and strip compiler wrappers (_orig_mod.)
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
        if not k.endswith(("total_ops", "total_params"))
    }

    is_fused = any("rbr_reparam" in k for k in state_dict.keys())
    arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
    up_mode = getattr(args, "upsample_mode", None) or detect_upsample_mode(state_dict)
    student = UNINA_DLA(
        num_classes=nc_data, deploy=is_fused, version=arch_version, upsample_mode=up_mode
    ).to(device)
    print(
        f"Student initialized with {nc_data} classes (YOLO26 Head, Gen-{arch_version}, "
        f"upsample={student.upsample_mode}, fused={is_fused})."
    )
    student.load_state_dict(state_dict, strict=True)

    # 2. Setup Teacher
    teacher_wrapper = None
    if args.enable_distillation:
        print(f"Loading Teacher from {args.teacher}...")
        teacher_wrapper = get_teacher_model(args.teacher)

    # 3. Fuse RepVGG before optimizer setup so optimizer tracks fused rbr_reparam Conv2d weights
    if not is_fused:
        print("Fusing RepVGG blocks...")
        student.eval()
        student.switch_to_deploy()
    student.deploy = False
    if hasattr(student, "head") and hasattr(student.head, "deploy"):
        student.head.deploy = False

    # 4. Setup Optimizer (AdamW for sparsity)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    # 5. Apply Sparsity
    student = enable_sparsity(student, optimizer, verbosity=1)
    print_sparse_model_summary(
        student,
        input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]),
        device=str(device),
        title="Model Summary (Post-Sparsity)",
    )

    # 6. EMA (After sparsity)
    ema = ModelEMA(student)

    # 7. Unified Trainer
    trainer = UnifiedTrainer(args, device, student, teacher_wrapper, data_cfg, writer)
    trainer.is_sparse = True
    trainer.zero_clamp = create_zero_clamp_hook(student)
    # Zero gradients before optimizer step to prevent momentum accumulation on pruned weights
    trainer.zero_grad_clamp = create_zero_grad_hook(student)

    # 8. BN Re-estimation
    # IMPORTANT: keep shuffle=False to preserve rectangular batch grouping.
    # If rect datasets are shuffled, collate can receive mixed HxW tensors and fail torch.stack().
    calib_loader = build_dataloader(val_loader.dataset, batch=args.batch, workers=4, shuffle=False)
    bn_reestimate(
        student, calib_loader, device, num_batches=min(len(calib_loader), args.bn_batches)
    )
    student.train()

    # 9. Setup Distillation
    adapters = nn.ModuleList()
    loss_balancer = None
    if args.enable_distillation:
        t_channels = (
            extract_teacher_channels(trainer.teacher_model, args.imgsz_hw, device)
            if trainer.teacher_model
            else []
        )

        with torch.no_grad():
            s_dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1], device=device)
            _, _, s_neck, _, _ = student(s_dummy)
            s_channels = [f.shape[1] for f in s_neck]

        adapters = nn.ModuleList(
            [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
        )

        if "adapters" in ckpt:
            adapters.load_state_dict(ckpt["adapters"])
        else:
            for m in adapters.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    m.weight.data *= 0.1

        optimizer.add_param_group(
            {"params": adapters.parameters(), "lr": 1e-3, "weight_decay": 0.0}
        )

        loss_balancer = AdaptiveLossBalancer(num_losses=3).to(device)
        if "balancer" in ckpt:
            loss_balancer.load_state_dict(ckpt["balancer"])
            loss_balancer.soft_reset()

        optimizer.add_param_group(
            {
                "params": list(loss_balancer.parameters()),
                "lr": 1e-3,
                "weight_decay": 0.0,
                "is_balancer": True,
            }
        )

        distill_losses = build_distill_losses(args, device)
        trainer.setup_distillation(adapters, loss_balancer, distill_losses)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_sparsity)
    task_loss_fn = build_dual_task_loss(student, args, device)
    restore_task_loss_state(task_loss_fn, ckpt)

    use_amp = args.amp and device.type == "cuda"
    # Use conservative initial scale (2^10) to prevent FP16 gradient overflow during sparsity fine-tuning
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=2**10)

    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, scaler, ema)

    if args.compile and hasattr(torch, "compile"):
        try:
            # CUDAGraphs (reduce-overhead) is incompatible with sparsity hooks/pruning;
            # use 'default' (Inductor) for dynamic graph safety.
            student = torch.compile(student, mode="default")
            trainer.student = student
        except Exception:
            pass

    # Resume
    start_epoch = 0
    resume_target = getattr(args, "resume_sparsity", None)
    if resume_target and os.path.isfile(resume_target):
        print(f"Resuming Sparsity Training from {resume_target}")
        ckpt = torch.load(resume_target, map_location=device)
        state_dict = (
            ckpt.get("ema", ckpt["model"])
            if "ema" in ckpt
            else (ckpt["model"] if "model" in ckpt else ckpt)
        )
        validate_dual_head_state_dict(state_dict, resume_target)

        student.load_state_dict(state_dict, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "adapters" in ckpt:
            adapters.load_state_dict(ckpt["adapters"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "balancer" in ckpt and loss_balancer:
            loss_balancer.load_state_dict(ckpt["balancer"])
        restore_task_loss_state(task_loss_fn, ckpt)
        if "scaler" in ckpt and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ema:
            ema.ema.load_state_dict(ckpt["ema"])
        if "map" in ckpt:
            trainer.best_map = ckpt["map"]
        if "early_stopping" in ckpt:
            trainer.load_early_stopping_state(ckpt["early_stopping"])

        print(f"Resumed from Epoch {start_epoch}")
    elif resume_target:
        print(f"[Warning] Resume file not found: {resume_target}")

    best_ckpt = (
        trainer.get_best_checkpoint_path()
        if os.path.isfile(trainer.get_best_checkpoint_path())
        else None
    )
    epochs_completed = 0
    if not trainer.should_stop:
        for epoch in range(start_epoch, args.epochs_sparsity):
            trainer.train_epoch(epoch, train_loader, prefix="sparse")
            best_ckpt = trainer.run_validation(val_loader, epoch, prefix="sparse")
            epochs_completed += 1
            if trainer.should_stop:
                break
    else:
        print("[EarlyStopping] Resumed sparsity checkpoint already reached stopping criterion.")

    args._phase_epochs_completed = epochs_completed

    # Verify Sparsity
    print("\n[Sparsity Verification]")
    is_valid, report = verify_sparsity(ema.ema)

    if is_valid:
        print("✓ All layers pass 2:4 sparsity verification.")
    else:
        print("✗ WARNING: Some layers failed sparsity verification!")

    return best_ckpt
