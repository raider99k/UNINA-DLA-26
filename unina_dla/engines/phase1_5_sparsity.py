import os

import torch
import torch.nn as nn
from ultralytics.data import build_dataloader
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA

from unina_dla.engines.trainer import UnifiedTrainer
from unina_dla.model.losses.adapters import DualYOLO26LossAdapter
from unina_dla.model.losses.distillation_losses import (
    AdaptiveLossBalancer,
    LogitDistillationLoss,
    OneToOneBoxLoss,
    SDFDistillationLoss,
)
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version
from unina_dla.utils.sparsity_utils import (
    create_zero_clamp_hook,
    create_zero_grad_hook,
    enable_sparsity,
    print_sparse_model_summary,
    verify_sparsity,
)

try:
    from unina_dla.utils.sparsity_utils import HAS_SPARSITY
except ImportError:
    HAS_SPARSITY = False


def _validate_dual_head_state_dict(state_dict, source_name):
    required = (
        "head.o2o_reg_preds.0.0.weight",
        "head.o2m_reg_preds.0.0.weight",
    )
    missing = [k for k in required if k not in state_dict]
    if missing:
        raise ValueError(
            f"Checkpoint '{source_name}' is pre-dual-head and unsupported for this pipeline. "
            "Retrain from scratch with dual-head TAL O2O."
        )


@torch.no_grad()
def bn_reestimate(model, loader, device, num_batches=200):
    """
    Handle post-pruning normalization and calibration:
    - If BatchNorm2d modules exist, re-estimate running mean/variance.
    - If fused Conv2d deployment architecture (0 BatchNorm2d layers), perform
      data-driven activation calibration and bias compensation across calibration batches.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if bn_modules:
        print(f"Re-estimating BN statistics for {len(bn_modules)} BatchNorm2d layers ({num_batches} batches)...")
        model.eval()
        for m in bn_modules:
            m.train()
            m.momentum = 0.1
            m.reset_running_stats()

        count = 0
        for batch in loader:
            if count >= num_batches:
                break
            imgs = batch["img"].to(device).float() / 255.0
            model(imgs)
            count += 1
        print("BN Re-estimation Complete.")
        return

    # Fused Conv2d deployment architecture: data-driven activation calibration
    conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.bias is not None]
    if not conv_modules:
        return

    print(f"[Sparse Calibration] Calibrating activation distributions and compensating bias for {len(conv_modules)} fused Conv2d layers ({num_batches} batches)...")
    model.eval()
    shifts = {m: [] for m in conv_modules}
    hooks = []

    for m in conv_modules:
        def make_hook(mod):
            def hook_fn(module, inp, out):
                with torch.no_grad():
                    shifts[mod].append(out.detach().mean(dim=(0, 2, 3)))
            return hook_fn
        hooks.append(m.register_forward_hook(make_hook(m)))

    count = 0
    for batch in loader:
        if count >= num_batches:
            break
        imgs = batch["img"].to(device).float() / 255.0
        model(imgs)
        count += 1

    for h in hooks:
        h.remove()

    adjusted_count = 0
    for m, stat_list in shifts.items():
        if stat_list and m.bias is not None:
            mean_channel_out = torch.stack(stat_list, dim=0).mean(dim=0)
            correction = (mean_channel_out * 0.05).clamp(min=-0.1, max=0.1)
            m.bias.data.add_(correction)
            adjusted_count += 1

    print(f"[Sparse Calibration] Applied activation calibration across {adjusted_count} fused Conv2d layers.")


def run_sparsity_finetune(args, device, data_cfg, train_loader, val_loader, distill_ckpt, writer):
    print("\n" + "=" * 60)
    print("PHASE 1.5: 2:4 STRUCTURED SPARSITY FINE-TUNING")
    print("=" * 60)

    nc_data = data_cfg.get("nc", 4)

    # 1. Load Student Checkpoint and Detect Architecture Generation
    print(f"Loading Student from {distill_ckpt}...")
    ckpt = torch.load(distill_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    _validate_dual_head_state_dict(state_dict, distill_ckpt)
    # Filter profiling
    profiling_keys = [k for k in state_dict.keys() if k.endswith(("total_ops", "total_params"))]
    for k in profiling_keys:
        del state_dict[k]

    arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
    student = UNINA_DLA(num_classes=nc_data, deploy=False, version=arch_version).to(device)
    print(f"Student initialized with {nc_data} classes (YOLO26 Head, Gen-{arch_version}).")
    student.load_state_dict(state_dict, strict=True)

    # 2. Setup Teacher
    teacher_wrapper = None
    if args.enable_distillation:
        print(f"Loading Teacher from {args.teacher}...")
        teacher_wrapper = get_teacher_model(args.teacher)

    # 3. Setup Optimizer (AdamW for sparsity)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-5)

    # 4. Fuse RepVGG
    print("Fusing RepVGG blocks...")
    student.eval()
    student.switch_to_deploy()
    student.deploy = False
    if hasattr(student, "head") and hasattr(student.head, "deploy"):
        student.head.deploy = False

    # 5. Apply Sparsity
    student = enable_sparsity(student, optimizer, verbosity=2)
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
        # Load Adapters/Balancer from checkpoint if available
        # Need channels first
        t_channels = []
        if trainer.teacher_model:
            # Temp hook dict
            t_hooks = []

            def get_ch(storage):
                def hook(model, input, output):
                    storage.append(output.shape[1])

                return hook

            head = trainer.teacher_model.model[-1]
            for idx in head.f:
                t_hooks.append(
                    trainer.teacher_model.model[idx].register_forward_hook(get_ch(t_channels))
                )
            dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]).to(device)
            trainer.teacher_model(dummy)
            for h in t_hooks:
                h.remove()

        s_channels = []
        s_dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]).to(device)
        s_box, s_cls, s_neck, _, _ = student(s_dummy)
        s_channels = [f.shape[1] for f in s_neck]

        adapters = nn.ModuleList(
            [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
        )

        # Init or Load
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

        sdf_global_rel_weight = (
            float(getattr(args, "kd_global_rel_weight", 0.0))
            if bool(getattr(args, "kd_mech_global_rel", False))
            else 0.0
        )
        distill_losses = {
            "sdf": SDFDistillationLoss(
                global_rel_weight=sdf_global_rel_weight,
                global_rel_max_tokens=int(getattr(args, "kd_global_rel_tokens", 256)),
            ).to(device),
            "logit": LogitDistillationLoss(w_fg=2.0, w_bg=1.0, temperature=4.0).to(device),
            "box": OneToOneBoxLoss().to(device),
        }
        trainer.setup_distillation(adapters, loss_balancer, distill_losses)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_sparsity)
    task_loss_fn = DualYOLO26LossAdapter(
        student,
        IterableSimpleNamespace(**DEFAULT_CFG.__dict__),
        return_aux=True,
        o2m_topk=int(args.o2m_topk),
        aux_weight_start=float(args.o2m_aux_weight_start),
        aux_weight_end=float(args.o2m_aux_weight_end),
        dup_loss_weight=float(getattr(args, "dup_loss_weight", 0.0)),
        dup_iou_thr=float(getattr(args, "dup_iou_thr", 0.6)),
        dup_margin=float(getattr(args, "dup_margin", 0.2)),
        dup_topk=int(getattr(args, "dup_topk", 20)),
        hard_bg_loss_weight=float(getattr(args, "hard_bg_loss_weight", 0.0)),
        hard_bg_iou_thr=float(getattr(args, "hard_bg_iou_thr", 0.2)),
        hard_bg_topk=int(getattr(args, "hard_bg_topk", 64)),
        dup_start_pct=float(getattr(args, "dup_start_pct", 0.6)),
    )
    task_loss_fn.loss_fn.to(device)
    task_loss_fn.o2m_loss_fn.to(device)
    if "task_loss" in ckpt:
        task_loss_fn.loss_fn.load_state_dict(ckpt["task_loss"])
    if "task_loss_o2m" in ckpt:
        try:
            task_loss_fn.o2m_loss_fn.load_state_dict(ckpt["task_loss_o2m"], strict=False)
        except Exception as e:
            print(f"[Warning] Could not fully restore O2M task loss state: {e}")

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
    if (
        args.resume and not args.skip_distillation
    ):  # Reuse skip flag or add new one? usually just check args.resume
        if os.path.isfile(args.resume):
            print(f"Resuming Sparsity Training from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            _validate_dual_head_state_dict(state_dict, args.resume)

            # Key: We might have loaded distill_ckpt earlier, but this resume ckpt
            # should be from the sparsity phase, so it overrides everything.
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
            if "balancer" in ckpt and loss_balancer:
                loss_balancer.load_state_dict(ckpt["balancer"])
            if "task_loss" in ckpt:
                task_loss_fn.loss_fn.load_state_dict(ckpt["task_loss"])
            if "task_loss_o2m" in ckpt:
                try:
                    task_loss_fn.o2m_loss_fn.load_state_dict(ckpt["task_loss_o2m"], strict=False)
                except Exception as e:
                    print(f"[Warning] Could not fully restore O2M task loss state: {e}")
            if "scaler" in ckpt and scaler:
                scaler.load_state_dict(ckpt["scaler"])
            if "ema" in ckpt and ema:
                ema.ema.load_state_dict(ckpt["ema"])

            print(f"Resumed from Epoch {start_epoch}")
        else:
            print(f"[Warning] Resume file not found: {args.resume}")

    best_ckpt = None
    for epoch in range(start_epoch, args.epochs_sparsity):
        trainer.train_epoch(epoch, train_loader, prefix="sparse")
        best_ckpt = trainer.run_validation(val_loader, epoch, prefix="sparse")

    # Verify Sparsity
    print("\n[Sparsity Verification]")
    is_valid, report = verify_sparsity(ema.ema)
    # Note: verify_sparsity checks the EMA model which tracks 'student'.
    # If standard training ensures EMA is up to date, this is fine.

    if is_valid:
        print("✓ All layers pass 2:4 sparsity verification.")
    else:
        print("✗ WARNING: Some layers failed sparsity verification!")

    # Final BatchNorm re-estimation on best checkpoint to align statistics
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n[Sparsity] Loading best checkpoint {best_ckpt} for BN re-estimation...")
        ckpt = torch.load(best_ckpt, map_location=device)
        # Prefer EMA model if present since best_map validation evaluated EMA
        state_dict = ckpt.get("ema", ckpt["model"]) if "ema" in ckpt else ckpt["model"]
        # Unwrap torch.compile's OptimizedModule wrapper before loading
        real_student = student._orig_mod if hasattr(student, "_orig_mod") else student
        real_student.load_state_dict(state_dict)

        bn_reestimate(
            real_student, calib_loader, device, num_batches=min(len(calib_loader), args.bn_batches)
        )

        # Update checkpoint
        ckpt["model"] = real_student.state_dict()
        if "ema" in ckpt:
            ckpt["ema"] = real_student.state_dict()
        torch.save(ckpt, best_ckpt)
        print("[Sparsity] Best checkpoint updated with re-estimated BN stats.")

    return best_ckpt
