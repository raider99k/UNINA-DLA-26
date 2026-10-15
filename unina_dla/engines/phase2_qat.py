import os

import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.data import build_dataloader
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

from unina_dla.engines.trainer import UnifiedTrainer
from unina_dla.model.losses.adapters import DualYOLO26LossAdapter
from unina_dla.model.losses.distillation_losses import (
    AdaptiveLossBalancer,
    LogitDistillationLoss,
    OneToOneBoxLoss,
    SDFDistillationLoss,
)
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.unina_dla import UNINA_DLA, detect_architecture_version, detect_upsample_mode
from unina_dla.utils.qat_utils import (
    enable_calibration,
    enable_quantization,
    prepare_qat_model,
    replace_modules_with_quant,
)
from unina_dla.utils.sparsity_utils import (
    create_zero_clamp_hook,
    create_zero_grad_hook,
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

    # Fused Conv2d deployment architecture: calibrate activation quantizer modules
    quant_modules = [
        m for m in model.modules()
        if hasattr(m, "_calibrator") or hasattr(m, "load_calib_amax") or hasattr(m, "enable_calib")
    ]
    if quant_modules:
        print(f"[Calibration] Calibrating {len(quant_modules)} activation quantizer modules ({num_batches} batches)...")
        enable_calibration(model)
        model.eval()
        count = 0
        for batch in loader:
            if count >= num_batches:
                break
            imgs = batch["img"].to(device).float() / 255.0
            model(imgs)
            count += 1

        # Compute and load calibrated amax into quantizers
        try:
            from pytorch_quantization import calib
            calib.compute_amax(model, method="percentile", percentile=99.99)
        except Exception:
            for m in quant_modules:
                if hasattr(m, "load_calib_amax"):
                    try:
                        m.load_calib_amax()
                    except Exception:
                        pass
        enable_quantization(model)
        print("[Calibration] Activation quantizer calibration complete.")
    else:
        print("[Calibration] No BatchNorm or calibrator modules present.")


def run_qat(args, device, data_cfg, train_loader, val_loader, distill_ckpt, writer):
    print("\n" + "=" * 60)
    print("PHASE 2: QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 60)

    if args.enable_distillation and args.teacher is None:
        raise ValueError("Teacher model required for QAT Distillation.")

    nc_data = data_cfg.get("nc", 4)
    ckpt = torch.load(distill_ckpt, map_location=device, weights_only=False)
    state_dict = (
        ckpt.get("ema", ckpt["model"])
        if "ema" in ckpt
        else (ckpt["model"] if "model" in ckpt else ckpt)
    )
    _validate_dual_head_state_dict(state_dict, distill_ckpt)

    # Profiling filter
    profiling_keys = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
    for k in profiling_keys:
        del state_dict[k]

    is_fused = any("rbr_reparam" in k for k in state_dict.keys())
    arch_version = getattr(args, "version", None) or detect_architecture_version(state_dict) or "2.5"
    up_mode = getattr(args, "upsample_mode", None) or detect_upsample_mode(state_dict)

    if is_fused:
        print(f"[QAT] Detected FUSED checkpoint (Gen-{arch_version}, upsample={up_mode}). Creating fused arch...")
        student = UNINA_DLA(num_classes=nc_data, deploy=True, version=arch_version, upsample_mode=up_mode).to(device)
        student.load_state_dict(state_dict, strict=True)
    else:
        print(f"[QAT] Detected UNFUSED checkpoint (Gen-{arch_version}, upsample={up_mode}). Fusing now...")
        # 1. Load as multi-branch first
        # We must load into deploy=False first because the weights match that structure
        student_temp = UNINA_DLA(num_classes=nc_data, deploy=False, version=arch_version, upsample_mode=up_mode).to(device)
        student_temp.load_state_dict(state_dict, strict=False)

        # 2. Fuse (Switch to deploy)
        # This collapses RepVGG blocks. Important: Logic must ensure in-place or return new
        # UNINA_DLA.switch_to_deploy() usually modifies in-place.
        student_temp.switch_to_deploy()
        student = student_temp  # Now it is fused

    # Ensure flag is set for safety before QAT Prep
    student.deploy = True
    if hasattr(student, "head") and hasattr(student.head, "deploy"):
        student.head.deploy = True

    # Prepare QAT
    student = prepare_qat_model(
        student, verification_input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1])
    ).to(device)
    print("Student converted to QAT.")

    # Use UnifiedTrainer
    # Teacher
    teacher_wrapper = None
    if args.enable_distillation:
        print(f"Loading Teacher from {args.teacher}...")
        teacher_wrapper = get_teacher_model(args.teacher)

    trainer = UnifiedTrainer(args, device, student, teacher_wrapper, data_cfg, writer)
    trainer.is_qat = True

    if args.enable_sparsity and HAS_SPARSITY:
        trainer.zero_clamp = create_zero_clamp_hook(student)
        # Zero gradients on pruned weights if sparsity was enabled previously
        trainer.zero_grad_clamp = create_zero_grad_hook(student)

    # Adapters setup
    adapters = nn.ModuleList()
    loss_balancer = None

    if args.enable_distillation:
        # 1. Channels
        t_channels = []
        if trainer.teacher_model:
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
        s_feats = student.backbone(dummy)
        s_neck = student.neck(s_feats)
        head_out = student.head(s_neck)  # Run head to verify
        if isinstance(head_out, tuple) and len(head_out) == 4:
            s_dummy_box, s_dummy_cls = head_out[0], head_out[1]
        else:
            s_dummy_box, s_dummy_cls = head_out
        s_channels = [f.shape[1] for f in s_neck]

        adapters = nn.ModuleList(
            [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
        ).to(device)

        if "adapters" in ckpt:
            adapters.load_state_dict(ckpt["adapters"])

        # Convert Adapters to Quant
        replace_modules_with_quant(adapters)
        adapters.to(device)

        # Calibration (DIRTY: Use Val)
        # IMPORTANT: keep shuffle=False to preserve rectangular batch grouping and avoid mixed HxW collation errors.
        calib_loader = build_dataloader(
            val_loader.dataset, batch=args.batch, workers=4, shuffle=False
        )
        enable_calibration(student)
        enable_calibration(adapters)
        student.eval()
        adapters.eval()
        student.deploy = False
        if hasattr(student, "head") and hasattr(student.head, "deploy"):
            student.head.deploy = False

        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= getattr(args, "calib_batches", 200):
                    break
                imgs = batch["img"].to(device).float() / 255.0
                s_f = student.backbone(imgs)
                s_n = student.neck(s_f)
                for sf, ad in zip(s_n, adapters):
                    ad(sf)
                student.head(s_n)

        # Compute and load calibrated amax into quantizers before enabling fake-quantization
        try:
            from pytorch_quantization import calib
            calib.compute_amax(student, method="percentile", percentile=99.99)
            calib.compute_amax(adapters, method="percentile", percentile=99.99)
        except Exception:
            for m in list(student.modules()) + list(adapters.modules()):
                if hasattr(m, "load_calib_amax"):
                    try:
                        m.load_calib_amax()
                    except Exception:
                        pass

        enable_quantization(student)
        enable_quantization(adapters)

        # Freeze BN
        for m in student.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # Freeze adapters if pretrained
        if "adapters" in ckpt:
            for p in adapters.parameters():
                p.requires_grad = False

        # Optim
        param_groups = [{"params": student.parameters(), "lr": 1e-5}]
        if any(p.requires_grad for p in adapters.parameters()):
            param_groups.append(
                {"params": [p for p in adapters.parameters() if p.requires_grad], "lr": 1e-4}
            )

        loss_balancer = AdaptiveLossBalancer(num_losses=3).to(device)
        if "balancer" in ckpt:
            loss_balancer.load_state_dict(ckpt["balancer"])
            loss_balancer.soft_reset()

        param_groups.append(
            {"params": list(loss_balancer.parameters()), "lr": 1e-4, "weight_decay": 0.0}
        )

        optimizer = optim.AdamW(param_groups)

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

    else:
        # No distillation
        # Calibration (Standard)
        # IMPORTANT: keep shuffle=False to preserve rectangular batch grouping and avoid mixed HxW collation errors.
        calib_loader = build_dataloader(
            val_loader.dataset, batch=args.batch, workers=4, shuffle=False
        )
        enable_calibration(student)
        student.eval()
        student.deploy = False
        if hasattr(student, "head") and hasattr(student.head, "deploy"):
            student.head.deploy = False
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= getattr(args, "calib_batches", 200):
                    break
                imgs = batch["img"].to(device).float() / 255.0
                student(imgs)

        # Compute and load calibrated amax into quantizers before enabling fake-quantization
        try:
            from pytorch_quantization import calib
            calib.compute_amax(student, method="percentile", percentile=99.99)
        except Exception:
            for m in student.modules():
                if hasattr(m, "load_calib_amax"):
                    try:
                        m.load_calib_amax()
                    except Exception:
                        pass

        enable_quantization(student)
        for m in student.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        optimizer = optim.AdamW(student.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_qat)
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

    # QAT: No AMP, No Compile
    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, None, None)  # No EMA for QAT
    trainer.use_amp = False

    # Resume Logic for QAT
    start_epoch = 0
    resume_target = getattr(args, "resume_qat", None)
    if resume_target and os.path.isfile(resume_target):
        print(f"Resuming QAT from {resume_target}")
        ckpt = torch.load(resume_target, map_location=device)
        state_dict = (
            ckpt.get("ema", ckpt["model"])
            if "ema" in ckpt
            else (ckpt["model"] if "model" in ckpt else ckpt)
        )
        _validate_dual_head_state_dict(state_dict, resume_target)

        student.load_state_dict(state_dict, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "task_loss" in ckpt:
            task_loss_fn.loss_fn.load_state_dict(ckpt["task_loss"])
        if "task_loss_o2m" in ckpt:
            try:
                task_loss_fn.o2m_loss_fn.load_state_dict(ckpt["task_loss_o2m"], strict=False)
            except Exception as e:
                print(f"[Warning] Could not fully restore O2M task loss state: {e}")

        print(f"Resumed QAT from Epoch {start_epoch}")

    # Loop
    best_ckpt = None
    for epoch in range(start_epoch, args.epochs_qat):
        trainer.train_epoch(epoch, train_loader, prefix="qat")
        best_ckpt = trainer.run_validation(val_loader, epoch, prefix="qat")

    # Final BN re-estimation on best checkpoint
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n[QAT] Loading Best Checkpoint {best_ckpt} for BN Re-estimation...")
        best_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        state_dict = (
            best_data.get("ema", best_data["model"])
            if "ema" in best_data
            else (best_data["model"] if "model" in best_data else best_data)
        )
        _validate_dual_head_state_dict(state_dict, best_ckpt)
        student.load_state_dict(state_dict, strict=False)

        # Reuse calib_loader (kept non-shuffled for rectangular-shape safety)
        bn_batches = getattr(args, "bn_batches", 200)
        if bn_batches > 0:
            try:
                bn_reestimate(student, calib_loader, device, num_batches=bn_batches)
            except Exception as e:
                print(f"[Warning] Post-QAT BN re-estimation failed: {e}")

        # Update checkpoint
        best_data["model"] = student.state_dict()
        if "ema" in best_data:
            best_data["ema"] = student.state_dict()
        torch.save(best_data, best_ckpt)
        print("[QAT] Best checkpoint updated with re-estimated BN stats.")

    return best_ckpt
