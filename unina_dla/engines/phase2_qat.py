import os
from itertools import islice

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
from unina_dla.utils.qat_utils import (
    enable_calibration,
    enable_quantization,
    prepare_qat_model,
    replace_modules_with_quant,
    sync_quant_device,
)
from unina_dla.utils.sparsity_utils import (
    HAS_SPARSITY,
    create_zero_clamp_hook,
    create_zero_grad_hook,
)


def _iter_limited_loader(loader, num_batches, desc):
    """Iterate at most `num_batches` batches from `loader` with an accurately bounded progress bar."""
    limit = max(0, int(num_batches))
    total = min(len(loader), limit) if hasattr(loader, "__len__") else limit
    return tqdm(islice(loader, limit), total=total, desc=desc)


def _compute_calib_amax(
    models, method="percentile", percentile=99.99, desc="[QAT] Computing amax", device=None
):
    """Compute and load calibrated amax buffers across modules and restore device placement."""
    if isinstance(models, nn.Module):
        models = [models]

    try:
        from pytorch_quantization import calib
    except ImportError:
        calib = None

    quant_modules = [
        m
        for root in models
        if root is not None
        for m in root.modules()
        if getattr(m, "_calibrator", None) is not None or hasattr(m, "load_calib_amax")
    ]
    if not quant_modules:
        return

    compute_amax_fn = getattr(calib, "compute_amax", None) if calib is not None else None
    max_calibrator_cls = getattr(calib, "MaxCalibrator", None) if calib is not None else None

    for m in tqdm(quant_modules, desc=desc):
        if compute_amax_fn is not None:
            try:
                compute_amax_fn(m, method=method, percentile=percentile)
                continue
            except Exception:
                pass

        if hasattr(m, "load_calib_amax"):
            calibrator = getattr(m, "_calibrator", None)
            try:
                if max_calibrator_cls is not None and isinstance(calibrator, max_calibrator_cls):
                    m.load_calib_amax()
                elif calibrator is not None:
                    try:
                        m.load_calib_amax(method=method, percentile=percentile)
                    except TypeError:
                        m.load_calib_amax()
                else:
                    m.load_calib_amax()
            except Exception:
                pass

    # load_calib_amax / HistogramCalibrator allocates `_amax` on CPU; re-align onto target device.
    for root in models:
        if root is not None:
            sync_quant_device(root, device)


def _calibrate_qat_models(student, val_loader, args, device, adapters=None):
    """Run activation calibration with progress bars and enable fake-quantization."""
    # IMPORTANT: keep shuffle=False to preserve rectangular batch grouping and avoid mixed HxW collation errors.
    calib_batches = getattr(args, "calib_batches", 200)
    print(f"[QAT] Calibrating quantizers (up to {calib_batches} batches)...")
    calib_loader = build_dataloader(
        val_loader.dataset, batch=args.batch, workers=4, shuffle=False
    )

    targets = [student] + ([adapters] if adapters is not None else [])
    for mod in targets:
        enable_calibration(mod)
        mod.eval()

    student.deploy = False
    if hasattr(student, "head") and hasattr(student.head, "deploy"):
        student.head.deploy = False

    with torch.no_grad():
        for batch in _iter_limited_loader(calib_loader, calib_batches, desc="[QAT] Calibration"):
            imgs = batch["img"].to(device).float() / 255.0
            if adapters is not None:
                s_f = student.backbone(imgs)
                s_n = student.neck(s_f)
                for sf, ad in zip(s_n, adapters):
                    ad(sf)
                student.head(s_n)
            else:
                student(imgs)

    # Compute and load calibrated amax into quantizers before enabling fake-quantization
    _compute_calib_amax(targets, method="percentile", percentile=99.99, device=device)

    for mod in targets:
        enable_quantization(mod)
        sync_quant_device(mod, device)

    # Freeze BN
    for m in student.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    return calib_loader


@torch.no_grad()
def bn_reestimate(model, loader, device, num_batches=200, force_calib=False):
    if reestimate_bn_stats(
        model,
        loader,
        device,
        num_batches=num_batches,
        batch_iter_fn=lambda ldr, n: _iter_limited_loader(ldr, n, desc="[QAT] BN Re-estimation"),
    ):
        return

    # Fused Conv2d deployment architecture: only calibrate quantizer modules if uncalibrated (or forced),
    # preventing post-QAT bn_reestimate from wiping out QAT-trained amax scales.
    quant_modules = [
        m for m in model.modules()
        if hasattr(m, "_calibrator") or hasattr(m, "load_calib_amax") or hasattr(m, "enable_calib")
    ]
    if quant_modules:
        uncalibrated = [
            m
            for m in quant_modules
            if getattr(m, "_amax", None) is None
            or (torch.is_tensor(m._amax) and m._amax.numel() == 0)
        ]
        if not uncalibrated and not force_calib:
            enable_quantization(model)
            sync_quant_device(model, device)
            print(
                f"[Calibration] {len(quant_modules)} quantizer modules already have calibrated amax buffers; preserving QAT scales."
            )
            return

        print(f"[Calibration] Calibrating {len(quant_modules)} activation quantizer modules ({num_batches} batches)...")
        enable_calibration(model)
        model.eval()
        for batch in _iter_limited_loader(loader, num_batches, desc="[QAT] Calibration"):
            imgs = batch["img"].to(device).float() / 255.0
            model(imgs)

        # Compute and load calibrated amax into quantizers
        _compute_calib_amax(model, method="percentile", percentile=99.99, device=device)
        enable_quantization(model)
        sync_quant_device(model, device)
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
    validate_dual_head_state_dict(state_dict, distill_ckpt)

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
        t_channels = (
            extract_teacher_channels(trainer.teacher_model, args.imgsz_hw, device)
            if trainer.teacher_model
            else []
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1], device=device)
            s_neck = student.neck(student.backbone(dummy))
            student.head(s_neck)
            s_channels = [f.shape[1] for f in s_neck]

        adapters = nn.ModuleList(
            [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
        ).to(device)

        if "adapters" in ckpt:
            adapters.load_state_dict(ckpt["adapters"])

        replace_modules_with_quant(adapters)
        adapters.to(device)

        calib_loader = _calibrate_qat_models(
            student, val_loader, args, device, adapters=adapters
        )

        if "adapters" in ckpt:
            for p in adapters.parameters():
                p.requires_grad = False

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
        distill_losses = build_distill_losses(args, device)
        trainer.setup_distillation(adapters, loss_balancer, distill_losses)

    else:
        calib_loader = _calibrate_qat_models(student, val_loader, args, device)
        optimizer = optim.AdamW(student.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_qat)
    task_loss_fn = build_dual_task_loss(student, args, device)
    restore_task_loss_state(task_loss_fn, ckpt)

    # QAT: No AMP, No Compile; enable EMA after QAT calibration to smooth STE oscillation
    ema = ModelEMA(student)
    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, None, ema)
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
        validate_dual_head_state_dict(state_dict, resume_target)

        student.load_state_dict(state_dict, strict=False)
        sync_quant_device(student, device)
        if "ema" in ckpt and ema is not None:
            ema.ema.load_state_dict(ckpt["ema"], strict=False)
            sync_quant_device(ema.ema, device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        restore_task_loss_state(task_loss_fn, ckpt)
        if "map" in ckpt:
            trainer.best_map = ckpt["map"]
        if "early_stopping" in ckpt:
            trainer.load_early_stopping_state(ckpt["early_stopping"])

        print(f"Resumed QAT from Epoch {start_epoch}")

    # Loop
    best_ckpt = (
        trainer.get_best_checkpoint_path()
        if os.path.isfile(trainer.get_best_checkpoint_path())
        else None
    )
    epochs_completed = 0
    if not trainer.should_stop:
        for epoch in range(start_epoch, args.epochs_qat):
            trainer.train_epoch(epoch, train_loader, prefix="qat")
            best_ckpt = trainer.run_validation(val_loader, epoch, prefix="qat")
            epochs_completed += 1
            if trainer.should_stop:
                break
    else:
        print("[EarlyStopping] Resumed QAT checkpoint already reached stopping criterion.")

    args._phase_epochs_completed = epochs_completed

    # Final BN re-estimation on best checkpoint
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n[QAT] Loading Best Checkpoint {best_ckpt} for BN Re-estimation...")
        best_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        state_dict = (
            best_data.get("ema", best_data["model"])
            if "ema" in best_data
            else (best_data["model"] if "model" in best_data else best_data)
        )
        validate_dual_head_state_dict(state_dict, best_ckpt)
        student.load_state_dict(state_dict, strict=False)
        sync_quant_device(student, device)

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
