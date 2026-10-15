import os

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from unina_dla.utils.optimizers import MuSGD
except ImportError:
    from ultralytics.engine.trainer import MuSGD
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
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.model_utils import model_summary

try:
    from unina_dla.utils.sparsity_utils import HAS_SPARSITY
except ImportError:
    HAS_SPARSITY = False


def _validate_dual_head_checkpoint(ckpt, ckpt_path):
    model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    required_keys = (
        "head.o2o_reg_preds.0.0.weight",
        "head.o2o_cls_preds.0.weight",
        "head.o2m_reg_preds.0.0.weight",
        "head.o2m_cls_preds.0.weight",
    )
    missing = [k for k in required_keys if k not in model_sd]
    if missing:
        missing_str = ", ".join(missing[:2]) + ("..." if len(missing) > 2 else "")
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is from pre-dual-head architecture and is intentionally unsupported. "
            f"Missing keys: {missing_str}. Retrain from scratch with the dual-head model."
        )


def run_distillation(args, device, data_cfg, train_loader, val_loader, writer):
    print("\n" + "=" * 60)
    print("PHASE 1: TRI-VECTOR KNOWLEDGE DISTILLATION (YOLO26)")
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

    model_summary(
        student, input_size=(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]), device=str(device)
    )

    # Unified Trainer
    trainer = UnifiedTrainer(args, device, student, teacher_wrapper, data_cfg, writer)

    # EMA
    ema = ModelEMA(student)
    print("EMA Initialized.")

    # Adapters setup
    adapters = nn.ModuleList()
    if args.enable_distillation:
        # Dummy forward to get channels
        with torch.no_grad():
            trainer.t_outputs["feats"].clear()
            # We need to run teacher manually here just once to get channels,
            # OR trust the user knows what they are doing.
            # Let's use the Trainer's teacher_model since it's already set in init if teacher_wrapper passed
            if trainer.teacher_model:
                # Need hooks registered? UnifiedTrainer does it in setup_distillation,
                # but we need channels to create adapters to pass TO setup_distillation.
                # bootstrapping dilemma.
                # Solution: Manually run dummy on teacher and student to get channels.
                pass

    # Setup Distillation Components
    loss_balancer = None
    distill_losses = {}

    if args.enable_distillation:
        # 1. Get Channels
        if trainer.teacher_model:
            # Register hooks temporarily to get channels
            t_hooks = []

            def get_ch(storage):
                def hook(model, input, output):
                    storage.append(output.shape[1])

                return hook

            t_channels = []
            head = trainer.teacher_model.model[-1]
            for idx in head.f:
                layer = trainer.teacher_model.model[idx]
                t_hooks.append(layer.register_forward_hook(get_ch(t_channels)))

            dummy = torch.zeros(1, 3, args.imgsz_hw[0], args.imgsz_hw[1]).to(device)
            trainer.teacher_model(dummy)
            for h in t_hooks:
                h.remove()

            s_channels = []
            s_feats = student.backbone(dummy)
            s_neck = student.neck(s_feats)
            s_channels = [f.shape[1] for f in s_neck]

            print(f"Teacher Channels: {t_channels}, Student Channels: {s_channels}")

            adapters = nn.ModuleList(
                [nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)]
            )

            # Init Adapters
            for m in adapters.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        m.bias.data.zero_()

        loss_balancer = AdaptiveLossBalancer(num_losses=3).to(device)
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
        print(
            f"[KD Mechanisms] cross_head={bool(getattr(args, 'kd_mech_cross_head', False))}"
            f" (w={float(getattr(args, 'kd_cross_head_weight', 0.0)):.3f}), "
            f"global_rel={bool(getattr(args, 'kd_mech_global_rel', False))}"
            f" (w={sdf_global_rel_weight:.3f}, tokens={int(getattr(args, 'kd_global_rel_tokens', 256))})"
        )

        trainer.setup_distillation(adapters, loss_balancer, distill_losses)

    # Optimizer selection (AdamW, SGD, or MuSGD)
    opt_name = str(getattr(args, "optimizer", "MuSGD")).strip().lower()
    base_lr = float(getattr(args, "lr", 0.001 if opt_name == "adamw" else 0.01))
    weight_decay = float(getattr(args, "weight_decay", 1e-4 if opt_name == "adamw" else 1e-5))
    backbone_lr_ratio = float(getattr(args, "backbone_lr_ratio", 1.0))

    if opt_name == "adamw":
        weights = [p for p in student.parameters() if p.ndim >= 2]
        biases_and_norms = [p for p in student.parameters() if p.ndim < 2]
        if adapters is not None:
            weights = weights + [p for p in adapters.parameters() if p.ndim >= 2]
            biases_and_norms = biases_and_norms + [p for p in adapters.parameters() if p.ndim < 2]

        param_groups = [
            {"params": weights, "lr": base_lr, "weight_decay": weight_decay},
            {"params": biases_and_norms, "lr": base_lr, "weight_decay": 0.0},
        ]
        if loss_balancer is not None:
            param_groups.append(
                {"params": list(loss_balancer.parameters()), "lr": 1e-3, "weight_decay": 0.0, "is_balancer": True}
            )
        optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
        print(f"[Optimizer] Initialized AdamW: lr={base_lr}, weight_decay={weight_decay} (no-bias-decay)")
    elif opt_name == "sgd":
        weights = [p for p in student.parameters() if p.ndim >= 2]
        biases_and_norms = [p for p in student.parameters() if p.ndim < 2]
        if adapters is not None:
            weights = weights + [p for p in adapters.parameters() if p.ndim >= 2]
            biases_and_norms = biases_and_norms + [p for p in adapters.parameters() if p.ndim < 2]

        param_groups = [
            {"params": weights, "lr": base_lr, "weight_decay": weight_decay, "momentum": 0.937, "nesterov": True},
            {"params": biases_and_norms, "lr": base_lr, "weight_decay": 0.0, "momentum": 0.937, "nesterov": True},
        ]
        if loss_balancer is not None:
            param_groups.append(
                {"params": list(loss_balancer.parameters()), "lr": 1e-3, "weight_decay": 0.0, "is_balancer": True}
            )
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
            if adapters is not None:
                scratch_params = scratch_params + list(adapters.parameters())

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
            scratch_str = "Neck + Heads + Adapters" if arch_version == "2.6" else "Stage 4 + Neck + Heads + Adapters"
            print(
                f"[Optimizer] Surgical Differential LR enabled:\n"
                f"  - Pretrained Stages ({stages_str}): LR = {base_lr * backbone_lr_ratio:.6f} ({backbone_lr_ratio:.2f}x ratio)\n"
                f"  - Scratch Layers ({scratch_str}): LR = {base_lr:.6f} (1.00x full rate)"
            )
        else:
            muon_params = [p for p in student.parameters() if p.ndim >= 2]
            sgd_params = [p for p in student.parameters() if p.ndim < 2] + (list(adapters.parameters()) if adapters else [])

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

        if loss_balancer is not None:
            param_groups.append(
                {
                    "params": list(loss_balancer.parameters()),
                    "use_muon": False,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "is_balancer": True,
                }
            )

        optimizer = MuSGD(param_groups, lr=base_lr, muon=0.1, sgd=1.0)
        print(f"[Optimizer] Initialized MuSGD: base_lr={base_lr}, weight_decay={weight_decay}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Task Loss
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    task_loss_fn = DualYOLO26LossAdapter(
        student,
        dataset_cfg,
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

    # Scaler
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP Enabled: {use_amp}")

    # Init Optimizer in Trainer
    trainer.set_optimizer(optimizer, scheduler, task_loss_fn, scaler, ema)

    # Compile
    if args.compile and hasattr(torch, "compile"):
        print("Compiling student model (torch.compile)...")
        # Suppress noisy cudagraph partition warnings from inductor
        import logging

        logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.ERROR)
        try:
            # NOTE: 'reduce-overhead' uses CUDA graphs which cause massive graph
            # partitioning with RepVGG's multi-branch forward. The backward graph
            # becomes so large that the compiled .so can be truncated/corrupted.
            # 'default' mode still provides operator fusion and kernel optimization.
            student = torch.compile(student, mode="default")
            trainer.student = student
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    # Resume
    start_epoch = 0
    if args.resume and not args.skip_distillation:
        if os.path.isfile(args.resume):
            print(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            _validate_dual_head_checkpoint(ckpt, args.resume)
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
            if "task_loss" in ckpt:
                task_loss_fn.loss_fn.load_state_dict(ckpt["task_loss"])
            if "task_loss_o2m" in ckpt:
                try:
                    task_loss_fn.o2m_loss_fn.load_state_dict(ckpt["task_loss_o2m"], strict=False)
                except Exception as e:
                    print(f"[Warning] Could not fully restore O2M task loss state: {e}")
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            print(f"Resumed from Epoch {start_epoch}")

    # Loop
    best_ckpt = None
    for epoch in range(start_epoch, args.epochs):
        trainer.train_epoch(epoch, train_loader, prefix="distill")
        best_ckpt = trainer.run_validation(val_loader, epoch, prefix="distill")

    print(f"Phase 1 Complete. Best Checkpoint: {best_ckpt}")
    return best_ckpt
