import logging

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from unina_dla.utils.optimizers import MuSGD
except ImportError:
    from ultralytics.engine.trainer import MuSGD
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

from unina_dla.model.losses.adapters import DualYOLO26LossAdapter
from unina_dla.model.losses.distillation_losses import (
    LogitDistillationLoss,
    OneToOneBoxLoss,
    SDFDistillationLoss,
)


def validate_dual_head_state_dict(ckpt_or_sd, source_name):
    """Ensure a checkpoint or state_dict contains dual-head (O2O + O2M) weights."""
    model_sd = (
        ckpt_or_sd["model"]
        if isinstance(ckpt_or_sd, dict) and "model" in ckpt_or_sd
        else ckpt_or_sd
    )
    required = (
        "head.o2o_reg_preds.0.0.weight",
        "head.o2m_reg_preds.0.0.weight",
    )
    missing = [k for k in required if k not in model_sd]
    if missing:
        missing_str = ", ".join(missing[:2]) + ("..." if len(missing) > 2 else "")
        raise ValueError(
            f"Checkpoint '{source_name}' is pre-dual-head and unsupported for this pipeline. "
            f"Missing keys: {missing_str}. Retrain from scratch with the dual-head model."
        )


def extract_teacher_channels(teacher_model, imgsz_hw, device):
    """Probe teacher neck output channel dimensions using temporary forward hooks."""
    t_channels = []
    t_hooks = []

    def _hook(_model, _inp, output):
        t_channels.append(output.shape[1])

    head = teacher_model.model[-1]
    for idx in head.f:
        t_hooks.append(teacher_model.model[idx].register_forward_hook(_hook))

    dummy = torch.zeros(1, 3, imgsz_hw[0], imgsz_hw[1], device=device)
    try:
        with torch.no_grad():
            teacher_model(dummy)
    finally:
        for h in t_hooks:
            h.remove()
    return t_channels


def build_distill_losses(args, device):
    """Construct SDF, Logit, and Box distillation losses for the active config."""
    sdf_global_rel_weight = (
        float(getattr(args, "kd_global_rel_weight", 0.0))
        if bool(getattr(args, "kd_mech_global_rel", False))
        else 0.0
    )
    return {
        "sdf": SDFDistillationLoss(
            global_rel_weight=sdf_global_rel_weight,
            global_rel_max_tokens=int(getattr(args, "kd_global_rel_tokens", 256)),
        ).to(device),
        "logit": LogitDistillationLoss(w_fg=2.0, w_bg=1.0, temperature=4.0).to(device),
        "box": OneToOneBoxLoss().to(device),
    }


def build_dual_task_loss(student, args, device):
    """Construct and place DualYOLO26LossAdapter on target device."""
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
    return task_loss_fn


def restore_task_loss_state(task_loss_fn, ckpt):
    """Restore O2O and O2M task-loss states from a checkpoint if present."""
    if "task_loss" in ckpt:
        task_loss_fn.loss_fn.load_state_dict(ckpt["task_loss"])
    if "task_loss_o2m" in ckpt:
        try:
            task_loss_fn.o2m_loss_fn.load_state_dict(ckpt["task_loss_o2m"], strict=False)
        except Exception as e:
            print(f"[Warning] Could not fully restore O2M task loss state: {e}")


def build_phase1_optimizer(student, args, arch_version, adapters=None, loss_balancer=None):
    """Build AdamW, SGD, or MuSGD optimizer (with optional surgical differential LR)."""
    opt_name = str(getattr(args, "optimizer", "MuSGD")).strip().lower()
    base_lr = float(getattr(args, "lr", 0.001 if opt_name == "adamw" else 0.01))
    weight_decay = float(getattr(args, "weight_decay", 1e-4 if opt_name == "adamw" else 5e-4))
    backbone_lr_ratio = float(getattr(args, "backbone_lr_ratio", 1.0))

    if opt_name in ("adamw", "sgd"):
        weights = [p for p in student.parameters() if p.ndim >= 2]
        biases_and_norms = [p for p in student.parameters() if p.ndim < 2]
        if adapters is not None:
            weights = weights + [p for p in adapters.parameters() if p.ndim >= 2]
            biases_and_norms = biases_and_norms + [p for p in adapters.parameters() if p.ndim < 2]

        extra = {"momentum": 0.937, "nesterov": True} if opt_name == "sgd" else {}
        param_groups = [
            {"params": weights, "lr": base_lr, "weight_decay": weight_decay, **extra},
            {"params": biases_and_norms, "lr": base_lr, "weight_decay": 0.0, **extra},
        ]
        if loss_balancer is not None:
            param_groups.append(
                {
                    "params": list(loss_balancer.parameters()),
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "is_balancer": True,
                }
            )
        if opt_name == "adamw":
            optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
            print(f"[Optimizer] Initialized AdamW: lr={base_lr}, weight_decay={weight_decay} (no-bias-decay)")
        else:
            optimizer = optim.SGD(param_groups, lr=base_lr)
            print(f"[Optimizer] Initialized SGD: lr={base_lr}, momentum=0.937, weight_decay={weight_decay}")
        return optimizer

    # Default: MuSGD
    if backbone_lr_ratio != 1.0 and hasattr(student, "backbone"):
        pretrained_params = []
        if hasattr(student.backbone, "stem"):
            pretrained_params.extend(list(student.backbone.stem.parameters()))
        if hasattr(student.backbone, "stages"):
            stage_limit = 4 if arch_version in ("2.6", "3.2") else 3
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
        stages_str = (
            "Stem + Stages 1-4 (100% Backbone)"
            if arch_version in ("2.6", "3.2")
            else "Stem + Stages 1-3"
        )
        suffix_ad = " + Adapters" if adapters is not None else ""
        scratch_str = (
            f"Neck + Heads{suffix_ad}"
            if arch_version in ("2.6", "3.2")
            else f"Stage 4 + Neck + Heads{suffix_ad}"
        )
        print(
            f"[Optimizer] Surgical Differential LR enabled:\n"
            f"  - Pretrained Stages ({stages_str}): LR = {base_lr * backbone_lr_ratio:.6f} ({backbone_lr_ratio:.2f}x ratio)\n"
            f"  - Scratch Layers ({scratch_str}): LR = {base_lr:.6f} (1.00x full rate)"
        )
    else:
        muon_params = [p for p in student.parameters() if p.ndim >= 2]
        sgd_params = [p for p in student.parameters() if p.ndim < 2] + (
            list(adapters.parameters()) if adapters else []
        )
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
    return optimizer


def compile_student_model(student, trainer, args):
    """Apply torch.compile('default') when enabled, suppressing noisy cudagraph warnings."""
    if not (getattr(args, "compile", False) and hasattr(torch, "compile")):
        return student
    print("Compiling student model (torch.compile)...")
    logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.ERROR)
    try:
        # Use 'default' mode: 'reduce-overhead' (CUDA graphs) causes heavy graph
        # partitioning on RepVGG multi-branch forwards.
        student = torch.compile(student, mode="default")
        trainer.student = student
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
    return student


@torch.no_grad()
def reestimate_bn_stats(model, loader, device, num_batches=200, batch_iter_fn=None):
    """Re-estimate BatchNorm2d running statistics using `loader`. Returns True if BN layers existed."""
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_modules:
        return False

    print(
        f"Re-estimating BN statistics for {len(bn_modules)} BatchNorm2d layers ({num_batches} batches)..."
    )
    model.eval()
    for m in bn_modules:
        m.train()
        m.momentum = 0.1
        m.reset_running_stats()

    iterator = batch_iter_fn(loader, num_batches) if batch_iter_fn is not None else loader
    count = 0
    for batch in iterator:
        if batch_iter_fn is None and count >= num_batches:
            break
        imgs = batch["img"].to(device).float() / 255.0
        model(imgs)
        count += 1
    print("BN Re-estimation Complete.")
    return True
