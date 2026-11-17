import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from unina_dla.utils.early_stopping import EarlyStopping, extract_monitor_score
from unina_dla.utils.qat_utils import freeze_quantization_observers, update_weight_quantizers
from unina_dla.utils.scheduling import (
    calculate_component_weight,
    calculate_task_weight,
    calculate_teacher_fade,
)
from unina_dla.utils.validator import validate
from unina_dla.utils.visualization import plot_results, save_batch_image


class UnifiedTrainer:
    def __init__(self, args, device, student, teacher_wrapper=None, data_cfg=None, writer=None):
        self.args = args
        self.device = device
        self.student = student
        self.teacher_wrapper = teacher_wrapper
        if self.teacher_wrapper:
            self.teacher_model = self.teacher_wrapper.model.to(device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        else:
            self.teacher_model = None

        self.data_cfg = data_cfg
        self.writer = writer
        self.nc = data_cfg.get("nc", 4)
        self.tb_log_every = max(1, int(getattr(args, "tb_log_every", 10)))
        self.tb_detail_every = max(self.tb_log_every, int(getattr(args, "tb_detail_every", 50)))

        # State
        self.ema = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.task_loss_fn = None

        # Distillation components
        self.adapters = nn.ModuleList()
        self.loss_balancer = None
        self.distill_losses = {}  # {name: loss_fn}
        self.teacher_hooks = []
        self.t_outputs = {"feats": [], "cls": [], "box": []}
        self.t_native_one2one = None
        self._last_kd_meta = {
            "teacher_source": "unknown",
            "assigner_mode": "strict",
            "mask_topk": 1,
            "mask_topk2": None,
        }

        # Phase specifics
        self.is_qat = False
        self.is_sparse = False
        self.zero_clamp = None
        self.zero_grad_clamp = None
        self.use_amp = bool(getattr(args, "amp", False)) and getattr(device, "type", str(device)) == "cuda"
        self.early_stopping = None
        self.should_stop = False

        # History
        self.best_map = 0.0
        self.history = {
            "loss": [],
            "map": [],
            "weights": [],
            "l_sdf": [],
            "l_logit": [],
            "l_box": [],
            "l_task": [],
            "l_task_w": [],
            "l_task_o2o": [],
            "l_task_o2m": [],
            "o2m_aux_w": [],
            "l_sdf_obj": [],
            "l_sdf_mag": [],
            "l_logit_cert": [],
            "l_logit_bce": [],
            "box_iou": [],
            "box_conf": [],
        }

    def _global_step(self, local_step):
        return int(getattr(self.args, "tb_step_offset", 0)) + int(local_step)

    def _global_epoch(self, local_epoch):
        return int(getattr(self.args, "tb_epoch_offset", 0)) + int(local_epoch)

    def setup_distillation(self, adapters, loss_balancer, distill_loss_fns):
        """Call this to enable distillation components"""
        if self.teacher_wrapper:
            self.teacher_model = self.teacher_wrapper.model
            self.adapters = adapters
            self.loss_balancer = loss_balancer
            self.distill_losses = distill_loss_fns
            self._register_teacher_hooks()

    def set_optimizer(self, optimizer, scheduler, task_loss_fn, scaler=None, ema=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_loss_fn = task_loss_fn
        self.scaler = scaler
        self.ema = ema
        if self.is_qat:
            self.use_amp = False  # Force off for QAT
        self.configure_early_stopping()

    def _resolve_early_stopping_config(self):
        """Resolve phase-aware early stopping configuration from args."""
        total_epochs = self._get_phase_total_epochs()
        warmup_epochs = self._get_phase_warmup_epochs(total_epochs)
        metric_name = str(getattr(self.args, "early_stop_metric", "mAP50") or "mAP50")
        min_delta = float(getattr(self.args, "min_delta", 0.0) or 0.0)

        if self.is_qat:
            raw_patience = getattr(self.args, "patience_qat", None)
            patience = 0 if raw_patience is None else int(raw_patience)
            min_epochs = warmup_epochs
        elif self.is_sparse:
            raw_patience = getattr(self.args, "patience_sparsity", None)
            patience = 0 if raw_patience is None else int(raw_patience)
            min_epochs = warmup_epochs
        else:
            raw_patience = getattr(self.args, "patience", 0)
            patience = 0 if raw_patience is None else int(raw_patience)
            cfg_min = getattr(self.args, "early_stop_min_epochs", None)
            min_epochs = warmup_epochs if cfg_min is None else max(0, int(cfg_min))

        return patience, min_delta, min_epochs, metric_name

    def configure_early_stopping(self):
        """Initialize or reconfigure phase-aware EarlyStopping state."""
        patience, min_delta, min_epochs, metric_name = self._resolve_early_stopping_config()
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            min_epochs=min_epochs,
            metric_name=metric_name,
        )
        self.should_stop = False

    def load_early_stopping_state(self, state):
        """Restore EarlyStopping state from a resumed checkpoint."""
        if self.early_stopping is None:
            self.configure_early_stopping()
        self.early_stopping.load_state_dict(state)
        self.should_stop = self.early_stopping.should_stop

    def _get_phase_name(self):
        """Resolve current training phase label for logs/scheduling."""
        if self.is_qat:
            return "qat"
        if self.is_sparse:
            return "sparse"
        if getattr(self.args, "classic", False):
            return "classic"
        if getattr(self.args, "enable_distillation", False):
            return "distill"
        return "standalone"

    def _get_phase_total_epochs(self):
        """Resolve total epochs for the current active phase."""
        if self.is_qat:
            return int(getattr(self.args, "epochs_qat", 0) or 0)
        if self.is_sparse:
            return int(getattr(self.args, "epochs_sparsity", 0) or 0)
        # Distill / standalone / classic all use phase-1 epoch budget
        return int(getattr(self.args, "epochs", 0) or 0)

    def _get_phase_warmup_epochs(self, total_epochs):
        """
        Resolve warmup epochs by phase.
        - distill/standalone/classic: up to 3 epochs (or CLI --warmup_epochs), capped by phase length
        - sparse/qat: <=3 -> 0, 4..10 -> 1, >=11 -> 2
        """
        total_epochs = max(int(total_epochs), 0)
        if total_epochs == 0:
            return 0
        if self.is_qat or self.is_sparse:
            if total_epochs <= 3:
                return 0
            if total_epochs <= 10:
                return 1
            return 2
        cfg_warmup = getattr(self.args, "warmup_epochs", None)
        default_warmup = 3 if cfg_warmup is None else max(int(cfg_warmup), 0)
        return min(default_warmup, total_epochs)

    def _is_final_epoch(self, epoch, total_epochs):
        """True when current epoch is the last epoch of the active phase."""
        return (epoch + 1) >= max(int(total_epochs), 1)

    def _register_teacher_hooks(self):
        """Sets up hooks to capture teacher internals."""

        def get_activation(storage_list):
            def hook(model, input, output):
                storage_list.append(output)

            return hook

        head = self.teacher_model.model[-1]
        head_from = head.f

        for idx in head_from:
            layer = self.teacher_model.model[idx]
            self.teacher_hooks.append(
                layer.register_forward_hook(get_activation(self.t_outputs["feats"]))
            )

    def _handle_warmup(self, epoch, i, len_loader):
        # Phase-aware warmup
        total_epochs = self._get_phase_total_epochs()
        warmup_epochs = self._get_phase_warmup_epochs(total_epochs)
        if warmup_epochs <= 0:
            return

        if epoch < warmup_epochs:
            # Linear warmup per-step
            total_warmup_steps = warmup_epochs * len_loader
            current_step = epoch * len_loader + i
            warmup_factor = current_step / total_warmup_steps

            for param_group in self.optimizer.param_groups:
                # Exclude uncertainty balancer from warmup (maintains dynamic variance learning)
                if param_group.get("is_balancer", False):
                    continue

                if "initial_lr" not in param_group:
                    param_group["initial_lr"] = param_group["lr"]
                param_group["lr"] = param_group["initial_lr"] * warmup_factor
        elif epoch == warmup_epochs and i == 0:
            # Cleanly restore full rate on the first step of post-warmup training
            for param_group in self.optimizer.param_groups:
                if param_group.get("is_balancer", False):
                    continue
                if "initial_lr" in param_group:
                    param_group["lr"] = param_group["initial_lr"]

    def _log_batch_metrics(
        self,
        epoch,
        i,
        len_loader,
        l_task,
        distill_loss_val,
        l_sdf=None,
        l_logit=None,
        l_box=None,
        sdf_bd=None,
        logit_bd=None,
        box_bd=None,
        fade_factor=0.0,
        l_task_box=None,
        l_task_cls=None,
        task_weight=1.0,
        l_task_weighted=None,
        kd_sdf_w=1.0,
        kd_logit_w=1.0,
        kd_box_w=1.0,
        l_task_o2o=None,
        l_task_o2m=None,
        o2m_aux_weight=0.0,
    ):
        local_step = epoch * len_loader + i
        step = self._global_step(local_step)

        if self.writer and (step % self.tb_log_every == 0):
            log_detailed = step % self.tb_detail_every == 0
            # Task Breakdown
            self.writer.add_scalar("Task/Loss", l_task.item(), step)
            self.writer.add_scalar("Task/Weight", task_weight, step)
            if l_task_weighted is not None:
                self.writer.add_scalar("Task/LossWeighted", l_task_weighted.item(), step)
            if l_task_box is not None:
                self.writer.add_scalar("Task/Box", l_task_box.item(), step)
            if l_task_cls is not None:
                self.writer.add_scalar("Task/Cls", l_task_cls.item(), step)
            if l_task_o2o is not None:
                self.writer.add_scalar("Task/O2O", l_task_o2o.item(), step)
            if l_task_o2m is not None:
                self.writer.add_scalar("Task/O2M", l_task_o2m.item(), step)
            self.writer.add_scalar("Task/O2MAuxWeight", float(o2m_aux_weight), step)

            if self.args.enable_distillation:
                task_term = l_task_weighted if l_task_weighted is not None else l_task
                self.writer.add_scalar("Distill/Loss", (task_term + distill_loss_val).item(), step)
                self.writer.add_scalar("Distill/L_Distill", distill_loss_val.item(), step)
                self.writer.add_scalar("Fade/Factor", fade_factor, step)
                if log_detailed:
                    self.writer.add_scalar("KDComp/SDF_Weight", kd_sdf_w, step)
                    self.writer.add_scalar("KDComp/Logit_Weight", kd_logit_w, step)
                    self.writer.add_scalar("KDComp/Box_Weight", kd_box_w, step)
                    kd_meta = getattr(self, "_last_kd_meta", {})
                    src_native = 1.0 if kd_meta.get("teacher_source") == "native_one2one" else 0.0
                    self.writer.add_scalar("KD/TeacherSourceNative", src_native, step)
                    self.writer.add_scalar(
                        "KD/AssignerParity",
                        1.0 if kd_meta.get("assigner_mode") == "parity" else 0.0,
                        step,
                    )
                    if kd_meta.get("mask_topk") is not None:
                        self.writer.add_scalar("KD/MaskTopK", float(kd_meta["mask_topk"]), step)
                    if kd_meta.get("mask_topk2") is not None:
                        self.writer.add_scalar("KD/MaskTopK2", float(kd_meta["mask_topk2"]), step)

                if log_detailed and self.loss_balancer:
                    distill_weights = self.loss_balancer.get_weights()
                    self.writer.add_scalar("Weights/SDF", distill_weights[0].item(), step)
                    self.writer.add_scalar("Weights/Logit", distill_weights[1].item(), step)
                    self.writer.add_scalar("Weights/Box", distill_weights[2].item(), step)

                if log_detailed and l_sdf is not None:
                    self.writer.add_scalar("SDF/Total", l_sdf.item(), step)
                    if sdf_bd:
                        self.writer.add_scalar("SDF/Obj", sdf_bd["obj"], step)
                        self.writer.add_scalar("SDF/Magnitude", sdf_bd["magnitude"], step)
                        if "global_rel" in sdf_bd:
                            self.writer.add_scalar("SDF/GlobalRel", sdf_bd["global_rel"], step)

                if log_detailed and l_logit is not None:
                    self.writer.add_scalar("Logit/Total", l_logit.item(), step)
                    if logit_bd:
                        self.writer.add_scalar("Logit/Certainty", logit_bd["certainty"], step)
                        self.writer.add_scalar("Logit/BCE", logit_bd["bce"], step)
                        if "cross_route" in logit_bd:
                            self.writer.add_scalar(
                                "Logit/CrossRoute", logit_bd["cross_route"], step
                            )

                if log_detailed and l_box is not None:
                    self.writer.add_scalar("Box/Total", l_box.item(), step)
                    if box_bd:
                        self.writer.add_scalar("Box/IoU", box_bd["iou_mean"], step)
                        self.writer.add_scalar("Box/ConfWeight", box_bd["conf_weight_mean"], step)
                        if "cross_route" in box_bd:
                            self.writer.add_scalar("Box/CrossRoute", box_bd["cross_route"], step)

    def train_epoch(self, epoch, loader, prefix="train"):
        self.student.train()

        # Assignment curriculum (O2O training only; deployment remains threshold-only).
        o2o_topk = None
        if hasattr(self.task_loss_fn, "set_assigner_topk"):
            o2o_topk = self._compute_o2o_topk(epoch)
            self.task_loss_fn.set_assigner_topk(o2o_topk)
        elif hasattr(self.task_loss_fn, "loss_fn") and hasattr(
            self.task_loss_fn.loss_fn, "set_assigner_topk"
        ):
            o2o_topk = self._compute_o2o_topk(epoch)
            self.task_loss_fn.loss_fn.set_assigner_topk(o2o_topk)

        # Stage duplicate suppression to late phase to protect early recall learning.
        if hasattr(self.task_loss_fn, "set_dup_suppression_progress"):
            phase_epochs = max(self._get_phase_total_epochs(), 1)
            progress = (epoch + 1) / phase_epochs
            self.task_loss_fn.set_dup_suppression_progress(progress)
        elif hasattr(self.task_loss_fn, "loss_fn") and hasattr(
            self.task_loss_fn.loss_fn, "set_dup_suppression_progress"
        ):
            phase_epochs = max(self._get_phase_total_epochs(), 1)
            progress = (epoch + 1) / phase_epochs
            self.task_loss_fn.loss_fn.set_dup_suppression_progress(progress)

        # O2M auxiliary schedule (active only for dual-task adapter).
        o2m_aux_weight = 0.0
        if hasattr(self.task_loss_fn, "set_aux_weight"):
            o2m_aux_weight = self._compute_o2m_aux_weight(epoch)
            self.task_loss_fn.set_aux_weight(o2m_aux_weight)

        if not getattr(self, "_phase_banner_logged", False):
            self._phase_banner_logged = True
            phase = self._get_phase_name()
            total_epochs = self._get_phase_total_epochs()
            warmup_epochs = self._get_phase_warmup_epochs(total_epochs)
            if not self.teacher_wrapper:
                fade_desc = "disabled"
                task_desc = "constant 1.0000"
            elif self.is_qat:
                fade_desc = (
                    f"cosine decay {self.args.fade_end:.4f} -> {self.args.fade_end * 0.5:.4f}"
                )
                task_desc = f"cosine rise {self.args.task_weight_end:.4f} -> 1.0000"
            elif self.is_sparse:
                fade_desc = f"constant {self.args.fade_end:.4f}"
                task_desc = f"constant {self.args.task_weight_end:.4f}"
            else:
                fade_desc = f"cosine decay {self.args.fade_start:.4f} -> {self.args.fade_end:.4f}"
                task_desc = f"cosine rise {self.args.task_weight_start:.4f} -> {self.args.task_weight_end:.4f}"

            if o2o_topk is None:
                curriculum_desc = "o2o_topk=n/a"
            elif self.is_qat or self.is_sparse:
                curriculum_desc = f"o2o_topk={o2o_topk} (fixed)"
            elif self.args.o2o_topk_start == self.args.o2o_topk_mid == self.args.o2o_topk_end:
                curriculum_desc = f"o2o_topk={self.args.o2o_topk_end} (fixed)"
            else:
                curriculum_desc = (
                    f"o2o_topk={self.args.o2o_topk_start}->{self.args.o2o_topk_mid}->{self.args.o2o_topk_end}"
                )

            if not hasattr(self.task_loss_fn, "set_aux_weight"):
                o2m_aux_desc = "n/a"
            elif self.is_qat or self.is_sparse:
                o2m_aux_desc = f"{o2m_aux_weight:.4f} (fixed)"
            else:
                start_w = float(getattr(self.args, "o2m_aux_weight_start", o2m_aux_weight))
                end_w = self._compute_o2m_aux_weight(max(total_epochs - 1, 0))
                if abs(start_w - end_w) < 1e-6:
                    o2m_aux_desc = f"{start_w:.4f} (fixed)"
                else:
                    o2m_aux_desc = f"{start_w:.4f}->{end_w:.4f}"

            print(
                f"[Trainer] Phase={phase} | epochs={total_epochs} | warmup={warmup_epochs} | "
                f"fade={fade_desc} | task={task_desc} | {curriculum_desc} | o2m_aux={o2m_aux_desc}"
            )
            if self.early_stopping is None:
                self.configure_early_stopping()
            if self.early_stopping.enabled:
                print(
                    f"[EarlyStopping] Enabled for {phase}: metric={self.early_stopping.metric_name}, "
                    f"patience={self.early_stopping.patience}, min_delta={self.early_stopping.min_delta:.4f}, "
                    f"min_epochs={self.early_stopping.min_epochs}"
                )
                if not self.is_qat and not self.is_sparse and not getattr(self.args, "classic", False):
                    stage2_pct = max(0.0, min(float(getattr(self.args, "o2o_topk_stage2_pct", 0.35)), 1.0))
                    stage2_epoch = int(total_epochs * stage2_pct)
                    if self.early_stopping.min_epochs < stage2_epoch:
                        print(
                            f"[EarlyStopping] Note: min_epochs ({self.early_stopping.min_epochs}) is below "
                            f"O2O stage2 curriculum epoch (~{stage2_epoch}); pass --early_stop_min_epochs "
                            "to defer early stopping until strict O2O begins."
                        )

        if o2o_topk is not None and not (self.is_qat or self.is_sparse):
            prev_topk = getattr(self, "_last_logged_o2o_topk", None)
            if prev_topk != o2o_topk:
                phase_epochs = max(self._get_phase_total_epochs(), 1)
                if prev_topk is None:
                    print(f"[Trainer] O2O curriculum: epoch {epoch + 1}/{phase_epochs}, topk={o2o_topk}")
                else:
                    print(
                        f"[Trainer] O2O curriculum transition: epoch {epoch + 1}/{phase_epochs}, "
                        f"topk {prev_topk}->{o2o_topk}"
                    )
                self._last_logged_o2o_topk = o2o_topk

        # Close Mosaic/Mixup/CopyPaste in final close_mosaic epochs of Phase 1
        if not self.is_qat and not self.is_sparse:
            total_epochs = self._get_phase_total_epochs()
            close_mosaic = int(getattr(self.args, "close_mosaic", 10) or 0)
            if close_mosaic > 0 and total_epochs > close_mosaic and epoch >= (total_epochs - close_mosaic):
                ds = getattr(loader, "dataset", None)
                if ds is not None and not getattr(ds, "_unina_mosaic_closed", False):
                    if hasattr(ds, "hyp") and ds.hyp is not None:
                        ds.hyp.mosaic = 0.0
                        ds.hyp.mixup = 0.0
                        ds.hyp.copy_paste = 0.0
                        if hasattr(ds, "close_mosaic") and callable(ds.close_mosaic):
                            try:
                                ds.close_mosaic(hyp=ds.hyp)
                            except Exception:
                                pass
                    if hasattr(ds, "mosaic"):
                        ds.mosaic = False
                    ds._unina_mosaic_closed = True
                    print(
                        f"[Trainer] Closed Mosaic/Mixup/CopyPaste for final {total_epochs - epoch} epochs "
                        f"(epoch {epoch + 1}/{total_epochs})."
                    )

        # QAT specific: freeze observers late in training (once)
        if (
            self.is_qat
            and epoch >= max(1, self.args.epochs_qat - 2)
            and not getattr(self.student, "_quant_observers_frozen", False)
        ):
            freeze_quantization_observers(self.student)

        # Keep BatchNorm statistics frozen during QAT fine-tuning
        if self.is_qat:
            for m in self.student.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        epoch_loss = 0
        pbar = tqdm(loader, desc=f"[{prefix.upper()}] Epoch {epoch + 1}")

        # DIAGNOSTICS: Epoch-level accumulators
        epoch_sdf, epoch_sdf_obj, epoch_sdf_mag = 0.0, 0.0, 0.0
        epoch_logit, epoch_logit_cert, epoch_logit_bce = 0.0, 0.0, 0.0
        epoch_box, epoch_box_iou, epoch_box_conf = 0.0, 0.0, 0.0
        epoch_task = 0.0
        epoch_task_weighted = 0.0
        epoch_task_o2o = 0.0
        epoch_task_o2m = 0.0
        epoch_o2m_aux_w = 0.0
        kd_epoch_logged = False

        for i, batch in enumerate(pbar):
            # Signal the start of a new step for torch.compile with CUDA graphs
            torch.compiler.cudagraph_mark_step_begin()

            step = epoch * len(loader) + i  # Phase-local step
            global_step = self._global_step(step)
            # Warmup logic (generic)
            self._handle_warmup(epoch, i, len(loader))

            if epoch == 0 and i == 0 and self.args.output_dir:
                save_batch_image(
                    batch,
                    os.path.join(self.args.output_dir, f"{prefix}_train_batch0.jpg"),
                    self.data_cfg.get("names", {}),
                    printer=pbar.write,
                )

            imgs = batch["img"].to(self.device).float() / 255.0
            self.optimizer.zero_grad()

            # Clear teacher buffers
            for k in self.t_outputs:
                self.t_outputs[k].clear()
            self.t_native_one2one = None

            # Context Manager for AMP (Disabled for QAT)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # 1. Run Teacher (No Grad)
                if self.teacher_wrapper:
                    with torch.no_grad():
                        teacher_out = self.teacher_model(imgs)
                        self.t_native_one2one = self._parse_teacher_one2one(teacher_out)

                # 2. Run Student
                student_out = self.student(imgs)
                s_o2m_box, s_o2m_cls = None, None
                if isinstance(student_out, tuple) and len(student_out) == 5:
                    s_box, s_cls, s_neck, s_o2m_box, s_o2m_cls = student_out
                elif isinstance(student_out, tuple) and len(student_out) == 3:
                    s_box, s_cls, s_neck = student_out
                else:
                    raise ValueError(
                        f"Unexpected student output format: {type(student_out)} len={len(student_out) if isinstance(student_out, tuple) else 'n/a'}"
                    )

                # 3. Calculate Task Loss
                # Dual adapter: use O2O + O2M. Classic adapter: keep one branch.
                if hasattr(self.task_loss_fn, "set_aux_weight") and (
                    s_o2m_box is not None and s_o2m_cls is not None
                ):
                    task_preds = {
                        "o2o_box": s_box,
                        "o2o_cls": s_cls,
                        "o2m_box": s_o2m_box,
                        "o2m_cls": s_o2m_cls,
                    }
                    task_result = self.task_loss_fn(task_preds, batch, aux_weight=o2m_aux_weight)
                else:
                    base_preds = (
                        (s_o2m_box, s_o2m_cls)
                        if (self.args.classic and s_o2m_box is not None and s_o2m_cls is not None)
                        else (s_box, s_cls)
                    )
                    task_result = self.task_loss_fn(base_preds, batch)

                if isinstance(task_result, tuple) and len(task_result) == 4:
                    l_task_sum, task_loss_items, s_o2o_masks, task_bd = task_result
                elif isinstance(task_result, tuple) and len(task_result) == 3:
                    l_task_sum, task_loss_items, s_o2o_masks = task_result
                    task_bd = {}
                else:
                    raise ValueError(
                        "Task loss adapter must return (loss, items, masks) or (loss, items, masks, breakdown)."
                    )

                # NORMALIZE: Standardize to Mean-reduction for optimization
                batch_size = imgs.shape[0]
                l_task = l_task_sum / batch_size

                l_task_o2o_sum = task_bd.get("l_o2o_sum", l_task_sum.detach())
                l_task_o2m_sum = task_bd.get("l_o2m_sum", torch.tensor(0.0, device=self.device))
                if not torch.is_tensor(l_task_o2o_sum):
                    l_task_o2o_sum = torch.tensor(float(l_task_o2o_sum), device=self.device)
                if not torch.is_tensor(l_task_o2m_sum):
                    l_task_o2m_sum = torch.tensor(float(l_task_o2m_sum), device=self.device)
                l_task_o2o = l_task_o2o_sum / batch_size
                l_task_o2m = l_task_o2m_sum / batch_size
                o2m_weight_eff = float(
                    task_bd.get(
                        "o2m_weight",
                        o2m_aux_weight if hasattr(self.task_loss_fn, "set_aux_weight") else 0.0,
                    )
                )

                # Task Breakdown (already weighted by hyp.box/hyp.cls in loss functions)
                l_task_box = task_loss_items[0]
                l_task_cls = task_loss_items[1]

                task_weight = 1.0
                l_task_weighted = l_task
                if self.teacher_wrapper:
                    task_weight = self._compute_task_weight(epoch)
                    l_task_weighted = l_task * task_weight

                total_loss = l_task_weighted
                distill_loss_val = torch.tensor(0.0, device=self.device)

                l_sdf, l_logit, l_box = None, None, None
                sdf_bd, logit_bd, box_bd = None, None, None
                fade_factor = 0.0
                kd_sdf_w, kd_logit_w, kd_box_w = 1.0, 1.0, 1.0

                # 4. Calculate Distillation Loss
                if self.teacher_wrapper:
                    distill_loss_val, breakdown = self._compute_distill_loss(
                        s_neck, s_box, s_cls, batch, epoch
                    )
                    # Avoid in-place aliasing with l_task_weighted (used for logging as T)
                    total_loss = total_loss + distill_loss_val

                    # Unpack breakdown
                    l_sdf = breakdown["l_sdf"]
                    l_logit = breakdown["l_logit"]
                    l_box = breakdown["l_box"]
                    sdf_bd = breakdown["sdf_bd"]
                    logit_bd = breakdown["logit_bd"]
                    box_bd = breakdown["box_bd"]
                    fade_factor = breakdown["fade_factor"]
                    kd_sdf_w = breakdown["kd_sdf_w"]
                    kd_logit_w = breakdown["kd_logit_w"]
                    kd_box_w = breakdown["kd_box_w"]
                    if not kd_epoch_logged:
                        kd_meta = getattr(self, "_last_kd_meta", {})
                        kd_sig = (
                            kd_meta.get("teacher_source", "unknown"),
                            kd_meta.get("assigner_mode", "strict"),
                            kd_meta.get("mask_topk"),
                            kd_meta.get("mask_topk2"),
                        )
                        if kd_sig != getattr(self, "_last_logged_kd_sig", None):
                            pbar.write(
                                f"[KD] Epoch {epoch + 1}: source={kd_sig[0]} | "
                                f"assigner={kd_sig[1]} (topk={kd_sig[2]}, topk2={kd_sig[3]})"
                            )
                            self._last_logged_kd_sig = kd_sig
                        kd_epoch_logged = True

            # 5. Backward & Step
            if not torch.isfinite(total_loss):
                pbar.write(f"WARNING: NaN Loss at epoch {epoch} batch {i}")
                self.optimizer.zero_grad()
                continue

            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Zero pruned-weight gradients before clipping to ensure accurate gradient norm
                if self.zero_grad_clamp:
                    self.zero_grad_clamp()

                # Guard clip_grad_norm_ against non-finite gradients
                _grads_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in self.student.parameters()
                    if p.grad is not None
                )

                if _grads_finite:
                    # Log gradient norms to TensorBoard
                    if self.writer and (global_step % self.tb_log_every == 0):
                        s_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
                        self.writer.add_scalar("GradNorm/Student", s_norm, global_step)
                        if len(self.adapters) > 0:
                            a_norm = torch.nn.utils.clip_grad_norm_(
                                self.adapters.parameters(), 10.0
                            )
                            self.writer.add_scalar("GradNorm/Adapters", a_norm, global_step)
                        if self.loss_balancer:
                            b_norm = (
                                self.loss_balancer.log_vars.grad.norm()
                                if self.loss_balancer.log_vars.grad is not None
                                else 0.0
                            )
                            self.writer.add_scalar("GradNorm/Balancer", b_norm, global_step)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
                        if len(self.adapters) > 0:
                            torch.nn.utils.clip_grad_norm_(self.adapters.parameters(), 10.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()

                # Zero pruned-weight gradients before clipping
                if self.zero_grad_clamp:
                    self.zero_grad_clamp()

                # Guard clip_grad_norm_ against non-finite gradients
                _grads_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in self.student.parameters()
                    if p.grad is not None
                )

                if not _grads_finite:
                    # Non-finite gradients without scaler — skip step
                    pbar.write(
                        f"WARNING: Non-finite gradients at epoch {epoch} batch {i}, skipping step"
                    )
                    self.optimizer.zero_grad()
                    continue

                # Log gradient norms to TensorBoard
                if self.writer and (global_step % self.tb_log_every == 0):
                    s_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
                    self.writer.add_scalar("GradNorm/Student", s_norm, global_step)
                    if len(self.adapters) > 0:
                        a_norm = torch.nn.utils.clip_grad_norm_(self.adapters.parameters(), 10.0)
                        self.writer.add_scalar("GradNorm/Adapters", a_norm, global_step)
                    if self.loss_balancer:
                        b_norm = (
                            self.loss_balancer.log_vars.grad.norm()
                            if self.loss_balancer.log_vars.grad is not None
                            else 0.0
                        )
                        self.writer.add_scalar("GradNorm/Balancer", b_norm, global_step)
                else:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
                    if len(self.adapters) > 0:
                        torch.nn.utils.clip_grad_norm_(self.adapters.parameters(), 10.0)

                self.optimizer.step()

            # 6. Post-Step Hooks (Sparsity/QAT/EMA)
            if self.zero_clamp:
                self.zero_clamp()

            if self.is_qat:
                update_weight_quantizers(self.student)

            if self.ema:
                self.ema.update(self.student)

            # Logging
            epoch_loss += total_loss.item()

            # Pass new metrics to logger
            self._log_batch_metrics(
                epoch,
                i,
                len(loader),
                l_task,
                distill_loss_val,
                l_sdf,
                l_logit,
                l_box,
                sdf_bd,
                logit_bd,
                box_bd,
                fade_factor,
                l_task_box,
                l_task_cls,
                task_weight,
                l_task_weighted,
                kd_sdf_w,
                kd_logit_w,
                kd_box_w,
                l_task_o2o=l_task_o2o,
                l_task_o2m=l_task_o2m,
                o2m_aux_weight=o2m_weight_eff,
            )

            # Update progress bar metrics: total loss, task components, and distillation terms

            # Line 1: High Level
            postfix = {}
            postfix["L"] = f"{total_loss.item():.2f}"
            postfix["To2o"] = f"{l_task_o2o.item():.2f}"
            postfix["To2m"] = f"{l_task_o2m.item():.2f}"
            postfix["Aw"] = f"{o2m_weight_eff:.2f}"

            if self.args.enable_distillation:
                postfix["T"] = f"{l_task_weighted.item():.2f}"
                postfix["D"] = f"{distill_loss_val.item():.2f}"
                postfix["f"] = f"{fade_factor:.2f}"
                postfix["Tw"] = f"{task_weight:.2f}"
                postfix["Kd"] = f"[{kd_sdf_w:.2f},{kd_logit_w:.2f},{kd_box_w:.2f}]"

            if self.loss_balancer:
                w = self.loss_balancer.get_weights()
                postfix["w"] = f"[{w[0].item():.2f},{w[1].item():.2f},{w[2].item():.2f}]"

            # Line 2: Raw & Breakdowns
            if self.args.enable_distillation and l_sdf is not None:
                postfix["Raw"] = f"[{l_sdf.item():.1f},{l_logit.item():.1f},{l_box.item():.2f}]"

            if self.args.enable_distillation:
                if sdf_bd:
                    postfix["SDF"] = f"[o={sdf_bd['obj']:.2f},m={sdf_bd['magnitude']:.2f}]"
                if logit_bd:
                    postfix["Log"] = f"[c={logit_bd['certainty']:.2f},b={logit_bd['bce']:.2f}]"

            # Task Breakdown & Metrics
            postfix["Task"] = f"[c={l_task_cls.item():.2f},b={l_task_box.item():.2f}]"

            if box_bd:  # Distillation IoU
                postfix["IoU"] = f"{box_bd['iou_mean']:.3f}"
                postfix["Conf"] = f"{box_bd['conf_weight_mean']:.3f}"

            postfix["LR"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_postfix(postfix)

            # Accumulate History
            if self.args.enable_distillation and l_sdf is not None:
                epoch_sdf += l_sdf.item()
                epoch_sdf_obj += sdf_bd["obj"]
                epoch_sdf_mag += sdf_bd["magnitude"]
                epoch_logit += l_logit.item()
                epoch_logit_cert += logit_bd["certainty"]
                epoch_logit_bce += logit_bd["bce"]
                epoch_box += l_box.item()
                epoch_box_iou += box_bd["iou_mean"]
                epoch_box_conf += box_bd["conf_weight_mean"]
            epoch_task += l_task.item()
            epoch_task_weighted += l_task_weighted.item()
            epoch_task_o2o += l_task_o2o.item()
            epoch_task_o2m += l_task_o2m.item()
            epoch_o2m_aux_w += o2m_weight_eff

        # End of Epoch
        total_epochs = self._get_phase_total_epochs()
        warmup_epochs = self._get_phase_warmup_epochs(total_epochs)
        if epoch >= warmup_epochs:  # After phase-specific warmup
            self.scheduler.step()

        # Add to history
        n_batches = len(loader)
        self.history["loss"].append(epoch_loss / n_batches)
        self.history["l_task"].append(epoch_task / n_batches)
        self.history["l_task_w"].append(epoch_task_weighted / n_batches)
        self.history["l_task_o2o"].append(epoch_task_o2o / n_batches)
        self.history["l_task_o2m"].append(epoch_task_o2m / n_batches)
        self.history["o2m_aux_w"].append(epoch_o2m_aux_w / n_batches)
        if self.args.enable_distillation:
            self.history["l_sdf"].append(epoch_sdf / n_batches)
            self.history["l_sdf_obj"].append(epoch_sdf_obj / n_batches)
            self.history["l_sdf_mag"].append(epoch_sdf_mag / n_batches)
            self.history["l_logit"].append(epoch_logit / n_batches)
            self.history["l_logit_cert"].append(epoch_logit_cert / n_batches)
            self.history["l_logit_bce"].append(epoch_logit_bce / n_batches)
            self.history["l_box"].append(epoch_box / n_batches)
            self.history["box_iou"].append(epoch_box_iou / n_batches)
            self.history["box_conf"].append(epoch_box_conf / n_batches)
            if self.loss_balancer:
                self.history["weights"].append(
                    self.loss_balancer.get_weights().detach().cpu().numpy().tolist()
                )

        return epoch_loss / len(loader)

    def _compute_task_weight(self, epoch):
        """
        Phase-aware inverse-fade schedule for GT task authority.
        """
        if self.is_qat:
            return calculate_task_weight(
                epoch, self.args.epochs_qat, start_weight=self.args.task_weight_end, end_weight=1.0
            )
        if self.is_sparse:
            return self.args.task_weight_end
        return calculate_task_weight(
            epoch,
            self.args.epochs,
            start_weight=self.args.task_weight_start,
            end_weight=self.args.task_weight_end,
        )

    def _compute_o2o_topk(self, epoch):
        """
        3-stage O2O assignment curriculum: start -> mid -> end.
        In Sparsity and QAT recovery phases, strict O2O (topk=1) is strictly enforced.
        """
        if self.is_qat or self.is_sparse:
            return 1

        total_epochs = max(self._get_phase_total_epochs(), 1)
        progress = (epoch + 1) / total_epochs

        # Robust ordering in case args are misconfigured.
        stage1 = max(0.0, min(float(self.args.o2o_topk_stage1_pct), 1.0))
        stage2 = max(stage1, min(float(self.args.o2o_topk_stage2_pct), 1.0))

        if progress <= stage1:
            return max(1, int(self.args.o2o_topk_start))
        if progress <= stage2:
            return max(1, int(self.args.o2o_topk_mid))
        return max(1, int(self.args.o2o_topk_end))

    def _compute_o2m_aux_weight(self, epoch):
        """
        O2M auxiliary loss weight schedule.
        Distill/standalone/classic: cosine decay start -> end.
        Sparse/QAT: strictly 0.0 to prevent destroying O2O calibration before deployment.
        """
        if self.is_qat or self.is_sparse:
            return float(getattr(self.args, "o2m_aux_weight_sparse_qat", 0.0))
        start_w = getattr(self.args, "o2m_aux_weight_start", 0.8)
        end_w = getattr(self.args, "o2m_aux_weight_end", 0.0)
        # Distillation-specific end weight:
        # If user explicitly set o2m_aux_weight_end (non-zero), respect it unless
        # o2m_aux_weight_end_distill was explicitly set by the user.
        distill_end = getattr(self.args, "o2m_aux_weight_end_distill", None)
        if self.teacher_wrapper and distill_end is not None:
            if end_w != 0.0 and distill_end == 0.0 and not getattr(self.args, "_o2m_aux_weight_end_distill_explicit", False):
                pass
            else:
                end_w = distill_end
        return float(
            calculate_component_weight(
                epoch,
                self.args.epochs,
                start_w,
                end_w,
            )
        )

    def _compute_kd_component_weights(self, epoch):
        """
        Per-vector KD multipliers relative to global Teacher Fade.
        """
        # Keep component multipliers fixed in sparse/QAT recovery phases.
        if self.is_qat or self.is_sparse:
            return (
                float(self.args.kd_sdf_end),
                float(self.args.kd_logit_end),
                float(self.args.kd_box_end),
            )

        return (
            calculate_component_weight(
                epoch, self.args.epochs, self.args.kd_sdf_start, self.args.kd_sdf_end
            ),
            calculate_component_weight(
                epoch, self.args.epochs, self.args.kd_logit_start, self.args.kd_logit_end
            ),
            calculate_component_weight(
                epoch, self.args.epochs, self.args.kd_box_start, self.args.kd_box_end
            ),
        )

    @staticmethod
    def _sort_scale_tensors(tensors):
        """
        Sort per-scale tensors from highest to lowest spatial resolution (P3 -> P5).
        """
        return sorted(tensors, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)

    def _parse_teacher_one2one(self, teacher_out):
        """
        Parse native Ultralytics teacher outputs and convert one2one branch into
        per-level [B, C, H, W] maps.
        """
        preds = None
        if (
            isinstance(teacher_out, tuple)
            and len(teacher_out) > 1
            and isinstance(teacher_out[1], dict)
        ):
            preds = teacher_out[1]
        elif isinstance(teacher_out, dict):
            preds = teacher_out
        if not isinstance(preds, dict):
            return None

        branch = preds.get("one2one")
        if not isinstance(branch, dict) and all(k in preds for k in ("boxes", "scores", "feats")):
            branch = preds
        if not isinstance(branch, dict):
            return None

        boxes = branch.get("boxes")
        scores = branch.get("scores")
        feats = branch.get("feats")
        if not torch.is_tensor(boxes) or not torch.is_tensor(scores):
            return None
        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            return None
        if boxes.ndim != 3 or scores.ndim != 3:
            return None
        if boxes.shape[1] != 4:
            # UNINA-DLA distills direct 4D box regressions; unsupported teacher shape.
            return None

        feat_list = list(feats)
        spatial_sizes = [int(f.shape[-2]) * int(f.shape[-1]) for f in feat_list]
        total_spatial = sum(spatial_sizes)
        if boxes.shape[-1] != total_spatial or scores.shape[-1] != total_spatial:
            return None

        level_boxes, level_scores = [], []
        offset = 0
        for feat, level_size in zip(feat_list, spatial_sizes):
            h, w = int(feat.shape[-2]), int(feat.shape[-1])
            b_slice = (
                boxes[:, :, offset : offset + level_size]
                .contiguous()
                .view(boxes.shape[0], boxes.shape[1], h, w)
            )
            s_slice = (
                scores[:, :, offset : offset + level_size]
                .contiguous()
                .view(scores.shape[0], scores.shape[1], h, w)
            )
            level_boxes.append(b_slice)
            level_scores.append(s_slice)
            offset += level_size

        return {
            "feats": self._sort_scale_tensors(feat_list),
            "box": self._sort_scale_tensors(level_boxes),
            "cls": self._sort_scale_tensors(level_scores),
        }

    def _teacher_branch_from_feats(self, t_feats):
        """
        Fallback path when native teacher output parsing is unavailable:
        compute teacher O2O per-level maps directly from teacher head modules.
        """
        if self.teacher_model is None:
            return None, None
        head = self.teacher_model.model[-1]
        box_head = getattr(head, "one2one_cv2", None) or getattr(head, "cv2", None)
        cls_head = getattr(head, "one2one_cv3", None) or getattr(head, "cv3", None)
        if box_head is None or cls_head is None:
            return None, None

        n = min(len(t_feats), len(box_head), len(cls_head))
        if n == 0:
            return None, None
        if n != len(t_feats):
            return None, None

        with torch.no_grad():
            t_box = [box_head[i](t_feats[i]) for i in range(n)]
            t_cls = [cls_head[i](t_feats[i]) for i in range(n)]
        return self._sort_scale_tensors(t_box), self._sort_scale_tensors(t_cls)

    def _aligned_teacher_outputs(self):
        """
        Return teacher (feats, box, cls) tensors aligned by scale.

        Teacher hooks can be brittle across model variants. We align by spatial
        resolution to enforce consistent P3/P4/P5 ordering before KD.
        """
        # Primary path: native Ultralytics one2one outputs from teacher forward.
        if isinstance(self.t_native_one2one, dict):
            t_feats = list(self.t_native_one2one.get("feats", []))
            t_box = list(self.t_native_one2one.get("box", []))
            t_cls = list(self.t_native_one2one.get("cls", []))
            n = min(len(t_feats), len(t_box), len(t_cls))
            if n > 0:
                self._last_kd_meta["teacher_source"] = "native_one2one"
                return (
                    self._sort_scale_tensors(t_feats[:n]),
                    self._sort_scale_tensors(t_box[:n]),
                    self._sort_scale_tensors(t_cls[:n]),
                )

        # Fallback path: use hooked features + teacher head modules.
        t_feats = self._sort_scale_tensors(list(self.t_outputs.get("feats", [])))
        if len(t_feats) == 0:
            raise RuntimeError(
                "Teacher produced no usable outputs for KD. "
                "Neither native one2one predictions nor feature hooks are available."
            )
        t_box, t_cls = self._teacher_branch_from_feats(t_feats)
        if t_box is None or t_cls is None:
            raise RuntimeError("Teacher head branch extraction failed for KD.")
        self._last_kd_meta["teacher_source"] = "fallback_head"
        return t_feats, t_box, t_cls

    def _cross_route_teacher_head(self, s_neck, t_feats):
        """
        Cross-head KD mechanism: pass adapted student neck features through frozen
        teacher O2O head branches to produce alternative teacher targets.
        """
        if not bool(getattr(self.args, "kd_mech_cross_head", False)):
            return None, None
        if float(getattr(self.args, "kd_cross_head_weight", 0.0)) <= 0.0:
            return None, None
        if self.teacher_model is None or len(self.adapters) == 0:
            return None, None

        head = self.teacher_model.model[-1]
        box_head = getattr(head, "one2one_cv2", None) or getattr(head, "cv2", None)
        cls_head = getattr(head, "one2one_cv3", None) or getattr(head, "cv3", None)
        if box_head is None or cls_head is None:
            return None, None

        s_feats = self._sort_scale_tensors(list(s_neck))
        t_feats = self._sort_scale_tensors(list(t_feats))
        n = min(len(s_feats), len(t_feats), len(self.adapters), len(box_head), len(cls_head))
        if n == 0:
            return None, None
        if n != len(s_feats) or n != len(t_feats):
            # Keep mechanism conservative: only run when all scales align cleanly.
            return None, None

        routed_box, routed_cls = [], []

        for i in range(n):
            s_feat = s_feats[i]
            t_feat = t_feats[i]

            adapted = self.adapters[i](s_feat)
            if adapted.shape[-2:] != t_feat.shape[-2:]:
                adapted = F.interpolate(
                    adapted, size=t_feat.shape[-2:], mode="bilinear", align_corners=False
                )
            adapted = adapted.to(device=t_feat.device, dtype=t_feat.dtype, non_blocking=True)

            # Gradient flows back into student neck & adapters through frozen teacher head
            routed_box.append(box_head[i](adapted))
            routed_cls.append(cls_head[i](adapted))

        if not routed_box or not routed_cls:
            return None, None
        return self._sort_scale_tensors(routed_box), self._sort_scale_tensors(routed_cls)

    def _compute_distill_loss(self, s_neck, s_box, s_cls, batch, epoch):
        """Centralized distillation math"""
        t_feats, t_box, t_cls = self._aligned_teacher_outputs()

        # Teacher O2O masks for KD supervision.
        # Prefer dedicated mask extractor to avoid mutating optimization loss state.
        current_student_topk = (
            self._compute_o2o_topk(epoch) if hasattr(self, "_compute_o2o_topk") else 1
        )
        if bool(getattr(self.args, "kd_assigner_parity", False)):
            # Assigner parity: align teacher KD mask top-k with student O2O targets
            # to eliminate supervisory self-cancellation.
            kd_mask_topk = current_student_topk
            kd_mask_topk2 = None
            kd_assigner_mode = "parity_aligned"
        else:
            kd_mask_topk = max(1, int(getattr(self.args, "kd_mask_topk", 1)))
            kd_mask_topk2 = None
            kd_assigner_mode = "strict"

        self._last_kd_meta.update(
            assigner_mode=kd_assigner_mode,
            mask_topk=kd_mask_topk,
            mask_topk2=kd_mask_topk2,
        )

        def _set_topk(setter, topk, topk2=None):
            if setter is None:
                return
            try:
                setter(topk, topk2=topk2)
            except TypeError:
                setter(topk)

        if hasattr(self.task_loss_fn, "compute_o2o_masks"):
            t_o2o_masks = self.task_loss_fn.compute_o2o_masks(
                (t_box, t_cls),
                batch,
                topk=kd_mask_topk,
                topk2=kd_mask_topk2,
            )
        else:
            # Backward-compatible fallback for older adapters.
            prev_mode = self.task_loss_fn.loss_fn.training
            prev_topk = None
            prev_topk2 = None
            topk_setter = None
            if hasattr(self.task_loss_fn, "set_assigner_topk"):
                topk_setter = self.task_loss_fn.set_assigner_topk
            elif hasattr(self.task_loss_fn, "loss_fn") and hasattr(
                self.task_loss_fn.loss_fn, "set_assigner_topk"
            ):
                topk_setter = self.task_loss_fn.loss_fn.set_assigner_topk
            if hasattr(self.task_loss_fn, "loss_fn") and hasattr(
                self.task_loss_fn.loss_fn, "assigner_topk"
            ):
                prev_topk = int(self.task_loss_fn.loss_fn.assigner_topk)
            if hasattr(self.task_loss_fn, "loss_fn") and hasattr(
                self.task_loss_fn.loss_fn, "assigner_topk2"
            ):
                prev_topk2 = self.task_loss_fn.loss_fn.assigner_topk2
            needs_override = (
                topk_setter is not None
                and prev_topk is not None
                and (prev_topk != kd_mask_topk or prev_topk2 != kd_mask_topk2)
            )

            self.task_loss_fn.loss_fn.train(False)
            try:
                if needs_override:
                    _set_topk(topk_setter, kd_mask_topk, kd_mask_topk2)
                with torch.no_grad():
                    teacher_task = self.task_loss_fn((t_box, t_cls), batch)
                    if isinstance(teacher_task, tuple) and len(teacher_task) == 4:
                        _, _, t_o2o_masks, _ = teacher_task
                    else:
                        _, _, t_o2o_masks = teacher_task
            finally:
                if needs_override:
                    _set_topk(topk_setter, prev_topk, prev_topk2)
                self.task_loss_fn.loss_fn.train(prev_mode)

        # SDF Loss (with O2O foreground mask injection so weak teacher activations on small cones are never ignored)
        sdf_masks = self.distill_losses["sdf"].generate_attention_mask(t_feats, fg_masks=t_o2o_masks)
        l_sdf, sdf_bd = self.distill_losses["sdf"](
            s_neck, t_feats, self.adapters, attention_mask=sdf_masks, return_breakdown=True
        )

        # Logit Loss
        l_logit, logit_bd = self.distill_losses["logit"](
            s_cls, t_cls, t_o2o_masks, return_breakdown=True
        )

        # Box Loss
        t_conf_maps = [tc.sigmoid().max(dim=1, keepdim=True).values for tc in t_cls]
        l_box, box_bd = self.distill_losses["box"](
            s_box, t_box, t_o2o_masks, confidence_weights=t_conf_maps, return_breakdown=True
        )

        # Optional CrossKD-style mechanism: route adapted student features through
        # frozen teacher O2O head and distill from routed predictions as auxiliary KD.
        cross_head_weight = max(0.0, float(getattr(self.args, "kd_cross_head_weight", 0.0)))
        if bool(getattr(self.args, "kd_mech_cross_head", False)) and cross_head_weight > 0.0:
            routed_box, routed_cls = self._cross_route_teacher_head(s_neck, t_feats)
            if routed_box is not None and routed_cls is not None:
                # True CrossKD: evaluate routed predictions against true teacher targets (t_cls, t_box),
                # training the student neck representations without circular student-head feedback.
                l_logit_route, _ = self.distill_losses["logit"](
                    routed_cls, t_cls, t_o2o_masks, return_breakdown=True
                )
                l_box_route, _ = self.distill_losses["box"](
                    routed_box,
                    t_box,
                    t_o2o_masks,
                    confidence_weights=t_conf_maps,
                    return_breakdown=True,
                )
                l_logit = l_logit + cross_head_weight * l_logit_route
                l_box = l_box + cross_head_weight * l_box_route
                if logit_bd is None:
                    logit_bd = {}
                if box_bd is None:
                    box_bd = {}
                logit_bd["cross_route"] = float(l_logit_route.detach().item())
                logit_bd["cross_route_w"] = cross_head_weight
                box_bd["cross_route"] = float(l_box_route.detach().item())
                box_bd["cross_route_w"] = cross_head_weight

        # Balancing & Fading
        # Pass mean losses to the uncertainty balancer so the regularization term acts as a proper penalty
        kd_sdf_w, kd_logit_w, kd_box_w = self._compute_kd_component_weights(epoch)
        losses = [l_sdf * kd_sdf_w, l_logit * kd_logit_w, l_box * kd_box_w]
        balanced_loss = self.loss_balancer(losses)

        # Determine Fade Strategy based on phase
        # 3-stage trajectory:
        #   Distill/Standalone: cosine fade_start -> fade_end
        #   Sparsity: constant fade_end
        #   QAT: cosine fade_end -> fade_end*0.5
        if self.is_qat:
            fade = calculate_teacher_fade(
                epoch,
                self.args.epochs_qat,
                min_fade=self.args.fade_end,
                max_fade=self.args.fade_end * 0.5,
            )
        elif self.is_sparse:
            fade = self.args.fade_end  # Constant fade for sparsity
        else:
            fade = calculate_teacher_fade(
                epoch, self.args.epochs, min_fade=self.args.fade_start, max_fade=self.args.fade_end
            )

        # Distillation Loss is already a Mean (matching l_task)
        total_distill = balanced_loss * fade

        breakdown = {
            "l_sdf": l_sdf,
            "l_logit": l_logit,
            "l_box": l_box,
            "sdf_bd": sdf_bd,
            "logit_bd": logit_bd,
            "box_bd": box_bd,
            "fade_factor": fade,
            "kd_sdf_w": kd_sdf_w,
            "kd_logit_w": kd_logit_w,
            "kd_box_w": kd_box_w,
        }
        return total_distill, breakdown

    def run_validation(self, loader, epoch, prefix):
        """Unified validation logic"""
        phase_total_epochs = self._get_phase_total_epochs()
        is_final_epoch = self._is_final_epoch(epoch, phase_total_epochs)
        model_to_val = self.ema.ema if self.ema else self.student

        # Unwrap torch.compile wrapper if present to configure deployment mode
        real_model = model_to_val
        if hasattr(real_model, "_orig_mod"):
            real_model = real_model._orig_mod

        old_deploy = None
        old_head_deploy = None
        try:
            if hasattr(real_model, "deploy"):
                old_deploy = real_model.deploy
                real_model.deploy = True  # Always validate with deploy runtime branch
                if hasattr(real_model, "head") and hasattr(real_model.head, "deploy"):
                    old_head_deploy = real_model.head.deploy
                    real_model.head.deploy = True

            # Using default conf_thres=0.001 and iou_thres=0.6 for training validation
            is_periodic_or_final = ((epoch + 1) % 5 == 0) or is_final_epoch
            metrics, val_metrics = validate(
                model_to_val,
                loader,
                self.device,
                num_classes=self.nc,
                max_det=int(getattr(self.args, "max_det", 300)),
                names=self.data_cfg.get("names"),
                save_dir=self.args.output_dir,
                prefix=prefix,
                save_images=is_periodic_or_final,
                save_plots=False,
            )

            # Logging
            if self.writer:
                epoch_step = self._global_epoch(epoch)
                self.writer.add_scalar(f"{prefix}/mAP50", metrics["mAP50"], epoch_step)
            self.history["map"].append(metrics["mAP50"])

            is_new_best = metrics["mAP50"] > self.best_map
            if is_new_best:
                self.best_map = metrics["mAP50"]

            if self.early_stopping is None:
                self.configure_early_stopping()
            if self.early_stopping.enabled:
                es_score = extract_monitor_score(metrics, self.early_stopping.metric_name)
                self.should_stop = self.early_stopping.step(epoch, es_score)
                if self.writer:
                    epoch_step = self._global_epoch(epoch)
                    self.writer.add_scalar(
                        f"{prefix}/EarlyStop_Counter", self.early_stopping.counter, epoch_step
                    )
                if self.should_stop:
                    best_str = (
                        f"{self.early_stopping.best_score:.4f}"
                        if self.early_stopping.best_score is not None
                        else "n/a"
                    )
                    print(
                        f"[EarlyStopping] Stopping {self._get_phase_name()} early at epoch "
                        f"{epoch + 1}/{phase_total_epochs}: {self.early_stopping.metric_name} "
                        f"did not improve by > {self.early_stopping.min_delta:.4f} for "
                        f"{self.early_stopping.patience} epochs "
                        f"(best={best_str} at epoch "
                        f"{self.early_stopping.best_epoch + 1})."
                    )
                elif self.early_stopping.counter > 0:
                    best_str = (
                        f"{self.early_stopping.best_score:.4f}"
                        if self.early_stopping.best_score is not None
                        else "n/a"
                    )
                    print(
                        f"[EarlyStopping] No improvement in {self.early_stopping.metric_name} "
                        f"({es_score:.4f} <= {best_str} + "
                        f"{self.early_stopping.min_delta:.4f}): "
                        f"{self.early_stopping.counter}/{self.early_stopping.patience}"
                    )

            if is_new_best:
                self.save_checkpoint("best", epoch, metrics["mAP50"])
                print(f"New Best Model (mAP50: {self.best_map:.4f})")

            self.save_checkpoint("last", epoch, metrics["mAP50"])

            # COMPARATIVE VALIDATION (Every 5 epochs + Last + Early Stop)
            compare_due = ((epoch + 1) % 5 == 0) or is_final_epoch or self.should_stop
            if (is_new_best or compare_due) and self.args.output_dir and val_metrics is not None:
                val_metrics.plot_all(save_dir=self.args.output_dir, prefix=prefix)

            if compare_due:
                print(f"\n[Validation] Running Comparative Analysis (Epoch {epoch + 1})...")

                # Architecture heuristic: reg_max=0 indicates O2O-style head.
                is_o2o_architecture = False
                if hasattr(real_model, "head") and hasattr(real_model.head, "reg_max"):
                    is_o2o_architecture = real_model.head.reg_max == 0

                # Existing low-threshold comparison (AP-oriented).
                secondary_nms = True if is_o2o_architecture else False
                mode_name = "NMS" if secondary_nms else "O2O"
                print(f"[Validation] Secondary Pass: {mode_name} (nms={secondary_nms})")

                metrics_sec, _ = validate(
                    model_to_val,
                    loader,
                    self.device,
                    num_classes=self.nc,
                    max_det=int(getattr(self.args, "max_det", 300)),
                    names=self.data_cfg.get("names"),
                    save_dir=self.args.output_dir,
                    prefix=f"{prefix}_{mode_name}",
                    nms=secondary_nms,
                    save_images=True,
                    save_plots=True,
                )

                map_primary = metrics["mAP50"]
                map_secondary = metrics_sec["mAP50"]
                gap = map_primary - map_secondary

                if self.writer:
                    epoch_step = self._global_epoch(epoch)
                    self.writer.add_scalar(f"{prefix}/mAP50_{mode_name}", map_secondary, epoch_step)
                    self.writer.add_scalar(f"{prefix}/mAP50_Gap", gap, epoch_step)

                print(f"[{prefix.upper()}] Comparative Results:")
                print(f"  Primary (Default): {map_primary:.4f}")
                print(f"  Secondary ({mode_name}): {map_secondary:.4f}")
                print(f"  Gap (Prim - Sec):  {gap:.4f}")

                if is_o2o_architecture:
                    if gap > -0.01:
                        print("  > SUCCESS: O2O matches/beats NMS. Model is NMS-free.")
                    else:
                        print(f"  > WARNING: O2O lags behind NMS by {abs(gap):.4f}. Tuning needed.")
                else:
                    if gap > 0.1:
                        print("  > CONFIRMED: Classic model relies heavily on NMS (Expected).")
                    else:
                        print("  > SURPRISE: Classic model works without NMS?")

                # Deployment-threshold comparison (operating-point oriented).
                deploy_conf = float(getattr(self.args, "deploy_eval_conf", 0.25))
                deploy_iou = float(getattr(self.args, "deploy_eval_iou", 0.6))
                print(
                    f"[Validation] Deployment Passes: conf={deploy_conf:.3f}, iou={deploy_iou:.3f} "
                    "(force O2O and force NMS)"
                )

                metrics_deploy_o2o, _ = validate(
                    model_to_val,
                    loader,
                    self.device,
                    conf_thres=deploy_conf,
                    iou_thres=deploy_iou,
                    num_classes=self.nc,
                    max_det=int(getattr(self.args, "max_det", 300)),
                    names=self.data_cfg.get("names"),
                    save_dir=None,  # keep deployment diagnostics lightweight
                    prefix=f"{prefix}_deploy_o2o",
                    nms=False,
                )
                metrics_deploy_nms, _ = validate(
                    model_to_val,
                    loader,
                    self.device,
                    conf_thres=deploy_conf,
                    iou_thres=deploy_iou,
                    num_classes=self.nc,
                    max_det=int(getattr(self.args, "max_det", 300)),
                    names=self.data_cfg.get("names"),
                    save_dir=None,  # keep deployment diagnostics lightweight
                    prefix=f"{prefix}_deploy_nms",
                    nms=True,
                )

                deploy_o2o = metrics_deploy_o2o["mAP50"]
                deploy_nms = metrics_deploy_nms["mAP50"]
                deploy_gap = deploy_o2o - deploy_nms

                if self.writer:
                    epoch_step = self._global_epoch(epoch)
                    self.writer.add_scalar(f"{prefix}/DeployConf", deploy_conf, epoch_step)
                    self.writer.add_scalar(f"{prefix}/DeployIoU", deploy_iou, epoch_step)
                    self.writer.add_scalar(f"{prefix}/Deploy_mAP50_O2O", deploy_o2o, epoch_step)
                    self.writer.add_scalar(f"{prefix}/Deploy_mAP50_NMS", deploy_nms, epoch_step)
                    self.writer.add_scalar(f"{prefix}/Deploy_mAP50_Gap", deploy_gap, epoch_step)
                    self.writer.add_scalar(
                        f"{prefix}/Deploy_mAP50_95_O2O", metrics_deploy_o2o["mAP50-95"], epoch_step
                    )
                    self.writer.add_scalar(
                        f"{prefix}/Deploy_mAP50_95_NMS", metrics_deploy_nms["mAP50-95"], epoch_step
                    )

                print(f"[{prefix.upper()}] Deployment-Threshold Results:")
                print(f"  O2O (nms=False): {deploy_o2o:.4f}")
                print(f"  NMS (nms=True):  {deploy_nms:.4f}")
                print(f"  Gap (O2O-NMS):   {deploy_gap:.4f}")

            # Plotting
            if compare_due:
                plot_title = f"{prefix} Metrics"
                plot_results(self.history, self.args.output_dir, prefix, plot_title)

            return self.get_best_checkpoint_path()
        finally:
            if old_deploy is not None:
                real_model.deploy = old_deploy
                if old_head_deploy is not None and hasattr(real_model, "head"):
                    real_model.head.deploy = old_head_deploy

    def save_checkpoint(self, name_prefix, epoch, map_score):
        # Unwrap compiled model
        student_model = self.student
        if hasattr(student_model, "_orig_mod"):
            student_model = student_model._orig_mod

        ckpt = {
            "epoch": epoch,
            "model": student_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "map": map_score,
            "history": self.history,
            "architecture_variant": "dual_head_tal_o2o",
        }
        if self.ema:
            ckpt["ema"] = self.ema.ema.state_dict()
            ckpt["ema_state"] = {"updates": self.ema.updates}
        if self.teacher_wrapper:
            ckpt["adapters"] = self.adapters.state_dict()
            ckpt["balancer"] = self.loss_balancer.state_dict()
        if self.task_loss_fn:
            ckpt["task_loss"] = self.task_loss_fn.loss_fn.state_dict()
            if hasattr(self.task_loss_fn, "o2m_loss_fn"):
                ckpt["task_loss_o2m"] = self.task_loss_fn.o2m_loss_fn.state_dict()
        if self.scaler:
            ckpt["scaler"] = self.scaler.state_dict()
        if self.early_stopping is not None:
            ckpt["early_stopping"] = self.early_stopping.state_dict()

        suffix = (
            "qat"
            if self.is_qat
            else (
                "sparse"
                if self.is_sparse
                else ("distill" if self.args.enable_distillation else "standalone")
            )
        )
        path = os.path.join(self.args.output_dir, f"{name_prefix}_{suffix}.pth")
        torch.save(ckpt, path)

    def get_best_checkpoint_path(self):
        suffix = (
            "qat"
            if self.is_qat
            else (
                "sparse"
                if self.is_sparse
                else ("distill" if self.args.enable_distillation else "standalone")
            )
        )
        return os.path.join(self.args.output_dir, f"best_{suffix}.pth")
