import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Backend non-interattivo per server
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from unina_dla.utils.box_ops import decode_bboxes


def save_batch_image(batch, filename, names):
    """Save a batch of images with ground truth boxes for inspection."""
    imgs = batch["img"].numpy()  # [B, 3, H, W]

    # Ultralytics batch format: separate tensors for batch_idx, cls, bboxes
    b_idx = batch["batch_idx"]  # [N] - which image each target belongs to
    b_cls = batch["cls"]  # [N, 1] or [N] - class IDs
    b_box = batch["bboxes"]  # [N, 4] - normalized (cx, cy, w, h)

    # Find first image in batch that has targets, or default to 0
    target_idx = 0
    if len(b_idx) > 0:
        unique_indices = torch.unique(b_idx)
        if len(unique_indices) > 0:
            target_idx = int(unique_indices[0].item())

    # Take selected image
    img = imgs[target_idx].transpose(1, 2, 0)

    # Scale normalized float images to [0, 255] before uint8 casting
    if img.max() <= 1.5:
        img = img * 255.0

    img = img.astype(np.uint8)

    h_img, w_img, _ = img.shape
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Filter targets for the selected image
    mask = b_idx == target_idx
    img_cls = b_cls[mask]
    img_box = b_box[mask]

    for j in range(len(img_cls)):
        cls_id = int(img_cls[j][0] if img_cls.ndim > 1 else img_cls[j])
        # x, y, w, h are normalized
        cx, cy, bw, bh = img_box[j]
        x1 = float((cx - bw / 2) * w_img)
        y1 = float((cy - bh / 2) * h_img)
        x2 = float((cx + bw / 2) * w_img)
        y2 = float((cy + bh / 2) * h_img)

        color = (255, 0, 0)  # Default red
        label = names.get(cls_id, f"class_{cls_id}")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), label, fill=color)

    img_pil.save(filename)
    print(f"Batch inspection image saved to {filename}")


def save_val_predictions(
    model, dataloader, device, names, save_dir, num_samples=4, conf_thres=0.25, prefix=""
):
    """
    Save sample validation images with GT boxes (green) and predictions (blue).
    Produces Ultralytics-style validation batches.
    """
    model.eval()
    saved = 0
    strides = [8, 16, 32]

    for batch in dataloader:
        if saved >= num_samples:
            break

        imgs = batch["img"].to(device).float() / 255.0

        with torch.no_grad():
            preds = model(imgs)

            # Handle output format
            is_deployed = False
            if isinstance(preds, tuple) and len(preds) == 6:
                preds_reg, preds_cls = preds[:3], preds[3:]
                is_deployed = True  # Already Sigmoided in model.forward()
            elif len(preds) == 2:
                preds_reg, preds_cls = preds
            else:
                continue

        # Process first image in batch
        img_np = (imgs[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        h_img, w_img = img_np.shape[:2]

        # Draw Ground Truth (GREEN)
        if "bboxes" in batch:
            b_idx = batch["batch_idx"]
            b_cls = batch["cls"]
            b_box = batch["bboxes"]

            for j in range(len(b_idx)):
                if b_idx[j] == 0:  # First image
                    cls_id = int(b_cls[j][0] if b_cls.ndim > 1 else b_cls[j])
                    cx, cy, bw, bh = b_box[j]
                    x1 = float((cx - bw / 2) * w_img)
                    y1 = float((cy - bh / 2) * h_img)
                    x2 = float((cx + bw / 2) * w_img)
                    y2 = float((cy + bh / 2) * h_img)
                    label = names.get(cls_id, f"class_{cls_id}")
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    draw.text((x1, y1 - 12), f"GT:{label}", fill=(0, 255, 0))

        # Decode predictions for first image
        # Generate anchors
        anchors_list = []
        for i, stride in enumerate(strides):
            H, W = int(h_img / stride), int(w_img / stride)
            sy, sx = torch.meshgrid(
                torch.arange(H, device=device) + 0.5,
                torch.arange(W, device=device) + 0.5,
                indexing="ij",
            )
            anchors = torch.stack((sx, sy), -1).reshape(-1, 2) * stride
            anchors_list.append(anchors)
        all_anchors = torch.cat(anchors_list)

        # Process predictions
        pred_boxes_list = []
        pred_cls_list = []
        for i, stride in enumerate(strides):
            reg = preds_reg[i][0]  # First image, [4, H, W]
            cls = preds_cls[i][0]  # First image, [nc, H, W]
            nc = cls.shape[0]  # Store num_classes before permute

            reg = reg.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
            cls = cls.permute(1, 2, 0).reshape(-1, nc)  # [H*W, nc]

            pred_boxes_list.append(reg)

            # Apply sigmoid only if not already applied in deploy mode
            if not is_deployed:
                pred_cls_list.append(cls.sigmoid())
            else:
                pred_cls_list.append(cls)

        pred_boxes = torch.cat(pred_boxes_list, 0)  # [N, 4]
        pred_cls = torch.cat(pred_cls_list, 0)  # [N, nc]

        # Decode boxes
        decoded = decode_bboxes(
            pred_boxes.unsqueeze(0),
            all_anchors,
            torch.cat(
                [
                    torch.full((a.shape[0], 1), s, device=device)
                    for a, s in zip(anchors_list, strides)
                ]
            ),
            xywh=False,
        )[0]  # [N, 4] in xyxy

        # Get top predictions
        conf, cls_idx = pred_cls.max(dim=-1)
        mask = conf > conf_thres

        if mask.any():
            top_boxes = decoded[mask][:50]  # Limit to 50 boxes
            top_conf = conf[mask][:50]
            top_cls = cls_idx[mask][:50]

            # Draw Predictions (BLUE)
            for k in range(len(top_boxes)):
                x1, y1, x2, y2 = top_boxes[k].cpu().numpy()
                cls_id = int(top_cls[k])
                c = float(top_conf[k])
                label = names.get(cls_id, f"class_{cls_id}")
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2)
                draw.text((x1, y2), f"{label}:{c:.2f}", fill=(0, 0, 255))

        # Save
        save_path = os.path.join(save_dir, f"{prefix}val_predictions_{saved}.jpg")
        img_pil.save(save_path)
        print(f"Validation prediction saved to {save_path}")
        saved += 1

    return saved


def plot_results(metrics, save_dir, prefix, title="Training Metrics"):
    """
    Plot evolution of Loss (Log Scale) and mAP.
    Separately plots Adaptive Weights if available.
    """
    epochs = range(1, len(metrics["loss"]) + 1)

    # 1. Main Results Plot (Loss + mAP)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Loss Plot (Log Scale)
    ax_loss = axes[0]
    ax_loss.set_ylabel("Total Loss (Log Scale)", fontsize=12, fontweight="bold")
    ax_loss.plot(
        epochs,
        metrics["loss"],
        color="#d62728",
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Total Loss",
    )
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="both", ls="--", alpha=0.4)
    ax_loss.legend(loc="upper right")
    ax_loss.set_title(f"{title} - Convergence", fontsize=14)

    # mAP Plot
    ax_map = axes[1]
    ax_map.set_ylabel("mAP@50", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_map.plot(
        epochs,
        metrics["map"],
        color="#1f77b4",
        marker="s",
        linestyle="-",
        linewidth=2,
        label="mAP@50",
    )
    ax_map.grid(True, which="both", ls="--", alpha=0.4)
    ax_map.legend(loc="lower right")

    filename_main = os.path.join(save_dir, f"results_{prefix}.png")
    plt.tight_layout()
    plt.savefig(filename_main, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Main results plot saved to {filename_main}")

    # 2. Adaptive Weights Plot (Separate File)
    if "weights" in metrics and len(metrics["weights"]) > 0:
        fig_w, ax_w = plt.subplots(figsize=(10, 6))
        ax_w.set_ylabel("Loss Weights (Lambda)", fontsize=12, fontweight="bold")
        ax_w.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_w.set_title(f"{title} - Adaptive Weights", fontsize=14)

        weights_np = np.array(metrics["weights"])
        if weights_np.ndim == 2 and weights_np.shape[1] >= 3:
            labels = ["SDF (Feat)", "Logit (Class)", "Box (Reg)", "Task (GT)"]
            colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

            # Limit plotting to min(available weights, available labels)
            num_w = min(weights_np.shape[1], len(labels))
            for i in range(num_w):
                ax_w.plot(
                    epochs,
                    weights_np[:, i],
                    color=colors[i],
                    linestyle="-",
                    linewidth=2,
                    label=labels[i],
                )

        ax_w.grid(True, which="both", ls="--", alpha=0.4)
        ax_w.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        filename_weights = os.path.join(save_dir, f"weights_{prefix}.png")
        plt.tight_layout()
        plt.savefig(filename_weights, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Weights plot saved to {filename_weights}")

    # 3. Loss Breakdown Plot (DIAGNOSTICS for Hallucination Analysis)
    # Only generate if sub-component data is available
    if "l_sdf" in metrics and len(metrics.get("l_sdf", [])) > 0:
        fig_bd, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # Row 1: All 4 Main Losses (Log Scale)
        ax1 = axes[0]
        ax1.set_ylabel("Loss (Log)", fontsize=11, fontweight="bold")
        ax1.set_title(f"{title} - Loss Breakdown", fontsize=14)
        ax1.plot(
            epochs, metrics["l_sdf"], "o-", color="#ff7f0e", linewidth=2, label="SDF", markersize=4
        )
        ax1.plot(
            epochs,
            metrics["l_logit"],
            "s-",
            color="#2ca02c",
            linewidth=2,
            label="Logit",
            markersize=4,
        )
        ax1.plot(
            epochs, metrics["l_box"], "^-", color="#9467bd", linewidth=2, label="Box", markersize=4
        )
        ax1.plot(
            epochs,
            metrics["l_task"],
            "d-",
            color="#8c564b",
            linewidth=2,
            label="Task",
            markersize=4,
        )
        ax1.set_yscale("log")
        ax1.grid(True, which="both", ls="--", alpha=0.4)
        ax1.legend(loc="upper right", ncol=2)

        # Row 2: SDF Breakdown (Obj vs Silence) for spatial activation diagnosis
        ax2 = axes[1]
        ax2.set_ylabel("SDF Sub-Losses", fontsize=11, fontweight="bold")
        ax2.set_title("SDF Breakdown: Object Mimicry + Magnitude Anchor", fontsize=12)
        ax2.plot(
            epochs,
            metrics["l_sdf_obj"],
            "o-",
            color="#1f77b4",
            linewidth=2,
            label="Obj (Foreground)",
            markersize=4,
        )
        ax2.plot(
            epochs,
            metrics["l_sdf_mag"],
            "^-",
            color="#7f7f7f",
            linewidth=2,
            label="Magnitude Anchor",
            markersize=4,
        )
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        ax2.legend(loc="upper right")

        # Row 3: Logit Breakdown (Certainty vs BCE)
        ax3 = axes[2]
        ax3.set_ylabel("Logit Sub-Losses", fontsize=11, fontweight="bold")
        ax3.set_title("Logit Breakdown: Certainty Matching vs BCE Regularizer", fontsize=12)
        ax3.plot(
            epochs,
            metrics["l_logit_cert"],
            "o-",
            color="#17becf",
            linewidth=2,
            label="Certainty (Log-Sigmoid)",
            markersize=4,
        )
        ax3.plot(
            epochs,
            metrics["l_logit_bce"],
            "s-",
            color="#bcbd22",
            linewidth=2,
            label="BCE Auxiliary Term",
            markersize=4,
        )
        ax3.grid(True, which="both", ls="--", alpha=0.4)
        ax3.legend(loc="upper right")

        # Row 4: Box Breakdown (IoU + Conf Weight)
        ax4 = axes[3]
        ax4.set_ylabel("Box Metrics", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax4.set_title("Box Breakdown: IoU & Teacher Confidence", fontsize=12)
        ax4.plot(
            epochs,
            metrics["box_iou"],
            "o-",
            color="#2ca02c",
            linewidth=2,
            label="IoU (Mean)",
            markersize=4,
        )
        ax4.plot(
            epochs,
            metrics["box_conf"],
            "s-",
            color="#e377c2",
            linewidth=2,
            label="Conf Weight (Teacher)",
            markersize=4,
        )
        ax4.set_ylim(0, 1.1)  # IoU and confidence are in [0, 1]
        ax4.grid(True, which="both", ls="--", alpha=0.4)
        ax4.legend(loc="upper right")

        filename_breakdown = os.path.join(save_dir, f"loss_breakdown_{prefix}.png")
        plt.tight_layout()
        plt.savefig(filename_breakdown, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss breakdown plot saved to {filename_breakdown}")
