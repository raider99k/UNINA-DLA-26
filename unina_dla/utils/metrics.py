"""
UNINA-DLA Detection Metrics.

COMPARISON WITH OFFICIAL ULTRALYTICS DetMetrics:
--------------------------------------------------
| Feature                     | Official (DetMetrics)            | Custom (UNINAMetrics)           |
|-----------------------------|----------------------------------|----------------------------------|
| Matching Algorithm          | Iterative per-threshold          | Iterative per-threshold (SAME)   |
| Sorting Criterion           | IoU (descending)                 | IoU (descending) (SAME)          |
| AP Calculation              | ap_per_class()                   | ap_per_class() (SAME - imported) |
| Confusion Matrix            | ConfusionMatrix class            | Inline np.ndarray (simpler)      |
| Plotting                    | plot_pr_curve, plot_mc_curve     | Custom matplotlib plots          |
| Input Format                | Dict-based batches               | List of tensors (custom format)  |
| Integration                 | Requires YOLO wrapper            | Works with raw nn.Module         |

WHY WE KEEP THIS CUSTOM CLASS:
------------------------------
1. INPUT FLEXIBILITY: Takes raw tensor lists, not Ultralytics batch dicts.
2. SIMPLER INTEGRATION: Direct usage with our custom validator.
3. OFFICIAL CORE: Uses `ultralytics.utils.metrics.ap_per_class` for the heavy math.
4. OFFICIAL MATCHING: Now uses iterative thresholding matching (as of 2026-01).

ACCURACY: mAP calculations are now identical to official Ultralytics benchmarks.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None
import torch

try:
    from ultralytics.utils.metrics import ap_per_class
except ImportError:
    ap_per_class = None


class UNINAMetrics:
    def __init__(
        self,
        num_classes=1,
        plot=False,
        names=None,
        save_dir=None,
        prefix="",
        cm_conf=0.25,
        cm_iou=0.5,
    ):
        self.num_classes = num_classes
        self.plot = plot
        self.save_dir = save_dir
        self.prefix = prefix
        self.file_prefix = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
        self.names = names or {i: f"class{i}" for i in range(num_classes)}
        self.stats = []  # list of (correct, conf, pred_cls, target_cls)
        # Confusion matrix aligned with Ultralytics:
        # rows=Predicted class, cols=True class, +1 background row/col.
        self.confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        self.total_gt = 0  # Track total ground truth separately to avoid undercounting empty images
        self.cm_conf = float(cm_conf)
        self.cm_iou = float(cm_iou)

    def update(self, preds, targets):
        """
        preds: list of tensors [N, 6] (xyxy, conf, cls) - one per image
        targets: list of tensors [M, 5] (cls, xyxy) - one per image
        """
        for si, pred in enumerate(preds):
            labels = targets[si]

            # Evaluate mAP matching using iterative IoU thresholding.
            iouv = torch.linspace(0.5, 0.95, 10, device=pred.device)

            # Update total ground truth count
            self.total_gt += labels.shape[0]

            correct = self._process_batch(pred, labels, iouv)
            self._update_confusion_matrix(pred, labels)

            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    def _process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix using official Ultralytics matching logic.

        Iterates over each IoU threshold and re-computes greedy matching,
        ensuring accurate mAP50-95 calculation. This matches the behavior of
        ultralytics.engine.validator.BaseValidator.match_predictions.

        Arguments:
            detections (Tensor): (N, 6) [x1, y1, x2, y2, conf, class]
            labels (Tensor): (M, 5) [class, x1, y1, x2, y2]
            iouv (Tensor): IoU thresholds [0.5, 0.55, ..., 0.95]
        Returns:
            correct (Tensor): (N, 10) - True if detection i is correct at IoU threshold j
        """
        import torchvision

        correct = torch.zeros(
            detections.shape[0], iouv.shape[0], dtype=torch.bool, device=detections.device
        )

        # IoU matrix [M, N] where M=num_gt, N=num_pred
        iou = torchvision.ops.box_iou(labels[:, 1:], detections[:, :4])

        # Zero out wrong classes for AP matching (Ultralytics behavior).
        correct_class = labels[:, 0:1] == detections[:, 5]  # [M, N]
        iou = iou * correct_class
        iou_np = iou.cpu().numpy()

        # Iterate over each IoU threshold (official logic)
        for i, threshold in enumerate(iouv.cpu().tolist()):
            matches = np.nonzero(iou_np >= threshold)  # (gt_idx_array, pred_idx_array)
            matches = np.array(matches).T  # [K, 2]

            if matches.shape[0]:
                if matches.shape[0] > 1:
                    # Sort by IoU (descending) - official behavior
                    matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    # Unique by pred index (keep first = highest IoU)
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # Re-sort by IoU (descending) because np.unique orders by pred_idx ascending
                    matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    # Unique by GT index (keep first = highest IoU)
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                correct[matches[:, 1].astype(int), i] = True

        return correct

    def _update_confusion_matrix(self, detections, labels):
        """
        Update confusion matrix using Ultralytics-compatible detection semantics:
        - Apply confidence filtering (cm_conf)
        - Match class-agnostically at IoU > cm_iou
        - matrix[pred_cls, true_cls] for matches
        - matrix[num_classes, true_cls] for FN
        - matrix[pred_cls, num_classes] for FP
        """
        import torchvision

        # Filter detections for confusion matrix only.
        detections = (
            detections[detections[:, 4] > self.cm_conf] if detections.shape[0] else detections
        )
        no_pred = detections.shape[0] == 0

        if labels.shape[0] == 0:
            if not no_pred:
                pred_classes = detections[:, 5].long().cpu().numpy()
                np.add.at(self.confusion_matrix, (pred_classes, self.num_classes), 1)  # FP
            return

        if no_pred:
            gt_classes = labels[:, 0].long().cpu().numpy()
            np.add.at(self.confusion_matrix, (self.num_classes, gt_classes), 1)  # FN
            return

        iou = torchvision.ops.box_iou(labels[:, 1:], detections[:, :4])  # [M, N]
        x = torch.where(iou > self.cm_iou)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3), dtype=np.float32)

        gt_classes = labels[:, 0].long().cpu().numpy()
        det_classes = detections[:, 5].long().cpu().numpy()
        has_match = matches.shape[0] > 0
        if has_match:
            m0, m1, _ = matches.transpose().astype(int)

        for i, gc in enumerate(gt_classes):
            j = (m0 == i) if has_match else np.zeros((0,), dtype=bool)
            if has_match and j.sum() == 1:
                dc = det_classes[m1[j].item()]
                self.confusion_matrix[dc, gc] += 1
            else:
                self.confusion_matrix[self.num_classes, gc] += 1

        for i, dc in enumerate(det_classes):
            if not (has_match and any(m1 == i)):
                self.confusion_matrix[dc, self.num_classes] += 1

    def compute(self):
        """Compute metrics and store results for plotting."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy

        # Count total predictions and ground truths for diagnostics
        total_preds = len(stats[1]) if len(stats) > 1 else 0
        total_gt = len(stats[3]) if len(stats) > 3 else 0

        if len(stats) and stats[0].any():
            tp, conf, pred_cls, target_cls = stats
            if ap_per_class is not None:
                # ap_per_class returns:
                # (tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values)
                # ap is [nc, 10] where columns are IoU thresholds from 0.5 to 0.95
                results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=self.names)

                # Correct index extraction (ultralytics API):
                # results[5] = ap array [nc, 10] (10 IoU thresholds)
                # results[6] = unique_classes (class indices, NOT ap50!)
                ap_all = results[5]  # [nc, 10]
                ap50_per_class = (
                    ap_all[:, 0] if len(ap_all) else np.array([])
                )  # IoU=0.5 is column 0
                ap_per_class_mean = (
                    ap_all.mean(axis=1) if len(ap_all) else np.array([])
                )  # Mean over IoU thresholds

                # Store for plotting
                self._results = {
                    "tp": tp,  # Store RAW boolean array for plotting functions
                    "tp_curve": results[0],  # Store computed curve as separate key if needed
                    "fp": results[1],
                    "precision": results[2],
                    "recall": results[3],
                    "f1": results[4],
                    "ap_class": results[6],  # unique_classes (class indices)
                    "ap50": ap50_per_class,  # AP @ IoU=0.5 per class
                    "ap": ap_per_class_mean,  # AP @ IoU=0.5:0.95 per class
                    "conf": conf,
                    "pred_cls": pred_cls,
                    "target_cls": target_cls,
                }

                # Extract mAP50 and mAP50-95
                ap50 = ap50_per_class.mean() if len(ap50_per_class) else 0.0
                ap = ap_per_class_mean.mean() if len(ap_per_class_mean) else 0.0
                mp = results[2].mean() if len(results[2]) else 0.0
                mr = results[3].mean() if len(results[3]) else 0.0

                # Print per-class AP table
                self._print_per_class_ap(results[6], ap50_per_class, ap_per_class_mean)

                return {
                    "mAP50": float(ap50),
                    "mAP50-95": float(ap),
                    "precision": float(mp),
                    "recall": float(mr),
                }

        # No valid predictions - set empty _results for plotting
        print(
            f"Validation: {total_preds} predictions, {total_gt} ground truths, 0 matches at IoU>=0.5"
        )
        self._results = {
            "tp": np.array([]),
            "conf": np.array([]),
            "pred_cls": np.array([]),
            "target_cls": np.array([]),
            "ap50": np.array([]),
            "ap": np.array([]),
            "precision": np.array([]),
            "recall": np.array([]),
            "f1": np.array([]),
            "ap_class": np.array([]),
            "fp": np.array([]),
        }
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

    def _print_per_class_ap(self, classes, ap50, ap):
        """Print per-class AP table."""
        if len(classes) == 0:
            return
        print(f"{'Class':<20} {'AP50':>10} {'AP50-95':>10}")
        print("-" * 42)
        for i, cls_idx in enumerate(classes):
            name = self.names.get(int(cls_idx), f"class{int(cls_idx)}")
            print(f"{name:<20} {ap50[i] * 100:>9.1f}% {ap[i] * 100:>9.1f}%")
        print("-" * 42)
        print(f"{'Mean':<20} {ap50.mean() * 100:>9.1f}% {ap.mean() * 100:>9.1f}%")

    def _get_phase_title(self, base_title):
        """Helper to format phase name for plot titles."""
        if not self.prefix:
            return base_title

        # Clean prefix: remove trailing underscore, replace others with space
        clean_prefix = self.prefix.rstrip("_").replace("_", " ")

        # Special casing for QAT
        if clean_prefix.lower() == "qat":
            phase_name = "QAT"
        else:
            phase_name = clean_prefix.title()

        return f"{phase_name} {base_title}"

    def plot_pr_curve(self, filename=None, verbose=True):
        """Plot Precision-Recall curve for all classes."""
        if self._results is None:
            print("No results to plot. Run compute() first.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}PR_curve.png"
            if self.save_dir
            else f"{self.file_prefix}PR_curve.png"
        )

        # Handle empty results
        if len(self._results["tp"]) == 0:
            ax.text(
                0.5,
                0.5,
                "No predictions matched ground truth",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(self._get_phase_title("Precision-Recall Curve (No Data)"))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            if verbose:
                print(f"PR curve (empty) saved to {save_path}")
            return save_path

        # Get raw data
        tp_arr = self._results["tp"]  # [N, 10] boolean array
        conf = self._results["conf"]  # [N] confidence scores
        ap50 = self._results["ap50"]

        # Sort by confidence (descending)
        sort_idx = np.argsort(-conf)
        tp_sorted = tp_arr[sort_idx]

        # Use IoU=0.5 (first column or entire array if 1D)
        if tp_sorted.ndim > 1:
            tp_50 = tp_sorted[:, 0]  # IoU=0.5
        else:
            tp_50 = tp_sorted

        # Cumulative TP and FP
        # n_gt = len(target_cls)  # Original: possibly undercounts if empty images are skipped
        n_gt = self.total_gt  # Revised: explicitly counted in update()
        cum_tp = np.cumsum(tp_50)
        cum_fp = np.arange(1, len(tp_50) + 1) - cum_tp

        # Precision and Recall at each point
        precision = cum_tp / (cum_tp + cum_fp + 1e-16)
        recall = cum_tp / (n_gt + 1e-16)

        # Plot PR curve
        mean_ap = ap50.mean() if hasattr(ap50, "mean") else float(ap50)
        ax.plot(
            recall, precision, linewidth=2, color="blue", label=f"All Classes (mAP50={mean_ap:.3f})"
        )
        ax.fill_between(recall, precision, alpha=0.2, color="blue")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(self._get_phase_title("Precision-Recall Curve"))
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}PR_curve.png"
            if self.save_dir
            else f"{self.file_prefix}PR_curve.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        if verbose:
            print(f"PR curve saved to {save_path}")
        return save_path

    def plot_f1_curve(self, filename=None, verbose=True):
        """Plot F1 score vs Confidence threshold."""
        if self._results is None:
            print("No results to plot. Run compute() first.")
            return None

        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}F1_curve.png"
            if self.save_dir
            else f"{self.file_prefix}F1_curve.png"
        )

        # Handle empty results
        if len(self._results["tp"]) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No predictions matched ground truth",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence Threshold")
            ax.set_ylabel("F1 Score")
            ax.set_title(self._get_phase_title("F1-Confidence Curve (No Data)"))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            if verbose:
                print(f"F1 curve (empty) saved to {save_path}")
            return save_path

        # Get raw data
        tp_arr = self._results["tp"]  # [N, 10] boolean array (correct at each IoU threshold)
        conf = self._results["conf"]  # [N] confidence scores

        # Calculate F1 at different confidence thresholds
        thresholds = np.linspace(0.0, 1.0, 101)
        # n_gt = len(target_cls)  # Total ground truth (approximation)
        n_gt = self.total_gt  # Revised: explicitly counted in update()

        f1_scores = []
        for thresh in thresholds:
            # Mask predictions above threshold
            mask = conf >= thresh
            if mask.sum() == 0:
                f1_scores.append(0.0)
                continue

            # TP at IoU=0.5 (first column)
            tp_at_thresh = tp_arr[mask, 0].sum() if tp_arr.ndim > 1 else tp_arr[mask].sum()
            fp_at_thresh = mask.sum() - tp_at_thresh

            # Precision and Recall
            precision = tp_at_thresh / (tp_at_thresh + fp_at_thresh + 1e-16)
            recall = tp_at_thresh / (n_gt + 1e-16)

            # F1
            f1 = 2 * precision * recall / (precision + recall + 1e-16)
            f1_scores.append(float(f1))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, f1_scores, linewidth=2, color="blue")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        best_f1 = max(f1_scores) if f1_scores else 0.0
        ax.set_title(self._get_phase_title(f"F1-Confidence Curve (Best F1={best_f1:.3f})"))
        ax.grid(True, alpha=0.3)

        # Mark best F1
        if f1_scores:
            best_idx = np.argmax(f1_scores)
            ax.axvline(x=thresholds[best_idx], color="red", linestyle="--", alpha=0.7)
            ax.scatter([thresholds[best_idx]], [f1_scores[best_idx]], color="red", s=100, zorder=5)

        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}F1_curve.png"
            if self.save_dir
            else f"{self.file_prefix}F1_curve.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        if verbose:
            print(f"F1 curve saved to {save_path}")
        return save_path

    def plot_confusion_matrix(self, filename=None, verbose=True):
        """Plot and save the confusion matrix (raw counts)."""
        labels = [self.names.get(i, f"class{i}") for i in range(self.num_classes)] + ["background"]

        fig, ax = plt.subplots(figsize=(10, 8))
        if sns is not None:
            sns.heatmap(
                self.confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
        else:
            ax.imshow(self.confusion_matrix, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(self._get_phase_title("Confusion Matrix (Ultralytics-Compatible, Raw Counts)"))

        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}confusion_matrix.png"
            if self.save_dir
            else f"{self.file_prefix}confusion_matrix.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        if verbose:
            print(f"Confusion matrix saved to {save_path}")
        return save_path

    def plot_confusion_matrix_normalized(self, filename=None, verbose=True):
        """
        Plot and save the normalized confusion matrix.

        Ultralytics-compatible normalization: normalize by columns (true classes).
        """
        labels = [self.names.get(i, f"class{i}") for i in range(self.num_classes)] + ["background"]

        # Normalize by column (true class) to match Ultralytics ConfusionMatrix.plot().
        cm = self.confusion_matrix.astype(np.float32)
        col_sums = cm.sum(axis=0, keepdims=True)
        # Avoid division by zero.
        col_sums = np.maximum(col_sums, 1e-6)
        cm_normalized = cm / col_sums

        fig, ax = plt.subplots(figsize=(10, 8))
        if sns is not None:
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2%",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                vmin=0,
                vmax=1,
            )
        else:
            ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(
            self._get_phase_title(
                "Normalized Confusion Matrix (Column = True, Ultralytics-Compatible)"
            )
        )

        save_path = filename or (
            self.save_dir + f"/{self.file_prefix}confusion_matrix_normalized.png"
            if self.save_dir
            else f"{self.file_prefix}confusion_matrix_normalized.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        if verbose:
            print(f"Normalized confusion matrix saved to {save_path}")
        return save_path

    def plot_all(self, save_dir=None, prefix=None, verbose=True):
        """Generate all plots: Confusion Matrix (raw + normalized), PR Curve, F1 Curve."""
        save_dir = save_dir or self.save_dir or "."
        raw_p = prefix if prefix is not None else self.prefix
        p = f"{raw_p}_" if raw_p and not raw_p.endswith("_") else raw_p

        paths = []
        paths.append(self.plot_confusion_matrix(f"{save_dir}/{p}confusion_matrix.png", verbose=False))
        paths.append(
            self.plot_confusion_matrix_normalized(
                f"{save_dir}/{p}confusion_matrix_normalized.png", verbose=False
            )
        )
        paths.append(self.plot_pr_curve(f"{save_dir}/{p}PR_curve.png", verbose=False))
        paths.append(self.plot_f1_curve(f"{save_dir}/{p}F1_curve.png", verbose=False))

        if verbose:
            print(
                f"Saved validation plots ({p}confusion_matrix*.png, {p}PR_curve.png, {p}F1_curve.png) to {save_dir}"
            )
        return paths
