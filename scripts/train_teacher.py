import argparse
import os
import sys

import yaml
from ultralytics import YOLO

# Ensure project root is in Python path (for Kaggle/remote execution)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts/ -> project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def train_teacher(
    data_yaml,
    epochs=300,
    batch_size=32,
    workers=8,
    project="unina_dla_teacher",
    name=None,
    data_root=None,
    lr0=0.01,
    imgsz=640,
    model_size="x",
    compile_model=True,
    linear_prob=True,
    is_rect=None,
):
    """
    Train the YOLO26 teacher model on the FSG dataset.
    """
    # Construct model name based on size
    model_name = f"yolo26{model_size}.pt"

    # Auto-detect rectangular if not explicitly set
    if is_rect is None:
        is_rect = isinstance(imgsz, (list, tuple)) and len(imgsz) == 2 and imgsz[0] != imgsz[1]

    # Default name if not provided
    if name is None:
        if isinstance(imgsz, (list, tuple)):
            res_str = "x".join(map(str, imgsz))
        else:
            res_str = str(imgsz)
        name = f"yolo26{model_size}_teacher_{res_str}"

    # Load pretrained YOLO26 model
    print(f"Loading teacher model: {model_name}")
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Handle dataset path override & Robust Resolution
    data_arg = data_yaml
    temp_yaml_path = None

    # Always read and process paths to ensure safety (expanduser, etc)
    # even if data_root is NOT provided
    print(f"Processing dataset config: {data_yaml}")
    with open(data_yaml, "r") as f:
        data_dict = yaml.safe_load(f)

    # Resolving Root
    if data_root is not None:
        print(f"Overriding dataset path with: {data_root}")
        data_dict["path"] = data_root

    # Expand user path '~' if present in 'path'
    if "path" in data_dict and isinstance(data_dict["path"], str):
        data_dict["path"] = os.path.expanduser(data_dict["path"])

    resolved_root = data_dict.get("path", "")

    # Path resolution and directory fallback logic
    for k in ["train", "val", "test"]:
        if k in data_dict:
            path_val = data_dict[k]
            # 1. Resolve Path
            if isinstance(path_val, str) and not os.path.isabs(path_val):
                data_dict[k] = os.path.join(resolved_root, path_val)

            # 2. Check Existence & Fallback
            if isinstance(data_dict[k], str) and not os.path.exists(data_dict[k]):
                print(f"Warning: {k} path not found at {data_dict[k]}. Checking fallbacks...")

                fallbacks = {
                    "val": [
                        "val/images",
                        "valid/images",
                        "validation/images",
                        "val",
                        "valid",
                        "validation",
                    ],
                    "test": ["test/images", "testing/images", "test", "testing"],
                }

                found = False
                if k in fallbacks:
                    for fb in fallbacks[k]:
                        fb_path = os.path.join(resolved_root, fb)
                        if os.path.exists(fb_path):
                            print(f"Found {k} split at fallback: {fb_path}")
                            data_dict[k] = fb_path
                            found = True
                            break

                if not found:
                    if k == "test":
                        print("Warning: Test set path not found. Disabling testing for teacher.")
                        del data_dict[k]
                    else:
                        print(f"Error: {k} set path not found at {data_dict[k]}!")
                        # We don't exit here, we let YOLO try and fail or user see the warning.

    # Write to a temporary YAML file (Ultralytics 8.4.4+ doesn't accept dicts properly)
    import tempfile

    temp_yaml_fd, temp_yaml_path = tempfile.mkstemp(suffix=".yaml", prefix="data_override_")
    os.close(temp_yaml_fd)
    with open(temp_yaml_path, "w") as f:
        yaml.dump(data_dict, f)
    data_arg = temp_yaml_path
    print(f"Created temporary robust data config: {temp_yaml_path}")

    # Pass augmentation parameters directly to override YAML defaults
    with open(data_yaml, "r") as f:
        hyp_config = yaml.safe_load(f)

    # YOLO26 optimizer selection

    # Known Ultralytics Hyperparameters
    hyp_keys = [
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "auto_augment",
        "erasing",
        "crop_fraction",
    ]

    train_args = {k: v for k, v in hyp_config.items() if k in hyp_keys}
    if train_args:
        print(f"Loaded Augmentations from YAML: {train_args}")

    # Train the model
    if linear_prob:
        print("\n" + "=" * 40)
        print("STAGE 1: HEAD ALIGNMENT (Warmup)")
        print("=" * 40)

        if epochs > 20:
            warmup_epochs = 10
        else:
            # Dynamic 20% logic for short debug runs (e.g. 5 epochs -> 1 warmup)
            warmup_epochs = max(1, int(epochs * 0.2))

        print(
            f"Total Epochs: {epochs} | Warmup: {warmup_epochs} | Fine-Tuning: {epochs - warmup_epochs}"
        )

        warmup_name = f"{name}_warmup" if name else "warmup"

        # Phase 1: Freeze Backbone, Auto Optimizer, NO AUGMENTATIONS
        model.train(
            data=data_arg,
            epochs=warmup_epochs,
            batch=batch_size,
            imgsz=imgsz,
            rect=is_rect,  # Enable rectangular training if h != w
            device=0,
            workers=workers,
            project=project,
            name=warmup_name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",  # Safest for frozen warmup
            lr0=0.01,
            freeze=10,  # Freeze backbone during warmup
        )

        # Load best weights from warmup
        # Handle absolute paths (e.g., Colab Drive paths) vs relative paths
        if os.path.isabs(project):
            warmup_ckpt = os.path.join(project, warmup_name, "weights", "best.pt")
        else:
            warmup_ckpt = os.path.join(PROJECT_ROOT, project, warmup_name, "weights", "best.pt")
        print(f"\nStage 1 Complete. Reloading weights from {warmup_ckpt} for Fine-Tuning...")

        # Reload model with updated head weights
        try:
            model = YOLO(warmup_ckpt)
        except Exception as e:
            print(f"Failed to reload warmup checkpoint: {e}. Exiting.")
            sys.exit(1)

        print("\n" + "=" * 40)
        print("STAGE 2: FULL FINE-TUNING")
        print("=" * 40)

        # Calculate remaining epochs
        remaining_epochs = max(1, epochs - warmup_epochs)

        # Phase 2: Unfreeze All, MuSGD, WITH AUGMENTATIONS
        results = model.train(
            data=data_arg,
            epochs=remaining_epochs,  # Corrected epoch math
            freeze=0,  # Unfreeze everything
            batch=batch_size,
            imgsz=imgsz,
            rect=is_rect,  # Enable rectangular training if h != w
            device=0,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer="MuSGD",  # YOLO26 optimal solver
            lr0=lr0 / 2.0,  # Gently shift the backbone
            cos_lr=True,
            compile=compile_model,
            patience=25,  # Prevent over-confidence
            **train_args,  # INJECT HEAVY AUGMENTATIONS HERE
        )

    else:
        # Standard Single-Stage Training (The "Call It A Day" approach)
        print("\n" + "=" * 40)
        print("STANDARD TRAINING (Single Stage)")
        print("=" * 40)
        results = model.train(
            data=data_arg,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            rect=is_rect,  # Enable rectangular training if h != w
            device=0,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer="MuSGD",
            lr0=lr0,
            cos_lr=True,
            compile=compile_model,
            patience=25,  # Added to standard run as well
            **train_args,
        )

    print("Teacher training completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 Teacher")
    parser.add_argument(
        "--data",
        type=str,
        default="unina_dla/config/teacher_data.yaml",
        help="Path to dataset yaml",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Override dataset root path (optional)"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument(
        "--project",
        type=str,
        default="unina_dla_teacher",
        help="Output project directory (can be absolute path for Colab/Drive)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: auto-generated from model size and resolution)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (MuSGD)")
    parser.add_argument("--imgsz", type=str, default="640", help="Image size: 640 or 768,480")
    parser.add_argument(
        "--model",
        type=str,
        default="x",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO26 model size (n, s, m, l, x)",
    )
    parser.add_argument("--compile", action="store_true", default=True, help="Enable torch.compile")
    parser.add_argument(
        "--no-compile", action="store_false", dest="compile", help="Disable torch.compile"
    )
    parser.add_argument(
        "--linear-prob",
        action="store_true",
        default=True,
        help="Enable Two-Stage Training (Linear Probing -> Fine-Tuning)",
    )

    args = parser.parse_args()

    # Parse image size (Support Rectangular)
    imgsz_parts = [int(x.strip()) for x in args.imgsz.split(",")]
    if len(imgsz_parts) == 1:
        imgsz = imgsz_parts[0]
        is_rect = False
        print(f"Training Teacher at imgsz={imgsz} (Square)")
    else:
        imgsz = imgsz_parts
        is_rect = True
        print(f"Training Teacher at imgsz={imgsz} (Rectangular, rect=True)")

    train_teacher(
        args.data,
        args.epochs,
        args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        data_root=args.data_path,
        lr0=args.lr,
        imgsz=imgsz,
        model_size=args.model,
        compile_model=args.compile,
        linear_prob=args.linear_prob,
        is_rect=is_rect,
    )
