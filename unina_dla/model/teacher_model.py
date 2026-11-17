from ultralytics import YOLO


def get_teacher_model(weights_path=None):
    """
    Initialize and return the YOLO26-X teacher model.
    If weights_path is provided, loads the weights.
    Otherwise, loads the pretrained yolo26x.pt.
    """
    try:
        src = weights_path or "yolo26x.pt"
        model = YOLO(src)
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"Teacher model loaded: {src} ({n_params:,} parameters)")
        return model
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return None


if __name__ == "__main__":
    # Smoke test
    get_teacher_model()
