# Ultralytics Inference Compatibility

This guide explains how to run UNINA-DLA models using the Ultralytics-compatible inference wrapper.

---

## Why This Is Needed

Standard Ultralytics `DetectionPredictor` expects a single concatenated tensor output. UNINA-DLA deploy export produces a 6-tensor tuple (`reg3, reg4, reg5, cls3, cls4, cls5`) optimized for zero-copy DLA execution.

The wrapper (`unina_dla/ultralytics_wrapper.py`) decodes these direct regression outputs while preserving the standard Ultralytics `Results` API (bounding boxes, confidences, class IDs, plotting).

---

## Relevant Files

- `unina_dla/ultralytics_wrapper.py`: Core wrapper implementation
- `scripts/infer_ultralytics.py`: Inference CLI script
- `scripts/convert_to_ultralytics_pt.py`: Checkpoint converter to `.pt`
- `scripts/embed_engine_metadata.py`: Embed dataset metadata into TensorRT engines

---

## 1. Convert Training Checkpoint to `.pt`

```bash
python scripts/convert_to_ultralytics_pt.py \
  --checkpoint runs/exp/best_qat.pth \
  --output runs/exp/unina_dla_ultra.pt \
  --num-classes 4 \
  --names unina_dla/config/unina_dla_data.yaml
```

---

## 2. Add Metadata to TensorRT Engine

When building an engine via `trtexec`, embed metadata so the Ultralytics wrapper can automatically read class names and strides:

```bash
python scripts/embed_engine_metadata.py \
  --engine runs/exp/unina_dla.engine \
  --names unina_dla/config/unina_dla_data.yaml \
  --imgsz 640 \
  --stride 32
```

Alternatively, use the provided build script with `--embed-metadata`:

```bash
python scripts/build_engine.py \
  --onnx runs/exp/unina_dla.onnx \
  --target dla \
  --dla_core 0 \
  --embed-metadata \
  --names unina_dla/config/unina_dla_data.yaml \
  --meta-imgsz 768,480
```

---

## 3. Run Inference

With a converted `.pt` model:

```bash
python scripts/infer_ultralytics.py \
  --model runs/exp/unina_dla_ultra.pt \
  --source path/to/image_or_video \
  --device 0
```

With a TensorRT `.engine`:

```bash
python scripts/infer_ultralytics.py \
  --model runs/exp/unina_dla.engine \
  --source path/to/image_or_video \
  --device 0
```

### O2O vs NMS Mode
- **Default**: Native O2O direct filtering (confidence threshold + top-k, no NMS suppression).
- **Comparison NMS**: Pass `--nms` to apply standard IoU suppression for comparison.
