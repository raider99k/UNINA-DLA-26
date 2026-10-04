"""
UNINA-DLA Benchmark Runner Module.

Executes unified benchmarking across candidate models:
1. Complexity & Footprint: Params (M), GFLOPs, Disk Size (MB), Peak VRAM (MB).
2. Speed & Latency: Microsecond-precision CUDA event timing, warm-up stabilization,
   distribution statistics (Mean, Median, Min, Max, P90, P95, P99, Std), and FPS.
3. Intelligence & Quality: Official Ultralytics-aligned mAP@0.50, mAP@0.50:0.95,
   Precision, Recall, F1 on target validation datasets.
"""

from __future__ import annotations

import gc
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from unina_dla.benchmark.config import BenchmarkConfig, ModelSpec
from unina_dla.benchmark.model_adapter import BaseModelAdapter, load_model_adapter


@dataclass
class LatencyStats:
    """Latency distribution statistics for a pipeline stage (in milliseconds)."""
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    fps: float = 0.0
    raw_samples: List[float] = field(default_factory=list)

    @classmethod
    def compute(cls, samples: List[float], batch_size: int = 1) -> LatencyStats:
        if not samples:
            return cls()
        arr = np.array(samples, dtype=np.float64)
        mean_val = float(np.mean(arr))
        fps_val = (batch_size * 1000.0 / mean_val) if mean_val > 0 else 0.0
        return cls(
            mean_ms=mean_val,
            median_ms=float(np.median(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            fps=fps_val,
            raw_samples=samples,
        )


@dataclass
class ModelBenchmarkResult:
    """Consolidated benchmark result for a single candidate model."""
    name: str
    path: str
    backend: str
    device: str
    batch_size: int
    resolution: Tuple[int, int]
    params_m: Optional[float] = None
    gflops: Optional[float] = None
    weight_size_mb: Optional[float] = None
    disk_size_mb: float = 0.0
    peak_vram_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None

    # Latency breakdowns (ms)
    preprocess: LatencyStats = field(default_factory=LatencyStats)
    inference: LatencyStats = field(default_factory=LatencyStats)
    postprocess: LatencyStats = field(default_factory=LatencyStats)
    e2e: LatencyStats = field(default_factory=LatencyStats)

    # Accuracy / Intelligence
    accuracy_evaluated: bool = False
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    class_metrics: Optional[Dict[str, Any]] = None

    error: Optional[str] = None


class BenchmarkRunner:
    """Orchestrates comprehensive multi-model benchmarking."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dataset_info = self._load_dataset_info()

    def _resolve_device(self, dev_str: str) -> torch.device:
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") in ("-1", ""):
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested ({dev_str}) but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(dev_str)

    def _load_dataset_info(self) -> Dict[str, Any]:
        """Load dataset metadata from YAML if specified, respecting data_path override."""
        if not self.config.data:
            return {"nc": 4, "names": {i: f"class_{i}" for i in range(4)}, "data_cfg": None}

        import yaml

        yaml_path = Path(self.config.data)
        data_cfg: Dict[str, Any] = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, "r") as f:
                    data_cfg = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[WARNING] Could not parse dataset YAML '{yaml_path}': {e}")
        else:
            try:
                from ultralytics.data.utils import check_det_dataset
                data_cfg = check_det_dataset(self.config.data)
            except Exception as e:
                print(f"[INFO] Ultralytics check_det_dataset for '{self.config.data}' failed: {e}")

        # Override dataset root path if provided
        if self.config.data_path:
            data_cfg["path"] = str(self.config.data_path)

        root_path = Path(data_cfg["path"]) if "path" in data_cfg and data_cfg["path"] else None

        # Resolve validation split path
        if root_path and root_path.exists():
            val_candidates = []
            val_key = data_cfg.get("val")
            if val_key:
                val_candidates.append(root_path / val_key)
                val_candidates.append(Path(val_key))
            val_candidates.extend([
                root_path / "valid" / "images",
                root_path / "val" / "images",
                root_path / "valid",
                root_path / "val",
                root_path / "images" / "valid",
                root_path / "images" / "val",
            ])
            for cand in val_candidates:
                if cand.exists():
                    data_cfg["val"] = str(cand.resolve())
                    break
            data_cfg["path"] = str(root_path.resolve())

        # Extract nc and class names
        nc = data_cfg.get("nc", 4)
        raw_names = data_cfg.get("names", {i: f"class_{i}" for i in range(nc)})
        if isinstance(raw_names, list):
            names = {i: n for i, n in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            names = {int(k): v for k, v in raw_names.items()}
        else:
            names = {i: f"class_{i}" for i in range(nc)}

        data_cfg["nc"] = nc
        data_cfg["names"] = names

        # Validate that 'val' directory actually exists
        val_resolved = data_cfg.get("val")
        if val_resolved and Path(val_resolved).exists():
            data_cfg["val"] = str(Path(val_resolved).resolve())
            return {"nc": nc, "names": names, "data_cfg": data_cfg}

        # If val still doesn't exist, log warning and set data_cfg to None to avoid crashing
        print(f"[INFO] Validation dataset path could not be resolved from '{self.config.data}'. Accuracy evaluation will be skipped.")
        return {"nc": nc, "names": names, "data_cfg": None}

    def get_system_metadata(self) -> Dict[str, Any]:
        """Collect target hardware and environment metadata."""
        meta: Dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            dev_idx = self.device.index if self.device.index is not None else 0
            meta["gpu_name"] = torch.cuda.get_device_name(dev_idx)
            meta["cuda_version"] = torch.version.cuda
            meta["vram_total_gb"] = round(torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3), 2)

        # Detect Jetson Orin / Tegra
        tegra_file = Path("/etc/nv_tegra_release")
        if tegra_file.exists():
            try:
                meta["jetson_release"] = tegra_file.read_text().strip()
            except Exception:
                pass

        try:
            import ultralytics
            meta["ultralytics"] = ultralytics.__version__
        except Exception:
            pass

        try:
            import tensorrt
            meta["tensorrt"] = tensorrt.__version__
        except Exception:
            pass

        try:
            import onnxruntime
            meta["onnxruntime"] = onnxruntime.__version__
        except Exception:
            pass

        return meta

    def run_suite(self) -> List[ModelBenchmarkResult]:
        """Execute benchmarking over all configured candidate models."""
        results: List[ModelBenchmarkResult] = []
        models = self.config.models

        if not models:
            print("[WARNING] No models specified for benchmark suite.")
            return results

        print(f"\n{'='*80}")
        print(f"  UNINA-DLA BENCHMARK SUITE: Starting evaluation of {len(models)} model(s)")
        print(f"  Device: {self.device} | Resolution: {self.config.imgsz[0]}x{self.config.imgsz[1]} | Batch: {self.config.batch_size}")
        print(f"  Warmup: {self.config.warmup} iters | Timed: {self.config.iterations} iters")
        print(f"{'='*80}\n")

        for idx, spec in enumerate(models, 1):
            print(f"[{idx}/{len(models)}] Benchmarking model: {spec.name} ({spec.path})...")
            res = self.benchmark_model(spec)
            results.append(res)
            print(f"      Backend: {res.backend} | Inference: {res.inference.mean_ms:.2f} ms | E2E FPS: {res.e2e.fps:.1f}")
            if res.accuracy_evaluated:
                print(f"      Accuracy: mAP50={res.map50:.4f}, mAP50-95={res.map50_95:.4f}")
            print()

        return results

    def benchmark_model(self, spec: ModelSpec) -> ModelBenchmarkResult:
        """Benchmark a single candidate model."""
        h, w = self.config.imgsz
        bs = self.config.batch_size

        result = ModelBenchmarkResult(
            name=spec.name,
            path=spec.path,
            backend="Unknown",
            device=str(self.device),
            batch_size=bs,
            resolution=(h, w),
        )

        try:
            # 1. Instantiate Adapter
            adapter: BaseModelAdapter = load_model_adapter(
                spec=spec,
                device=self.device,
                num_classes=self.dataset_info["nc"],
                names=self.dataset_info["names"],
                imgsz=(h, w),
            )
            result.backend = adapter.backend
            result.disk_size_mb = round(adapter.get_file_size_mb(), 2)
            result.weight_size_mb = round(adapter.get_weight_size_mb(), 2)

            # 2. Parameters & Complexity
            num_params = adapter.get_num_params()
            if num_params is not None:
                result.params_m = round(num_params / 1e6, 2)
            gflops = adapter.get_gflops((h, w))
            if gflops is not None:
                result.gflops = round(gflops, 1)

            # Ensure model parameters reside on target device
            adapter.to_device(self.device)

            # 3. Dummy Batch for Speed Profiling (respect native engine resolution)
            model_h, model_w = getattr(adapter, "imgsz", (h, w))
            result.resolution = (model_h, model_w)
            dummy_batch_cpu = torch.randint(0, 256, (bs, 3, model_h, model_w), dtype=torch.uint8)
            dummy_batch_dev = dummy_batch_cpu.to(self.device).float() / 255.0

            # 4. Warm-up
            if self.config.verbose:
                print(f"      Warming up ({self.config.warmup} iterations)...")
            adapter.warmup(dummy_batch_dev, iterations=self.config.warmup)

            # 5. Measure Latency & Throughput
            if self.config.verbose:
                print(f"      Profiling latency ({self.config.iterations} iterations)...")
            adapter.to_device(self.device)
            pre_times, inf_times, post_times, e2e_times = self._profile_speed(
                adapter, dummy_batch_cpu, iterations=self.config.iterations
            )

            result.preprocess = LatencyStats.compute(pre_times, batch_size=bs)
            result.inference = LatencyStats.compute(inf_times, batch_size=bs)
            result.postprocess = LatencyStats.compute(post_times, batch_size=bs)
            result.e2e = LatencyStats.compute(e2e_times, batch_size=bs)

            # 6. Peak Memory Profiling
            if self.device.type == "cuda":
                result.peak_vram_mb, result.peak_reserved_mb = self._profile_memory(
                    adapter, dummy_batch_dev
                )

            # 7. Accuracy / Intelligence Evaluation (if dataset available)
            if self.dataset_info["data_cfg"] and not self.config.speed_only:
                val_stats = self._evaluate_accuracy(adapter)
                if val_stats:
                    result.accuracy_evaluated = True
                    result.map50 = round(float(val_stats.get("mAP50", 0.0)), 4)
                    result.map50_95 = round(float(val_stats.get("mAP50-95", 0.0)), 4)
                    result.precision = round(float(val_stats.get("precision", 0.0)), 4)
                    result.recall = round(float(val_stats.get("recall", 0.0)), 4)
                    result.f1 = round(float(val_stats.get("f1", 0.0)), 4)
                    result.class_metrics = val_stats.get("ap_class", None)

            # Cleanup
            del adapter
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            result.error = str(e)
            print(f"      [ERROR] Benchmark failed for '{spec.name}': {e}")

        return result

    def _profile_speed(
        self,
        adapter: BaseModelAdapter,
        dummy_batch_cpu: torch.Tensor,
        iterations: int,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Profile preprocess, inference, postprocess, and e2e latencies with high precision."""
        h, w = self.config.imgsz
        is_cuda = (self.device.type == "cuda")

        pre_samples: List[float] = []
        inf_samples: List[float] = []
        post_samples: List[float] = []
        e2e_samples: List[float] = []

        # Pre-create CUDA events for zero overhead
        if is_cuda:
            t_pre_start = torch.cuda.Event(enable_timing=True)
            t_pre_end = torch.cuda.Event(enable_timing=True)
            t_inf_start = torch.cuda.Event(enable_timing=True)
            t_inf_end = torch.cuda.Event(enable_timing=True)
            t_post_start = torch.cuda.Event(enable_timing=True)
            t_post_end = torch.cuda.Event(enable_timing=True)

        for _ in range(iterations):
            # E2E Host Start
            t0 = time.perf_counter()

            # Preprocessing: Host to Device + Float normalizer
            if is_cuda:
                t_pre_start.record()
            x = dummy_batch_cpu.to(self.device, non_blocking=True).float() / 255.0
            if is_cuda:
                t_pre_end.record()
                t_pre_end.synchronize()
                pre_ms = t_pre_start.elapsed_time(t_pre_end)
            else:
                pre_ms = (time.perf_counter() - t0) * 1000.0

            # Raw Inference
            if is_cuda:
                t_inf_start.record()
            with torch.no_grad():
                preds = adapter.forward_raw(x)
            if is_cuda:
                t_inf_end.record()
                t_inf_end.synchronize()
                inf_ms = t_inf_start.elapsed_time(t_inf_end)
            else:
                t_inf_now = time.perf_counter()
                inf_ms = (t_inf_now - t0 - pre_ms / 1000.0) * 1000.0

            # Postprocessing & Decoding
            if is_cuda:
                t_post_start.record()
            with torch.no_grad():
                _ = adapter.postprocess(
                    preds,
                    image_shape=(h, w),
                    conf_thres=self.config.conf_thres,
                    iou_thres=self.config.iou_thres,
                    nms=adapter.spec.nms or self.config.nms,
                    max_det=self.config.max_det,
                )
            if is_cuda:
                t_post_end.record()
                t_post_end.synchronize()
                post_ms = t_post_start.elapsed_time(t_post_end)
            else:
                post_ms = (time.perf_counter() - t0 - (pre_ms + inf_ms) / 1000.0) * 1000.0

            e2e_ms = (time.perf_counter() - t0) * 1000.0

            pre_samples.append(pre_ms)
            inf_samples.append(inf_ms)
            post_samples.append(post_ms)
            e2e_samples.append(e2e_ms)

        return pre_samples, inf_samples, post_samples, e2e_samples

    def _profile_memory(
        self, adapter: BaseModelAdapter, dummy_batch_dev: torch.Tensor
    ) -> Tuple[float, float]:
        """Profile peak GPU VRAM allocation and reservation."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for _ in range(5):
                preds = adapter.forward_raw(dummy_batch_dev)
                _ = adapter.postprocess(
                    preds,
                    image_shape=(dummy_batch_dev.shape[2], dummy_batch_dev.shape[3]),
                    conf_thres=self.config.conf_thres,
                    nms=adapter.spec.nms or self.config.nms,
                )

        torch.cuda.synchronize(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        peak_res = torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0)
        return round(peak_alloc, 1), round(peak_res, 1)

    def _evaluate_accuracy(self, adapter: BaseModelAdapter) -> Optional[Dict[str, Any]]:
        """Run validation dataloader evaluation to measure intelligence/mAP."""
        data_cfg = self.dataset_info["data_cfg"]
        if not data_cfg or "val" not in data_cfg:
            return None

        adapter.to_device(self.device)
        print(f"      Evaluating intelligence (mAP) on {self.config.data}...")
        try:
            from ultralytics.data import build_dataloader, build_yolo_dataset
            from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

            dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
            for k, v in data_cfg.items():
                if k in dataset_cfg.__dict__:
                    setattr(dataset_cfg, k, v)

            target_h, target_w = getattr(adapter, "imgsz", self.config.imgsz)
            # Ultralytics BaseDataset requires dataset_cfg.imgsz to be an integer (max dimension)
            dataset_cfg.imgsz = int(max(target_h, target_w))
            is_rect = bool(target_h != target_w)

            val_set = build_yolo_dataset(
                dataset_cfg,
                data_cfg["val"],
                self.config.batch_size,
                data_cfg,
                mode="val",
                rect=is_rect,
                stride=32,
            )
            val_loader = build_dataloader(
                val_set,
                batch=self.config.batch_size,
                workers=self.config.num_workers,
                shuffle=False,
            )

            stats = adapter.eval_validation_loader(
                dataloader=val_loader,
                conf_thres=getattr(self.config, "conf_thres", 0.001) or 0.001,
                iou_thres=self.config.iou_thres,
                nms=adapter.spec.nms or self.config.nms,
                max_det=self.config.max_det,
            )
            return stats
        except Exception as e:
            print(f"      [WARNING] Accuracy evaluation failed: {e}")
            return None
