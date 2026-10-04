"""
UNINA-DLA Benchmark Reporter Module.

Generates comprehensive benchmark reports:
1. Formatted Terminal Console Tables.
2. GitHub-Flavored Markdown Tables (ready for README.md).
3. Machine-readable JSON reports (for CI/CD pipelines).
4. CSV reports (for spreadsheets and pandas analysis).
5. Pareto Frontier and Latency Breakdown comparison plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from unina_dla.benchmark.config import BenchmarkConfig
from unina_dla.benchmark.runner import ModelBenchmarkResult


class BenchmarkReporter:
    """Formats and exports benchmark suite findings."""

    @staticmethod
    def print_terminal_table(
        results: List[ModelBenchmarkResult],
        config: BenchmarkConfig,
        system_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print an aligned ASCII summary table to stdout."""
        lines = []
        lines.append("=" * 115)
        lines.append("                                        UNINA-DLA BENCHMARK SUITE")
        lines.append("=" * 115)

        # Environment info
        env_strs = [
            f"Device: {config.device}",
            f"Resolution: {config.imgsz[0]}x{config.imgsz[1]}",
            f"Batch: {config.batch_size}",
            f"Warmup: {config.warmup}",
            f"Iterations: {config.iterations}",
        ]
        if system_meta and "gpu_name" in system_meta:
            env_strs.insert(0, f"Hardware: {system_meta['gpu_name']}")
        lines.append(" | ".join(env_strs))
        lines.append("-" * 115)

        # Header
        headers = [
            "Model Name",
            "Backend",
            "Params(M)",
            "GFLOPs",
            "Size(MB)",
            "mAP50",
            "mAP50-95",
            "Infer(ms)",
            "Post(ms)",
            "E2E(ms)",
            "P95(ms)",
            "FPS",
        ]
        col_widths = [20, 24, 10, 8, 9, 8, 9, 10, 9, 9, 9, 8]

        header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * 115)

        for r in results:
            if r.error:
                err_line = f"{r.name:<20} | ERROR: {r.error[:85]}"
                lines.append(err_line)
                continue

            size_val = r.weight_size_mb if r.weight_size_mb is not None else r.disk_size_mb
            row = [
                r.name[:19],
                r.backend[:23],
                f"{r.params_m:.1f}M" if r.params_m is not None else "-",
                f"{r.gflops:.1f}" if r.gflops is not None else "-",
                f"{size_val:.1f}",
                f"{r.map50:.3f}" if r.map50 is not None else "-",
                f"{r.map50_95:.3f}" if r.map50_95 is not None else "-",
                f"{r.inference.mean_ms:.2f}",
                f"{r.postprocess.mean_ms:.2f}",
                f"{r.e2e.mean_ms:.2f}",
                f"{r.e2e.p95_ms:.2f}",
                f"{r.e2e.fps:.1f}",
            ]
            row_line = " | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
            lines.append(row_line)

        lines.append("=" * 115)
        lines.append("* Size(MB): pure model weights footprint (excluding optimizer/checkpoint states). GFLOPs: GMACs.")
        print("\n" + "\n".join(lines) + "\n")

    @staticmethod
    def to_markdown(
        results: List[ModelBenchmarkResult],
        config: BenchmarkConfig,
        system_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format benchmark findings into a GitHub Markdown document."""
        md = []
        md.append("# Benchmark Report: Object Detection Pipeline\n")
        md.append(f"> Auto-generated benchmark report across **{len(results)}** candidate models.\n")

        # System info block
        md.append("### Environment Specifications\n")
        md.append(f"- **Target Device**: `{config.device}`")
        if system_meta and "gpu_name" in system_meta:
            md.append(f"- **Silicon / GPU**: `{system_meta['gpu_name']}`")
        if system_meta and "jetson_release" in system_meta:
            md.append(f"- **Jetson JetPack**: `{system_meta['jetson_release']}`")
        md.append(f"- **Input Resolution**: `{config.imgsz[0]}x{config.imgsz[1]}` (Batch Size: `{config.batch_size}`)")
        md.append(f"- **Iterations**: `{config.iterations}` timed iterations (after `{config.warmup}` warm-up runs)")
        if config.data:
            md.append(f"- **Validation Dataset**: `{config.data}`")
        md.append("\n---\n")

        # Main Table
        md.append("### Benchmark Comparison Matrix\n")
        headers = [
            "Model",
            "Backend",
            "Params",
            "GFLOPs",
            "Size",
            "mAP@50",
            "mAP@50-95",
            "Inference (ms)",
            "Postprocess (ms)",
            "E2E Latency (ms)",
            "P95 (ms)",
            "FPS",
            "Peak VRAM",
        ]
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in results:
            if r.error:
                md.append(f"| **{r.name}** | *Error* | - | - | - | - | - | - | - | - | - | - | {r.error} |")
                continue

            p_str = f"{r.params_m:.1f}M" if r.params_m is not None else "-"
            g_str = f"{r.gflops:.1f}" if r.gflops is not None else "-"
            size_val = r.weight_size_mb if r.weight_size_mb is not None else r.disk_size_mb
            s_str = f"{size_val:.1f} MB"
            m50_str = f"{r.map50:.4f}" if r.map50 is not None else "-"
            m5095_str = f"{r.map50_95:.4f}" if r.map50_95 is not None else "-"
            inf_str = f"{r.inference.mean_ms:.2f}"
            post_str = f"{r.postprocess.mean_ms:.2f}"
            e2e_str = f"{r.e2e.mean_ms:.2f}"
            p95_str = f"{r.e2e.p95_ms:.2f}"
            fps_str = f"**{r.e2e.fps:.1f}**"
            vram_str = f"{r.peak_vram_mb:.1f} MB" if r.peak_vram_mb is not None else "-"

            row = [
                f"**{r.name}**",
                r.backend,
                p_str,
                g_str,
                s_str,
                m50_str,
                m5095_str,
                inf_str,
                post_str,
                e2e_str,
                p95_str,
                fps_str,
                vram_str,
            ]
            md.append("| " + " | ".join(row) + " |")

        md.append("\n*Note: **Size** reflects pure model weights memory footprint (excluding optimizer/EMA training states). **GFLOPs** reflects Multiply-Accumulate operations (GMACs) aligned with Ultralytics conventions.*\n")
        md.append("---\n")

        # Latency breakdown section
        md.append("### Latency & Jitter Breakdown (ms)\n")
        md.append("| Model | Preprocess (Mean) | Inference (Mean) | Postprocess (Mean) | E2E Min | E2E Median | E2E P95 | E2E P99 | E2E Max |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        for r in results:
            if r.error:
                continue
            md.append(
                f"| {r.name} | {r.preprocess.mean_ms:.2f} | {r.inference.mean_ms:.2f} | "
                f"{r.postprocess.mean_ms:.2f} | {r.e2e.min_ms:.2f} | {r.e2e.median_ms:.2f} | "
                f"{r.e2e.p95_ms:.2f} | {r.e2e.p99_ms:.2f} | {r.e2e.max_ms:.2f} |"
            )

        md.append("\n")
        return "\n".join(md)

    @staticmethod
    def to_json(
        results: List[ModelBenchmarkResult],
        config: BenchmarkConfig,
        system_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert benchmark data to serialized dictionary."""
        models_data = []
        for r in results:
            m_dict: Dict[str, Any] = {
                "name": r.name,
                "path": r.path,
                "backend": r.backend,
                "device": r.device,
                "batch_size": r.batch_size,
                "resolution": list(r.resolution),
                "params_m": r.params_m,
                "gflops": r.gflops,
                "weight_size_mb": r.weight_size_mb,
                "disk_size_mb": r.disk_size_mb,
                "peak_vram_mb": r.peak_vram_mb,
                "peak_reserved_mb": r.peak_reserved_mb,
                "error": r.error,
            }
            if not r.error:
                m_dict["latency_ms"] = {
                    "preprocess": {
                        "mean": r.preprocess.mean_ms,
                        "median": r.preprocess.median_ms,
                        "p95": r.preprocess.p95_ms,
                    },
                    "inference": {
                        "mean": r.inference.mean_ms,
                        "median": r.inference.median_ms,
                        "p95": r.inference.p95_ms,
                    },
                    "postprocess": {
                        "mean": r.postprocess.mean_ms,
                        "median": r.postprocess.median_ms,
                        "p95": r.postprocess.p95_ms,
                    },
                    "end_to_end": {
                        "mean": r.e2e.mean_ms,
                        "median": r.e2e.median_ms,
                        "min": r.e2e.min_ms,
                        "max": r.e2e.max_ms,
                        "std": r.e2e.std_ms,
                        "p90": r.e2e.p90_ms,
                        "p95": r.e2e.p95_ms,
                        "p99": r.e2e.p99_ms,
                        "fps": r.e2e.fps,
                    },
                }
                m_dict["accuracy"] = {
                    "evaluated": r.accuracy_evaluated,
                    "mAP50": r.map50,
                    "mAP50_95": r.map50_95,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "class_metrics": r.class_metrics,
                }
            models_data.append(m_dict)

        return {
            "system_metadata": system_meta or {},
            "config": {
                "imgsz": list(config.imgsz),
                "batch_size": config.batch_size,
                "device": config.device,
                "warmup": config.warmup,
                "iterations": config.iterations,
                "dataset": config.data,
                "nms": config.nms,
            },
            "models": models_data,
        }

    @staticmethod
    def to_csv(results: List[ModelBenchmarkResult]) -> str:
        """Export results as comma-separated values string."""
        import io

        output = io.StringIO()
        fieldnames = [
            "name",
            "backend",
            "params_m",
            "gflops",
            "weight_size_mb",
            "disk_size_mb",
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "f1",
            "infer_mean_ms",
            "post_mean_ms",
            "e2e_mean_ms",
            "e2e_p95_ms",
            "fps",
            "peak_vram_mb",
            "error",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "name": r.name,
                "backend": r.backend,
                "params_m": r.params_m if r.params_m is not None else "",
                "gflops": r.gflops if r.gflops is not None else "",
                "weight_size_mb": r.weight_size_mb if r.weight_size_mb is not None else "",
                "disk_size_mb": r.disk_size_mb,
                "mAP50": r.map50 if r.map50 is not None else "",
                "mAP50_95": r.map50_95 if r.map50_95 is not None else "",
                "precision": r.precision if r.precision is not None else "",
                "recall": r.recall if r.recall is not None else "",
                "f1": r.f1 if r.f1 is not None else "",
                "infer_mean_ms": f"{r.inference.mean_ms:.2f}" if not r.error else "",
                "post_mean_ms": f"{r.postprocess.mean_ms:.2f}" if not r.error else "",
                "e2e_mean_ms": f"{r.e2e.mean_ms:.2f}" if not r.error else "",
                "e2e_p95_ms": f"{r.e2e.p95_ms:.2f}" if not r.error else "",
                "fps": f"{r.e2e.fps:.1f}" if not r.error else "",
                "peak_vram_mb": r.peak_vram_mb if r.peak_vram_mb is not None else "",
                "error": r.error or "",
            })

        return output.getvalue()

    @staticmethod
    def plot_results(
        results: List[ModelBenchmarkResult],
        output_dir: Path,
    ) -> List[Path]:
        """Generate Pareto curve and Latency breakdown comparison plots."""
        saved_plots: List[Path] = []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[INFO] matplotlib not installed; skipping plot generation.")
            return saved_plots

        valid_results = [r for r in results if not r.error]
        if not valid_results:
            return saved_plots

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Latency Breakdown Stacked Bar Chart
        try:
            plt.figure(figsize=(10, max(5, len(valid_results) * 0.8)))
            names = [r.name for r in valid_results]
            pre = [r.preprocess.mean_ms for r in valid_results]
            inf = [r.inference.mean_ms for r in valid_results]
            post = [r.postprocess.mean_ms for r in valid_results]

            y_pos = range(len(names))
            plt.barh(y_pos, pre, label="Preprocess", color="#3498db")
            plt.barh(y_pos, inf, left=pre, label="Inference", color="#e74c3c")
            left_post = [p + i for p, i in zip(pre, inf)]
            plt.barh(y_pos, post, left=left_post, label="Postprocess/Decode", color="#2ecc71")

            plt.yticks(y_pos, names)
            plt.xlabel("Latency (ms)")
            plt.title("Pipeline Latency Breakdown (Preprocess vs Inference vs Postprocess)")
            plt.legend(loc="lower right")
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()

            breakdown_path = output_dir / "latency_breakdown.png"
            plt.savefig(breakdown_path, dpi=200)
            plt.close()
            saved_plots.append(breakdown_path)
        except Exception as e:
            print(f"[WARNING] Latency breakdown plot failed: {e}")

        # 2. Pareto Frontier (mAP vs FPS / Latency) if accuracy evaluated
        acc_results = [r for r in valid_results if r.accuracy_evaluated and r.map50_95 is not None]
        if acc_results:
            try:
                plt.figure(figsize=(9, 6))
                for r in acc_results:
                    plt.scatter(
                        r.e2e.fps,
                        r.map50_95,
                        s=150,
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=1.5,
                        label=r.name,
                    )
                    plt.annotate(
                        r.name,
                        (r.e2e.fps, r.map50_95),
                        textcoords="offset points",
                        xytext=(8, 5),
                        fontsize=9,
                        fontweight="bold",
                    )

                plt.xlabel("Throughput (Frames Per Second)")
                plt.ylabel("Accuracy (mAP@0.50:0.95)")
                plt.title("Pareto Efficiency: Accuracy vs Throughput (Higher & Further Right is Better)")
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()

                pareto_path = output_dir / "pareto_frontier.png"
                plt.savefig(pareto_path, dpi=200)
                plt.close()
                saved_plots.append(pareto_path)
            except Exception as e:
                print(f"[WARNING] Pareto curve plot failed: {e}")

        return saved_plots

    @classmethod
    def save_all(
        cls,
        results: List[ModelBenchmarkResult],
        config: BenchmarkConfig,
        system_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Save terminal output, markdown, JSON, CSV, and plots to output_dir."""
        out_dir = config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_files: Dict[str, Path] = {}

        # 1. Print terminal table
        cls.print_terminal_table(results, config, system_meta)

        # 2. Markdown
        if config.save_md:
            md_path = out_dir / "benchmark_summary.md"
            md_content = cls.to_markdown(results, config, system_meta)
            md_path.write_text(md_content, encoding="utf-8")
            saved_files["markdown"] = md_path
            print(f"[Benchmark] Markdown summary saved to: {md_path}")

        # 3. JSON
        if config.save_json:
            json_path = out_dir / "benchmark_results.json"
            json_data = cls.to_json(results, config, system_meta)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
            saved_files["json"] = json_path
            print(f"[Benchmark] JSON report saved to: {json_path}")

        # 4. CSV
        if config.save_csv:
            csv_path = out_dir / "benchmark_results.csv"
            csv_content = cls.to_csv(results)
            csv_path.write_text(csv_content, encoding="utf-8")
            saved_files["csv"] = csv_path
            print(f"[Benchmark] CSV report saved to: {csv_path}")

        # 5. Plots
        if config.plot:
            plots = cls.plot_results(results, out_dir)
            for p in plots:
                saved_files[p.stem] = p
                print(f"[Benchmark] Plot saved to: {p}")

        return saved_files
