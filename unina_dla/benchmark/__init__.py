"""
UNINA-DLA Benchmarking Suite.

Provides tools to benchmark object detection models across accuracy (intelligence),
speed (latency/throughput on NVIDIA GPU/DLA), and computational footprint.
"""

from unina_dla.benchmark.config import BenchmarkConfig, ModelSpec, parse_resolution


def run_benchmark(config: BenchmarkConfig):
    """Execute benchmarking suite and generate full reports."""
    from unina_dla.benchmark.reporter import BenchmarkReporter
    from unina_dla.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(config)
    system_meta = runner.get_system_metadata()
    results = runner.run_suite()
    BenchmarkReporter.save_all(results, config, system_meta)
    return results


__all__ = [
    "BenchmarkConfig",
    "ModelSpec",
    "parse_resolution",
    "run_benchmark",
]
