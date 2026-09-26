"""Benchmark and evaluation module for KYC Document Agent."""

from src.benchmark.benchmark_suite import (
    BenchmarkMetrics,
    BenchmarkPackage,
    BenchmarkScenario,
    CalibrationBin,
    SimulatedBenchmarkExtractor,
    compute_ece,
    compute_field_accuracy,
    generate_slide_summary,
    generate_synthetic_benchmark_split,
    run_benchmark,
)

__all__ = [
    "BenchmarkMetrics",
    "BenchmarkPackage",
    "BenchmarkScenario",
    "CalibrationBin",
    "SimulatedBenchmarkExtractor",
    "compute_ece",
    "compute_field_accuracy",
    "generate_slide_summary",
    "generate_synthetic_benchmark_split",
    "run_benchmark",
]
