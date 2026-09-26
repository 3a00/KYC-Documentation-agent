"""Tests for empirical benchmark suite.

Verifies:
1. Synthetic test split generation across clean, glare, blur, tilt, and mismatch splits.
2. Grounded field-level accuracy evaluation (numeric exactness and Arabic orthographic normalization).
3. Expected Calibration Error (ECE) partition and mathematical computation.
4. False Auto-Pass Rate measurement confirming < 0.5% safety gating.
5. End-to-end benchmark execution and slide-ready quantitative summary output.
"""

import pytest

from src.benchmark.benchmark_suite import (
    BIZ_FIELD_BASE_CONFIDENCES,
    NID_FIELD_BASE_CONFIDENCES,
    TAX_FIELD_BASE_CONFIDENCES,
    BenchmarkMetrics,
    BenchmarkScenario,
    CalibrationBin,
    SimulatedBenchmarkExtractor,
    compute_ece,
    compute_field_accuracy,
    generate_slide_summary,
    generate_synthetic_benchmark_split,
    run_benchmark,
)
from src.models.schemas import ExtractedField
from src.triage.triage_engine import TriageLifecycle


def test_synthetic_benchmark_split_generation():
    """Verify generator yields balanced packages across all 5 evaluation scenarios."""
    sample_count = 2
    packages = generate_synthetic_benchmark_split(sample_count_per_scenario=sample_count, seed=123)

    assert len(packages) == sample_count * 5
    scenarios_found = {pkg.scenario for pkg in packages}
    assert scenarios_found == {
        BenchmarkScenario.CLEAN,
        BenchmarkScenario.SPECULAR_GLARE,
        BenchmarkScenario.BLURRED,
        BenchmarkScenario.TILTED,
        BenchmarkScenario.IDENTITY_MISMATCH,
    }

    # Verify document images and ground truth are present
    for pkg in packages:
        assert pkg.national_id_image is not None
        assert pkg.business_license_image is not None
        assert pkg.tax_card_image is not None
        assert "national_id" in pkg.ground_truth
        assert "business_license" in pkg.ground_truth
        assert "tax_card" in pkg.ground_truth


def test_field_accuracy_evaluation():
    """Verify numeric exactness and Arabic normalized matching."""
    # 1. Numeric exact match
    field_num = ExtractedField[str](value="199512345678", confidence=0.95, obscured=False, bbox=[0, 0, 10, 10])
    assert compute_field_accuracy("national_id", field_num, "199512345678") is True
    assert compute_field_accuracy("national_id", field_num, "199512345679") is False

    # 2. Arabic text under orthographic normalization (e.g. hamza & spacing)
    field_ar = ExtractedField[str](value="عبدالله احمد", confidence=0.92, obscured=False, bbox=[0, 0, 10, 10])
    assert compute_field_accuracy("full_name", field_ar, "عبد الله أحمد") is True

    # 3. Obscured / null field must evaluate to False
    field_obscured = ExtractedField[str](value=None, confidence=0.50, obscured=True, bbox=[0, 0, 10, 10])
    assert compute_field_accuracy("full_name", field_obscured, "محمد جاسم") is False


def test_compute_ece_mathematical_properties():
    """Verify ECE computation against hand-calculated perfectly calibrated and uncalibrated cases."""
    # Perfect calibration: confidence 0.90 with 90% accuracy, 0.50 with 50% accuracy
    confs = [0.90] * 10 + [0.50] * 10
    accs = [1.0] * 9 + [0.0] * 1 + [1.0] * 5 + [0.0] * 5
    ece, bins = compute_ece(confs, accs, num_bins=10)

    # Bin [0.9, 1.0]: 10 samples, avg_conf=0.90, avg_acc=0.90 -> gap = 0.0
    # Bin [0.5, 0.6): 10 samples, avg_conf=0.50, avg_acc=0.50 -> gap = 0.0
    assert ece == pytest.approx(0.0, abs=1e-3)
    assert len(bins) == 10

    # Completely uncalibrated: confidence 1.0 with 0.0 accuracy
    confs_bad = [1.0, 1.0, 1.0, 1.0]
    accs_bad = [0.0, 0.0, 0.0, 0.0]
    ece_bad, bins_bad = compute_ece(confs_bad, accs_bad, num_bins=10)
    assert ece_bad == pytest.approx(1.0, abs=1e-3)


def test_end_to_end_benchmark_execution_and_safety_gating():
    """Verify full benchmark execution, safety gating (FAP < 0.5%), and slide summary generation."""
    # Generate 2 packages per scenario (10 packages total, 30 documents)
    packages = generate_synthetic_benchmark_split(sample_count_per_scenario=2, seed=42)
    extractor = SimulatedBenchmarkExtractor(seed=42)

    metrics: BenchmarkMetrics = run_benchmark(packages, extractor=extractor)

    assert metrics.total_packages == 10
    assert metrics.total_fields_evaluated > 0

    # Safety Gating: False Auto-Pass Rate must be strictly < 0.5% (here 0.0%)
    assert metrics.false_auto_pass_rate < 0.005
    assert metrics.false_auto_pass_count == 0
    assert metrics.non_pass_eligible_count == 6

    # High accuracy on clean/tilted, calibrated degradation on glare/blur
    assert metrics.clean_split_accuracy >= 0.95
    assert metrics.overall_field_accuracy >= 0.70
    assert metrics.expected_calibration_error < 0.20

    # Triage distribution reflects scenario mix
    assert metrics.auto_pass_rate > 0.0
    assert metrics.human_escalation_rate > 0.0
    assert metrics.hard_mismatch_rate > 0.0

    # Verify slide summary generation
    summary = generate_slide_summary(metrics)
    assert "Deterministic Triage & Calibration Stress-Test" in summary
    assert "False Auto-Pass Rate" in summary
    assert "(0/6)" in summary
    assert "PASSED (<0.5%)" in summary
    assert "Tier 1 Critical Field Accuracy" in summary
    assert "Tier 2 Contextual Field Accuracy" in summary
    assert "Tier 1 critical fields" in summary
    assert "Tier 2 contextual fields" in summary
    assert "Expected Calibration Error (ECE)" in summary


def test_simulated_extractor_defect_handling_across_all_documents():
    """Verify simulated extractor properly models defects across all three document types."""
    packages = generate_synthetic_benchmark_split(sample_count_per_scenario=3, seed=42)
    tilted_pkgs = [pkg for pkg in packages if pkg.scenario == BenchmarkScenario.TILTED]
    extractor = SimulatedBenchmarkExtractor(seed=42)

    doc_base_confs = {
        "national_id": NID_FIELD_BASE_CONFIDENCES,
        "business_license": BIZ_FIELD_BASE_CONFIDENCES,
        "tax_card": TAX_FIELD_BASE_CONFIDENCES,
    }

    assert len(tilted_pkgs) == 3
    for pkg in tilted_pkgs:
        target_doc = pkg.planted_defects[0]["target_document"]
        img = getattr(pkg, f"{target_doc}_image")
        schema = getattr(extractor, f"_extract_{target_doc}")(img)
        base_confs = doc_base_confs[target_doc]

        # Tilted fields have confidence degraded by -0.06 with +/-0.02 jitter.
        # By construction, tilt_conf <= base_conf - 0.04 (tested safely at <= base_conf - 0.03).
        for attr in type(schema).model_fields:
            field_conf = getattr(schema, attr).confidence
            assert field_conf <= base_confs[attr] - 0.03

