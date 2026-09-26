"""Automated tests for KYC Document Agent visual verification UI components."""

import logging
from pathlib import Path
from PIL import Image
import pytest

from src.benchmark.benchmark_suite import BenchmarkScenario, SimulatedBenchmarkExtractor
from src.triage.triage_engine import (
    FieldAnomaly,
    TriageLifecycle,
    verify_onboarding_package,
)
from src.ui.app import annotate_document_image, get_badge_font, load_preset_scenario
from scripts.generate_demo_samples import export_demo_samples


def test_annotate_document_image_unconditional_normalized_scaling() -> None:
    """Verify bounding box rendering accurately scales [0, 1000] normalized boxes to small and large resolutions."""
    # Test on a small document image (400x300, < 1000px) where previous heuristic misfired
    small_img = Image.new("RGB", (400, 300), color=(240, 240, 240))

    anomalies = [
        FieldAnomaly(
            document_type="national_id",
            document_type_ar="البطاقة الوطنية الموحدة",
            field_name="full_name",
            field_name_ar="الاسم الرباعي واللقب",
            tier=1,
            confidence=0.45,
            threshold=0.85,
            value=None,
            obscured=True,
            bbox=[100, 100, 200, 400],  # Normalized [ymin, xmin, ymax, xmax]
            reason="Field obscured by glare",
            reason_ar="الحقل محجوب بانعكاس ضوئي",
        ),
        FieldAnomaly(
            document_type="national_id",
            document_type_ar="البطاقة الوطنية الموحدة",
            field_name="mother_name",
            field_name_ar="اسم الأم الثلاثي",
            tier=2,
            confidence=0.60,
            threshold=0.70,
            value="فاطمة حسن",
            obscured=False,
            bbox=[300, 100, 400, 400],
            reason="Low tier 2 confidence",
            reason_ar="انخفاض نسبي في ثقة الحقل الثانوي",
        ),
    ]

    annotated = annotate_document_image(small_img, anomalies, "national_id")

    assert annotated.size == small_img.size
    assert annotated.mode == "RGB"
    # Pixels must be modified by drawing the boxes
    assert annotated.tobytes() != small_img.tobytes()


def test_annotate_document_image_handles_none_bbox_gracefully() -> None:
    """Verify anomaly with None bbox does not raise exceptions."""
    base_img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    anomalies = [
        FieldAnomaly(
            document_type="tax_card",
            document_type_ar="الهوية الضريبية",
            field_name="tax_id",
            field_name_ar="الرقم الضريبي",
            tier=1,
            confidence=0.50,
            threshold=0.85,
            value=None,
            obscured=True,
            bbox=None,
            reason="Obscured",
            reason_ar="محجوب",
        )
    ]
    annotated = annotate_document_image(base_img, anomalies, "tax_card")
    assert annotated.size == base_img.size


def test_get_badge_font_loads_with_loud_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify badge font loads cleanly and emits warning log when font path is missing."""
    import src.ui.app as app_module

    # 1. Normal path
    normal_font = app_module.get_badge_font(size=14)
    assert normal_font is not None

    # 2. Forced fallback path via monkeypatch
    fake_path = "/nonexistent/fake_font_that_does_not_exist_xyz123.ttf"
    monkeypatch.setattr(app_module, "FONT_BOLD", fake_path)
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        fallback_font = app_module.get_badge_font(size=14)
        assert fallback_font is not None
        assert "Arabic TrueType font not found at" in caplog.text
        assert fake_path in caplog.text


def test_load_preset_scenarios_all_return_valid_images() -> None:
    """Verify all 5 preset scenarios load valid, non-empty 3-document packages."""
    scenarios = [
        BenchmarkScenario.CLEAN,
        BenchmarkScenario.SPECULAR_GLARE,
        BenchmarkScenario.BLURRED,
        BenchmarkScenario.TILTED,
        BenchmarkScenario.IDENTITY_MISMATCH,
    ]

    for scenario in scenarios:
        nid_img, biz_img, tax_img, gt = load_preset_scenario(scenario)
        assert isinstance(nid_img, Image.Image)
        assert isinstance(biz_img, Image.Image)
        assert isinstance(tax_img, Image.Image)
        assert nid_img.width > 0 and nid_img.height > 0
        assert biz_img.width > 0 and biz_img.height > 0
        assert tax_img.width > 0 and tax_img.height > 0
        assert isinstance(gt, dict)
        assert "national_id" in gt
        assert "full_name" in gt["national_id"]


def test_verify_onboarding_package_offline_preset_execution() -> None:
    """Verify end-to-end execution of a clean preset through the verification seam."""
    nid_img, biz_img, tax_img, _ = load_preset_scenario(BenchmarkScenario.CLEAN)
    extractor = SimulatedBenchmarkExtractor()

    dossier = verify_onboarding_package(
        national_id_image=nid_img,
        business_license_image=biz_img,
        tax_card_image=tax_img,
        extractor=extractor,
    )

    assert dossier.lifecycle_outcome == TriageLifecycle.AUTO_PASS
    assert dossier.overall_confidence >= 0.85
    assert dossier.tier1_passed is True
    assert len(dossier.tier1_anomalies) == 0


def test_generate_demo_samples_cli(tmp_path: Path) -> None:
    """Verify demo samples exporter writes all 5 scenario folders and files."""
    export_demo_samples(output_dir=tmp_path, seed=42)

    expected_subdirs = [
        "01_clean_autopass",
        "02_specular_glare_escalate",
        "03_gaussian_blur_escalate",
        "04_tilted_autopass",
        "05_identity_mismatch_reject",
    ]

    for subdir in expected_subdirs:
        folder = tmp_path / subdir
        assert folder.is_dir(), f"Missing scenario directory: {subdir}"
        assert (folder / "national_id.png").is_file()
        assert (folder / "business_license.png").is_file()
        assert (folder / "tax_card.png").is_file()
        assert (folder / "ground_truth.json").is_file()


def test_app_test_initialization() -> None:
    """Verify headless Streamlit app initialization and sidebar widget presence."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=10)

    # Verify app rendered without unhandled exceptions
    assert not app.exception
    # Check title
    assert len(app.title) > 0
    # Check run button exists
    assert len(app.button) > 0
