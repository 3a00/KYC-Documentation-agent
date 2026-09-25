"""Tests for OpenCV quality diagnostics and confidence calibration."""

import numpy as np
from PIL import Image
import pytest

from src.data.synthetic_generator import generate_synthetic_document
from src.vision.diagnostics import (
    QualityMetrics,
    assess_field_qualities,
    assess_quality,
    compute_blur_variance,
    compute_calibrated_confidence,
    compute_glare_ratio,
    crop_bounding_box,
    to_cv2_bgr,
)


def test_to_cv2_bgr_pil_and_numpy():
    """Verify PIL and numpy array conversions to BGR format."""
    pil_img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    bgr = to_cv2_bgr(pil_img)
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == (50, 50, 3)
    # Red in RGB is Blue in BGR: B=0, G=0, R=255
    assert (bgr[0, 0] == [0, 0, 255]).all()

    # Pass-through for numpy array
    assert to_cv2_bgr(bgr) is bgr

    with pytest.raises(TypeError):
        to_cv2_bgr("not_an_image")  # type: ignore[arg-type]


def test_crop_bounding_box():
    """Verify bounding box cropping with clamping behavior."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop_bounding_box(img, (10, 20, 30, 40))
    assert cropped.shape == (20, 20, 3)

    # None bbox returns original
    assert crop_bounding_box(img, None).shape == (100, 100, 3)

    # Clamping out-of-bounds coordinates
    clamped = crop_bounding_box(img, (-10, -10, 150, 150))
    assert clamped.shape == (100, 100, 3)

    # Inverted bbox returns original safely
    inverted = crop_bounding_box(img, (50, 50, 20, 20))
    assert inverted.shape == (100, 100, 3)


def test_compute_blur_variance_clean_vs_blur():
    """Blur variance must be significantly lower on blurred document images."""
    clean_doc = generate_synthetic_document("unified_national_card", seed=42)
    blur_doc = generate_synthetic_document("unified_national_card", defect_types=["blur"], seed=42)

    var_clean = compute_blur_variance(clean_doc.clean_image)
    var_blur = compute_blur_variance(blur_doc.degraded_image)

    assert var_clean > 100.0
    assert var_blur < var_clean
    assert var_blur < 100.0


def test_compute_glare_ratio_clean_vs_glare():
    """Glare ratio must be ~0 on clean documents and high on glarified regions."""
    clean_doc = generate_synthetic_document("tax_card", seed=42)
    tax_id_bbox = clean_doc.fields["tax_id"].bbox

    clean_glare = compute_glare_ratio(clean_doc.clean_image, bbox=tax_id_bbox)
    assert clean_glare == 0.0

    glare_doc = generate_synthetic_document(
        "tax_card", defect_types=["glare"], target_field="tax_id", seed=42
    )
    degraded_glare = compute_glare_ratio(glare_doc.degraded_image, bbox=tax_id_bbox)
    assert degraded_glare > 0.50


def test_assess_quality_structured_output():
    """assess_quality outputs structured metrics without raising exceptions."""
    clean_doc = generate_synthetic_document("business_license", seed=42)
    metrics = assess_quality(clean_doc.clean_image)

    assert isinstance(metrics, QualityMetrics)
    assert not metrics.is_blurry
    assert not metrics.has_glare
    assert metrics.clarity_factor >= 0.90


def test_assess_field_qualities():
    """assess_field_qualities maps each field bounding box to its QualityMetrics."""
    clean_doc = generate_synthetic_document("unified_national_card", seed=42)
    bboxes = {name: field_obj.bbox for name, field_obj in clean_doc.fields.items()}

    field_metrics = assess_field_qualities(clean_doc.clean_image, bboxes)
    assert set(field_metrics.keys()) == set(bboxes.keys())
    for metric in field_metrics.values():
        assert isinstance(metric, QualityMetrics)


def test_compute_calibrated_confidence():
    """Confidence calibration incorporates model signal, clarity, and format factor."""
    # Perfect scenario
    conf_clean = compute_calibrated_confidence(0.95, 1.0, format_valid=True)
    assert conf_clean == 0.95

    # Penalized by clarity factor
    conf_degraded = compute_calibrated_confidence(0.95, 0.2, format_valid=True)
    assert round(conf_degraded, 3) == 0.19

    # Format invalid fails closed to 0.0
    conf_invalid = compute_calibrated_confidence(0.95, 1.0, format_valid=False)
    assert conf_invalid == 0.0


def test_multi_document_field_diagnostics_integration():
    """Verify diagnostics across all three document types on clean and degraded samples."""
    test_docs = ["unified_national_card", "business_license", "tax_card"]

    for doc_type in test_docs:
        clean_doc = generate_synthetic_document(doc_type, seed=42)
        blur_doc = generate_synthetic_document(doc_type, defect_types=["blur"], seed=42)

        for field_name, field_obj in clean_doc.fields.items():
            bbox = field_obj.bbox
            clean_metrics = assess_quality(clean_doc.clean_image, bbox=bbox)
            blur_metrics = assess_quality(blur_doc.degraded_image, bbox=bbox)

            assert clean_metrics.blur_variance > 1000.0
            assert not clean_metrics.is_blurry
            assert clean_metrics.glare_ratio == 0.0
            assert not clean_metrics.has_glare
            assert clean_metrics.clarity_factor >= 0.95

            # Local glare defect on target field
            glare_doc = generate_synthetic_document(
                doc_type, defect_types=["glare"], target_field=field_name, seed=42
            )
            glare_metrics = assess_quality(glare_doc.degraded_image, bbox=bbox)
            assert glare_metrics.glare_ratio > 0.70
            assert glare_metrics.has_glare
            assert glare_metrics.clarity_factor < 0.25

