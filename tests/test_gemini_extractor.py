"""Comprehensive tests for Gemini multimodal extractor and bounded 1x retry.

Verifies:
1. Parallel isolated execution without token contamination.
2. Verified timeout budget computation via pure function and coordinator assertions.
3. Integration assertion confirming future.result(timeout=...) receives computed budget.
4. Strict synchronization between TIER1_PRIORITY_KEYS and Schema.get_tier1_fields().
5. Downstream contract: has_unresolved_tier1_defects flags remaining defects accurately.
6. Grounded composite confidence computation (Model Signal * Clarity * Format).
7. Deterministic field priority ordering in retry selection.
8. Optical blur routes to unrecoverable bypass (CLAHE cannot fix blur).
9. Strict 1x retry limit (retries never exceed 1 attempt per document).
10. Real post-CLAHE OpenCV clarity used without artificial floors.
11. Grounded format validation distinguishing genuine IDs from hallucinated fragments.
12. Token logprob alignment and extraction with subword tokens.
13. Unrecoverable bypass for structurally invalid field formats.
"""

from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.extraction.gemini_extractor import (
    DocumentExtractionResult,
    DocumentType,
    ExtractionAuditTrail,
    GeminiMultimodalExtractor,
    LogprobStatus,
    RawBusinessLicenseResponse,
    RawFieldPayload,
    RawNationalIDResponse,
    RawRetryPayload,
    RawTaxCardResponse,
    RetryStatus,
    TIER1_PRIORITY_KEYS,
    compute_package_timeout,
    denormalize_bounding_box,
    enhance_crop_clahe,
    make_null_field,
    resolve_model_signal,
    validate_field_format,
)
from src.models.schemas import BusinessLicenseSchema, ExtractedField, NationalIDSchema, TaxCardSchema
from src.vision.diagnostics import QualityMetrics


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create standard 800x500 synthetic test image canvas."""
    return np.full((500, 800, 3), 245, dtype=np.uint8)


def test_tier1_keys_consistency_with_schemas():
    """Verify TIER1_PRIORITY_KEYS strictly matches Schema.get_tier1_fields() across all documents."""
    nid = NationalIDSchema(
        national_id=make_null_field(),
        full_name=make_null_field(),
        mother_name=make_null_field(),
        expiry_date=make_null_field(),
        issue_date=make_null_field(),
        province=make_null_field(),
    )
    biz = BusinessLicenseSchema(
        business_name=make_null_field(),
        full_name=make_null_field(),
        license_number=make_null_field(),
        issue_date=make_null_field(),
    )
    tax = TaxCardSchema(
        tax_id=make_null_field(),
        taxpayer_name=make_null_field(),
        fiscal_year=make_null_field(),
        issue_date=make_null_field(),
    )

    assert set(TIER1_PRIORITY_KEYS[DocumentType.NATIONAL_ID]) == set(nid.get_tier1_fields().keys())
    assert set(TIER1_PRIORITY_KEYS[DocumentType.BUSINESS_LICENSE]) == set(biz.get_tier1_fields().keys())
    assert set(TIER1_PRIORITY_KEYS[DocumentType.TAX_CARD]) == set(tax.get_tier1_fields().keys())


def test_compute_package_timeout():
    """Verify standalone package timeout computation logic and override handling."""
    # Default: 2 * request_timeout_seconds + 10.0
    assert compute_package_timeout(20.0) == 50.0
    assert compute_package_timeout(25.0) == 60.0

    # Explicit override takes precedence
    assert compute_package_timeout(25.0, override=45.0) == 45.0
    assert compute_package_timeout(30.0, override=90.0) == 90.0


def test_extract_onboarding_package_passes_computed_timeout(blank_image):
    """Integration test: Verify extract_onboarding_package_parallel passes compute_package_timeout into future.result()."""
    extractor = GeminiMultimodalExtractor(request_timeout_seconds=20.0)
    expected_timeout = compute_package_timeout(20.0)

    mock_result = DocumentExtractionResult(
        schema=extractor._create_escalated_fallback_schema(DocumentType.NATIONAL_ID),
        audit=ExtractionAuditTrail(document_type=DocumentType.NATIONAL_ID, retry_status=RetryStatus.NOT_NEEDED),
    )

    with patch("src.extraction.gemini_extractor.ThreadPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = mock_result
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        extractor.extract_onboarding_package_parallel(
            national_id_image=blank_image,
            business_license_image=blank_image,
            tax_card_image=blank_image,
            enable_retry=True,
        )

        assert mock_future.result.call_count == 3
        for call_args in mock_future.result.call_args_list:
            assert call_args.kwargs.get("timeout") == expected_timeout


def test_validate_field_format_distinguishes_real_from_hallucinations():
    """Verify format validation accepts realistic IDs while rejecting junk/hallucinated fragments."""
    # National ID: strict 12 digits (spec.md line 71)
    assert validate_field_format("national_id", "199512345678") is True
    assert validate_field_format("national_id", "١٩٩٥١٢٣٤٥٦٧٨") is True
    assert validate_field_format("national_id", "123456789") is False
    assert validate_field_format("national_id", "19951234567A") is False
    assert validate_field_format("national_id", None) is False

    # Business License: >= 5 chars and >= 3 digits
    assert validate_field_format("license_number", "BL-2023-8841") is True
    assert validate_field_format("license_number", "BL-99881") is True
    assert validate_field_format("license_number", "1029384") is True
    assert validate_field_format("license_number", "ABX7") is False
    assert validate_field_format("license_number", "---") is False
    assert validate_field_format("license_number", "12") is False

    # Tax ID: >= 5 chars and >= 4 digits
    assert validate_field_format("tax_id", "TIN-9988112") is True
    assert validate_field_format("tax_id", "TX-776655") is True
    assert validate_field_format("tax_id", "9988112") is True
    assert validate_field_format("tax_id", "TIN-1") is False
    assert validate_field_format("tax_id", "TAX") is False
    assert validate_field_format("tax_id", None) is False


def test_make_null_field_creates_distinct_instances():
    """Verify make_null_field does not return shared mutable singletons."""
    field_instance_one = make_null_field()
    field_instance_two = make_null_field()
    assert field_instance_one is not field_instance_two
    assert field_instance_one.value is None and field_instance_one.confidence == 0.0 and field_instance_one.obscured is True


def test_has_unresolved_tier1_defects_property():
    """Verify downstream contract property correctly flags unrecoverable defects."""
    valid_field = ExtractedField[str](value="199512345678", confidence=0.95, obscured=False)
    schema = NationalIDSchema(
        national_id=valid_field,
        full_name=valid_field,
        mother_name=valid_field,
        expiry_date=valid_field,
        issue_date=valid_field,
        province=valid_field,
    )

    clean_audit = ExtractionAuditTrail(document_type=DocumentType.NATIONAL_ID, retry_status=RetryStatus.NOT_NEEDED)
    res_clean = DocumentExtractionResult(schema=schema, audit=clean_audit)
    assert res_clean.has_unresolved_tier1_defects is False

    escalated_audit = ExtractionAuditTrail(
        document_type=DocumentType.NATIONAL_ID,
        retry_status=RetryStatus.APPLIED,
        retried_field="full_name",
        unrecoverable_fields=["national_id"],
    )
    res_escalated = DocumentExtractionResult(schema=schema, audit=escalated_audit)
    assert res_escalated.has_unresolved_tier1_defects is True


def test_denormalize_bounding_box():
    """Verify conversion from Gemini 0-1000 normalized coordinates to image pixel space."""
    shape = (500, 800)
    norm_bbox = [100, 200, 300, 600]

    pixel_bbox = denormalize_bounding_box(norm_bbox, shape)
    assert pixel_bbox is not None
    xmin, ymin, xmax, ymax = pixel_bbox
    assert xmin == 160
    assert ymin == 50
    assert xmax == 480
    assert ymax == 150


def test_enhance_crop_clahe_contrast():
    """Verify CLAHE enhances contrast without altering image dimensions."""
    low_contrast_patch = np.full((100, 200, 3), 180, dtype=np.uint8)
    low_contrast_patch[40:60, 50:150] = 190

    enhanced = enhance_crop_clahe(low_contrast_patch)
    assert enhanced.shape == low_contrast_patch.shape
    assert enhanced.dtype == np.uint8
    assert float(np.std(enhanced)) >= float(np.std(low_contrast_patch))


def test_resolve_model_signal_fallback():
    """Verify fallback to verbalized confidence when candidate lacks logprobs."""
    candidate_mock = MagicMock()
    candidate_mock.logprobs_result = None

    signal, status = resolve_model_signal(candidate_mock, "national_id", 0.92)
    assert signal == pytest.approx(0.92)
    assert status == LogprobStatus.NOT_RETURNED


def test_resolve_model_signal_extracted():
    """Verify token logprob extraction when subword tokens are present."""
    candidate_mock = MagicMock()
    token_one = MagicMock(token='"national', log_probability=None)
    token_two = MagicMock(token='_id": ', log_probability=None)
    token_three = MagicMock(token='"199512', log_probability=-0.05)
    token_four = MagicMock(token='345678"', log_probability=-0.03)
    token_five = MagicMock(token=',', log_probability=None)

    candidate_mock.logprobs_result.chosen_candidates = [
        token_one, token_two, token_three, token_four, token_five
    ]

    signal, status = resolve_model_signal(candidate_mock, "national_id", 0.80)
    assert status == LogprobStatus.EXTRACTED
    assert signal > 0.90


def test_bounded_1x_retry_on_recoverable_defect(blank_image):
    """Verify CLAHE retry triggers on low Tier 1 confidence with optical glare and recovers."""
    client_mock = MagicMock()

    mock_nid_raw = RawNationalIDResponse(
        national_id=RawFieldPayload(
            value="199512345678",
            model_confidence=0.80,
            obscured=False,
            bbox_normalized=[100, 200, 200, 600],
        ),
        full_name=RawFieldPayload(value="محمد حسن كاظم الزبيدي", model_confidence=0.95, obscured=False, bbox_normalized=[220, 200, 300, 700]),
        mother_name=RawFieldPayload(value="فاطمة علي", model_confidence=0.90, obscured=False, bbox_normalized=[320, 200, 380, 500]),
        expiry_date=RawFieldPayload(value="2030/12/31", model_confidence=0.95, obscured=False, bbox_normalized=[400, 200, 450, 450]),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False, bbox_normalized=[400, 500, 450, 750]),
        province=RawFieldPayload(value="بغداد", model_confidence=0.90, obscured=False, bbox_normalized=[460, 200, 490, 400]),
    )

    mock_retry_payload = RawRetryPayload(
        value="199512345678",
        model_confidence=0.98,
        obscured=False,
    )

    extractor = GeminiMultimodalExtractor(client=client_mock)

    glare_quality = QualityMetrics(blur_variance=200.0, glare_ratio=0.60, is_blurry=False, has_glare=True, clarity_factor=0.60)
    enhanced_quality = QualityMetrics(blur_variance=220.0, glare_ratio=0.05, is_blurry=False, has_glare=False, clarity_factor=0.92)

    with patch.object(extractor, "_call_gemini_structured", return_value=(mock_nid_raw, MagicMock(logprobs_result=None))), \
         patch("src.extraction.gemini_extractor.assess_quality", side_effect=[glare_quality] * 6 + [enhanced_quality]), \
         patch.object(extractor, "_retry_single_field_clahe", return_value=mock_retry_payload):

        result = extractor.extract_document(blank_image, DocumentType.NATIONAL_ID, enable_retry=True)

        assert isinstance(result.schema, NationalIDSchema)
        assert result.audit.retry_status == RetryStatus.APPLIED
        assert result.audit.retried_field == "national_id"
        assert result.audit.pre_retry_confidence < 0.85
        assert result.audit.post_retry_confidence >= 0.85
        assert result.schema.national_id.confidence >= 0.85
        assert result.schema.national_id.value == "199512345678"


def test_deterministic_priority_order_on_multiple_low_confidence(blank_image):
    """Verify that when multiple Tier 1 fields have low confidence, priority order decides retry."""
    client_mock = MagicMock()

    mock_nid_raw = RawNationalIDResponse(
        national_id=RawFieldPayload(value="199512345678", model_confidence=0.75, obscured=False, bbox_normalized=[100, 200, 200, 600]),
        full_name=RawFieldPayload(value="محمد حسن كاظم", model_confidence=0.70, obscured=False, bbox_normalized=[220, 200, 300, 700]),
        mother_name=RawFieldPayload(value="فاطمة علي", model_confidence=0.90, obscured=False, bbox_normalized=[320, 200, 380, 500]),
        expiry_date=RawFieldPayload(value="2030/12/31", model_confidence=0.95, obscured=False, bbox_normalized=[400, 200, 450, 450]),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False, bbox_normalized=[400, 500, 450, 750]),
        province=RawFieldPayload(value="بغداد", model_confidence=0.90, obscured=False, bbox_normalized=[460, 200, 490, 400]),
    )

    mock_retry_payload = RawRetryPayload(value="199512345678", model_confidence=0.98, obscured=False)

    extractor = GeminiMultimodalExtractor(client=client_mock)
    glare_quality = QualityMetrics(blur_variance=200.0, glare_ratio=0.60, is_blurry=False, has_glare=True, clarity_factor=0.60)
    enhanced_quality = QualityMetrics(blur_variance=220.0, glare_ratio=0.05, is_blurry=False, has_glare=False, clarity_factor=0.92)

    with patch.object(extractor, "_call_gemini_structured", return_value=(mock_nid_raw, MagicMock(logprobs_result=None))), \
         patch("src.extraction.gemini_extractor.assess_quality", side_effect=[glare_quality] * 6 + [enhanced_quality]), \
         patch.object(extractor, "_retry_single_field_clahe", return_value=mock_retry_payload) as mock_retry:

        result = extractor.extract_document(blank_image, DocumentType.NATIONAL_ID, enable_retry=True)

        assert result.audit.retried_field == "national_id"
        mock_retry.assert_called_once()


def test_optical_blur_bypasses_retry(blank_image):
    """Verify that pure optical blur routes to unrecoverable bypass because CLAHE cannot fix blur."""
    client_mock = MagicMock()

    mock_nid_raw = RawNationalIDResponse(
        national_id=RawFieldPayload(value="199512345678", model_confidence=0.75, obscured=False, bbox_normalized=[100, 200, 200, 600]),
        full_name=RawFieldPayload(value="محمد حسن كاظم", model_confidence=0.95, obscured=False, bbox_normalized=[220, 200, 300, 700]),
        mother_name=RawFieldPayload(value="فاطمة علي", model_confidence=0.90, obscured=False, bbox_normalized=[320, 200, 380, 500]),
        expiry_date=RawFieldPayload(value="2030/12/31", model_confidence=0.95, obscured=False, bbox_normalized=[400, 200, 450, 450]),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False, bbox_normalized=[400, 500, 450, 750]),
        province=RawFieldPayload(value="بغداد", model_confidence=0.90, obscured=False, bbox_normalized=[460, 200, 490, 400]),
    )

    extractor = GeminiMultimodalExtractor(client=client_mock)

    blur_quality = QualityMetrics(blur_variance=40.0, glare_ratio=0.0, is_blurry=True, has_glare=False, clarity_factor=0.35)

    with patch.object(extractor, "_call_gemini_structured", return_value=(mock_nid_raw, MagicMock(logprobs_result=None))), \
         patch("src.extraction.gemini_extractor.assess_quality", return_value=blur_quality), \
         patch.object(extractor, "_retry_single_field_clahe") as mock_retry:

        result = extractor.extract_document(blank_image, DocumentType.NATIONAL_ID, enable_retry=True)

        mock_retry.assert_not_called()
        assert result.audit.retry_status == RetryStatus.BYPASSED_UNRECOVERABLE
        assert "national_id" in result.audit.unrecoverable_fields


def test_invalid_format_bypasses_retry(blank_image):
    """Verify that structurally invalid formats bypass retry and are marked unrecoverable."""
    client_mock = MagicMock()

    mock_nid_raw = RawNationalIDResponse(
        national_id=RawFieldPayload(value="12345", model_confidence=0.75, obscured=False, bbox_normalized=[100, 200, 200, 600]),
        full_name=RawFieldPayload(value="محمد حسن كاظم", model_confidence=0.95, obscured=False, bbox_normalized=[220, 200, 300, 700]),
        mother_name=RawFieldPayload(value="فاطمة علي", model_confidence=0.90, obscured=False, bbox_normalized=[320, 200, 380, 500]),
        expiry_date=RawFieldPayload(value="2030/12/31", model_confidence=0.95, obscured=False, bbox_normalized=[400, 200, 450, 450]),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False, bbox_normalized=[400, 500, 450, 750]),
        province=RawFieldPayload(value="بغداد", model_confidence=0.90, obscured=False, bbox_normalized=[460, 200, 490, 400]),
    )

    extractor = GeminiMultimodalExtractor(client=client_mock)

    with patch.object(extractor, "_call_gemini_structured", return_value=(mock_nid_raw, MagicMock(logprobs_result=None))), \
         patch.object(extractor, "_retry_single_field_clahe") as mock_retry:

        result = extractor.extract_document(blank_image, DocumentType.NATIONAL_ID, enable_retry=True)

        mock_retry.assert_not_called()
        assert result.audit.retry_status == RetryStatus.BYPASSED_UNRECOVERABLE
        assert "national_id" in result.audit.unrecoverable_fields


def test_parallel_extraction_fault_isolation(blank_image):
    """Verify that an API exception on tax card does not crash national ID or business license."""
    client_mock = MagicMock()

    mock_nid = RawNationalIDResponse(
        national_id=RawFieldPayload(value="199512345678", model_confidence=0.95, obscured=False),
        full_name=RawFieldPayload(value="علي حسن جاسم", model_confidence=0.95, obscured=False),
        mother_name=RawFieldPayload(value="زينب كريم", model_confidence=0.90, obscured=False),
        expiry_date=RawFieldPayload(value="2030/01/01", model_confidence=0.95, obscured=False),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False),
        province=RawFieldPayload(value="البصرة", model_confidence=0.90, obscured=False),
    )

    mock_biz = RawBusinessLicenseResponse(
        business_name=RawFieldPayload(value="شركة النور للتجارة", model_confidence=0.95, obscured=False),
        full_name=RawFieldPayload(value="علي حسن جاسم", model_confidence=0.95, obscured=False),
        license_number=RawFieldPayload(value="BL-2023-9988", model_confidence=0.95, obscured=False),
        issue_date=RawFieldPayload(value="2021/05/10", model_confidence=0.90, obscured=False),
    )

    def side_effect_structured(image_input, doc_type):
        if doc_type == DocumentType.NATIONAL_ID:
            return mock_nid, MagicMock(logprobs_result=None)
        if doc_type == DocumentType.BUSINESS_LICENSE:
            return mock_biz, MagicMock(logprobs_result=None)
        if doc_type == DocumentType.TAX_CARD:
            raise RuntimeError("Gemini API connection reset")
        raise ValueError()

    extractor = GeminiMultimodalExtractor(client=client_mock)

    with patch.object(extractor, "_call_gemini_structured", side_effect=side_effect_structured):
        package_result = extractor.extract_onboarding_package_parallel(
            national_id_image=blank_image,
            business_license_image=blank_image,
            tax_card_image=blank_image,
            enable_retry=True,
        )

        assert package_result.national_id.schema.national_id.value == "199512345678"
        assert package_result.business_license.schema.business_name.value == "شركة النور للتجارة"

        assert package_result.tax_card.schema.tax_id.value is None
        assert package_result.tax_card.schema.tax_id.obscured is True
        assert "WorkerException" in (package_result.tax_card.audit.retry_error or "")


def test_parallel_extraction_timeout_handling(blank_image):
    """Verify that timeout in worker thread triggers safe fallback without crashing coordinator."""
    extractor = GeminiMultimodalExtractor(request_timeout_seconds=5.0)

    mock_future = MagicMock()
    mock_future.result.side_effect = FutureTimeoutError("Timed out")

    with patch("src.extraction.gemini_extractor.ThreadPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        package_result = extractor.extract_onboarding_package_parallel(
            national_id_image=blank_image,
            business_license_image=blank_image,
            tax_card_image=blank_image,
        )

        assert package_result.national_id.schema.national_id.value is None
        assert "TimeoutError" in (package_result.national_id.audit.retry_error or "")


def test_resolve_model_signal_nested_schema_extracted():
    """Verify token logprob extraction when tokens follow nested schema {'field': {'value': '...'}}."""
    candidate_mock = MagicMock()
    token_one = MagicMock(token='"national', log_probability=None)
    token_two = MagicMock(token='_id": {', log_probability=None)
    token_three = MagicMock(token='"value": ', log_probability=None)
    token_four = MagicMock(token='"199512', log_probability=-0.04)
    token_five = MagicMock(token='345678"', log_probability=-0.02)
    token_six = MagicMock(token=',', log_probability=None)

    candidate_mock.logprobs_result.chosen_candidates = [
        token_one, token_two, token_three, token_four, token_five, token_six
    ]

    signal, status = resolve_model_signal(candidate_mock, "national_id", 0.80)
    assert status == LogprobStatus.EXTRACTED
    assert signal > 0.90


def test_invalid_format_bypasses_retry_when_retry_disabled(blank_image):
    """Verify that format-invalid fields are sanitized to null without crashing when enable_retry is False."""
    client_mock = MagicMock()

    mock_nid_raw = RawNationalIDResponse(
        national_id=RawFieldPayload(value="12345", model_confidence=0.75, obscured=False, bbox_normalized=[100, 200, 200, 600]),
        full_name=RawFieldPayload(value="محمد حسن كاظم", model_confidence=0.95, obscured=False, bbox_normalized=[220, 200, 300, 700]),
        mother_name=RawFieldPayload(value="فاطمة علي", model_confidence=0.90, obscured=False, bbox_normalized=[320, 200, 380, 500]),
        expiry_date=RawFieldPayload(value="2030/12/31", model_confidence=0.95, obscured=False, bbox_normalized=[400, 200, 450, 450]),
        issue_date=RawFieldPayload(value="2020/01/01", model_confidence=0.90, obscured=False, bbox_normalized=[400, 500, 450, 750]),
        province=RawFieldPayload(value="بغداد", model_confidence=0.90, obscured=False, bbox_normalized=[460, 200, 490, 400]),
    )

    extractor = GeminiMultimodalExtractor(client=client_mock)

    with patch.object(extractor, "_call_gemini_structured", return_value=(mock_nid_raw, MagicMock(logprobs_result=None))):
        result = extractor.extract_document(blank_image, DocumentType.NATIONAL_ID, enable_retry=False)

        assert result.audit.retry_status == RetryStatus.BYPASSED_UNRECOVERABLE
        assert "national_id" in result.audit.unrecoverable_fields
        assert result.schema.national_id.value is None
        assert result.schema.national_id.obscured is True

