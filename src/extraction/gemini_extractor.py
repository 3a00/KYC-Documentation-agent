"""Isolated document-parallel multimodal extraction pipeline using Gemini.

Enforces structured Pydantic schemas, grounds field confidence in physical OpenCV
clarity diagnostics, probes token logprobs with verbalized fallback, and executes
a strictly bounded 1x CLAHE crop retry on recoverable optical defects.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import math
import time
from typing import Any

import cv2
from google import genai
from google.genai import types
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from src.models.schemas import (
    BusinessLicenseSchema,
    ExtractedField,
    NationalIDSchema,
    TaxCardSchema,
    normalize_arabic_numerals,
)
from src.vision.diagnostics import (
    QualityMetrics,
    assess_quality,
    compute_calibrated_confidence,
    to_cv2_bgr,
)

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported Iraqi onboarding document types."""

    NATIONAL_ID = "national_id"
    BUSINESS_LICENSE = "business_license"
    TAX_CARD = "tax_card"


class RetryStatus(str, Enum):
    """Status of the bounded 1x retry mechanism."""

    NOT_NEEDED = "not_needed"
    APPLIED = "applied"
    BYPASSED_UNRECOVERABLE = "bypassed_unrecoverable"
    EXHAUSTED = "exhausted"


class LogprobStatus(str, Enum):
    """Status of token logprob extraction."""

    EXTRACTED = "extracted"
    NOT_RETURNED = "not_returned"
    ALIGNMENT_FAILED = "alignment_failed"


# Deterministic priority ordering for Tier 1 fields per document type.
# Kept strictly synchronized with Schema.get_tier1_fields().
TIER1_PRIORITY_KEYS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.NATIONAL_ID: ("national_id", "full_name", "expiry_date"),
    DocumentType.BUSINESS_LICENSE: ("license_number", "business_name", "full_name"),
    DocumentType.TAX_CARD: ("tax_id", "taxpayer_name"),
}


def make_null_field() -> ExtractedField[str]:
    """Create a fresh, distinct null ExtractedField instance for fail-closed fallbacks."""
    return ExtractedField[str](value=None, confidence=0.0, obscured=True)


def compute_package_timeout(
    request_timeout_seconds: float,
    override: float | None = None,
) -> float:
    """Compute nested package wait budget: 2x request timeout + 10s margin unless explicitly overridden.

    Allows sequential primary extraction + CLAHE retry calls within a single document worker to complete naturally.
    """
    if override is not None:
        return float(override)
    return float(request_timeout_seconds * 2 + 10.0)


class RawFieldPayload(BaseModel):
    """Raw structured extraction response from Gemini for a single field."""

    value: str | None = None
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    obscured: bool = False
    bbox_normalized: list[int] | None = None  # [ymin, xmin, ymax, xmax] in 0-1000 scale


class RawRetryPayload(BaseModel):
    """Targeted re-inspection response for a single cropped field."""

    value: str | None = None
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    obscured: bool = False


class RawNationalIDResponse(BaseModel):
    """Raw JSON extraction schema for Unified National Card."""

    national_id: RawFieldPayload
    full_name: RawFieldPayload
    mother_name: RawFieldPayload
    expiry_date: RawFieldPayload
    issue_date: RawFieldPayload
    province: RawFieldPayload


class RawBusinessLicenseResponse(BaseModel):
    """Raw JSON extraction schema for Business License."""

    business_name: RawFieldPayload
    full_name: RawFieldPayload
    license_number: RawFieldPayload
    issue_date: RawFieldPayload
    business_activity: RawFieldPayload | None = None
    province: RawFieldPayload | None = None


class RawTaxCardResponse(BaseModel):
    """Raw JSON extraction schema for General Commission for Taxes Card."""

    tax_id: RawFieldPayload
    taxpayer_name: RawFieldPayload
    fiscal_year: RawFieldPayload
    issue_date: RawFieldPayload


@dataclass(frozen=True)
class ExtractionAuditTrail:
    """Audit metadata tracking extraction execution, diagnostics, and retry decisions."""

    document_type: DocumentType
    retry_status: RetryStatus
    retried_field: str | None = None
    pre_retry_confidence: float | None = None
    post_retry_confidence: float | None = None
    unrecoverable_fields: list[str] = field(default_factory=list)
    field_qualities: dict[str, QualityMetrics] = field(default_factory=dict)
    logprob_status: LogprobStatus = LogprobStatus.NOT_RETURNED
    retry_error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass(frozen=True)
class DocumentExtractionResult:
    """Unified container for extracted schema and audit trail."""

    schema: NationalIDSchema | BusinessLicenseSchema | TaxCardSchema
    audit: ExtractionAuditTrail

    @property
    def has_unresolved_tier1_defects(self) -> bool:
        """True if any Tier 1 field remains below 0.85 calibrated confidence or unrecoverable."""
        if bool(self.audit.unrecoverable_fields):
            return True
        priority_keys = TIER1_PRIORITY_KEYS.get(self.audit.document_type, ())
        return any(
            getattr(self.schema, key).confidence < 0.85
            for key in priority_keys
            if hasattr(self.schema, key)
        )


@dataclass(frozen=True)
class OnboardingPackageExtractionResult:
    """Parallel extraction results for all three onboarding documents."""

    national_id: DocumentExtractionResult
    business_license: DocumentExtractionResult
    tax_card: DocumentExtractionResult
    total_execution_time_seconds: float


def denormalize_bounding_box(
    bbox_normalized: list[int] | tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Convert Gemini normalized [ymin, xmin, ymax, xmax] (0-1000) to pixel (xmin, ymin, xmax, ymax)."""
    if bbox_normalized is None or len(bbox_normalized) != 4:
        return None

    height, width = image_shape[:2]
    ymin_norm, xmin_norm, ymax_norm, xmax_norm = bbox_normalized

    xmin = int(np.clip(xmin_norm / 1000.0 * width, 0, width))
    xmax = int(np.clip(xmax_norm / 1000.0 * width, 0, width))
    ymin = int(np.clip(ymin_norm / 1000.0 * height, 0, height))
    ymax = int(np.clip(ymax_norm / 1000.0 * height, 0, height))

    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def enhance_crop_clahe(
    bgr_crop: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance localized text contrast and attenuate specular glare using CLAHE on LAB luminance."""
    if bgr_crop.size == 0:
        return bgr_crop

    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def crop_with_padding(
    bgr_image: np.ndarray,
    pixel_bbox: tuple[int, int, int, int],
    padding_fraction: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop localized bounding box with a margin to provide contextual background for re-extraction."""
    height, width = bgr_image.shape[:2]
    xmin, ymin, xmax, ymax = pixel_bbox

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    pad_x = int(bbox_width * padding_fraction)
    pad_y = int(bbox_height * padding_fraction)

    padded_xmin = max(0, xmin - pad_x)
    padded_xmax = min(width, xmax + pad_x)
    padded_ymin = max(0, ymin - pad_y)
    padded_ymax = min(height, ymax + pad_y)

    crop = bgr_image[padded_ymin:padded_ymax, padded_xmin:padded_xmax]
    return crop, (padded_xmin, padded_ymin, padded_xmax, padded_ymax)


def validate_field_format(field_name: str, value: str | None) -> bool:
    """Deterministically validate format constraints across Iraqi document fields.

    Domain Rules:
    1. national_id: exactly 12 numeric digits (spec.md line 71, Section 5).
    2. license_number: structured municipal commercial registration (>= 5 chars, >= 3 digits).
       Guards against hallucinated token fragments while accommodating regional municipal variants.
    3. tax_id: structured General Commission for Taxes identification (>= 5 chars, >= 4 digits).
    """
    if value is None or not str(value).strip():
        return False

    clean_str = str(value).strip()
    if field_name == "national_id":
        norm_digits = normalize_arabic_numerals(clean_str)
        return len(norm_digits) == 12 and norm_digits.isdigit()

    if field_name == "license_number":
        norm_str = normalize_arabic_numerals(clean_str).replace(" ", "").replace("-", "")
        digit_count = sum(1 for character in norm_str if character.isdigit())
        return len(norm_str) >= 5 and digit_count >= 3

    if field_name == "tax_id":
        norm_str = normalize_arabic_numerals(clean_str).replace(" ", "").replace("-", "")
        digit_count = sum(1 for character in norm_str if character.isdigit())
        return len(norm_str) >= 5 and digit_count >= 4

    return True


def resolve_model_signal(
    candidate: Any,
    field_name: str,
    verbalized_confidence: float,
) -> tuple[float, LogprobStatus]:
    """Probe Gemini token logprobs for extracted field, falling back to verbalized confidence.

    Reconstructs subword token windows to handle key fragmentation.
    """
    logprobs_result = getattr(candidate, "logprobs_result", None)
    if logprobs_result is None or not hasattr(logprobs_result, "chosen_candidates"):
        return (float(np.clip(verbalized_confidence, 0.0, 1.0)), LogprobStatus.NOT_RETURNED)

    chosen_tokens = logprobs_result.chosen_candidates
    if not chosen_tokens:
        return (float(np.clip(verbalized_confidence, 0.0, 1.0)), LogprobStatus.NOT_RETURNED)

    # Reconstruct text stream to detect field boundaries across subwords
    field_logprobs: list[float] = []
    in_target_field = False
    in_value_block = False
    window_buffer = ""

    for token_info in chosen_tokens:
        token_text = getattr(token_info, "token", "")
        token_logprob = getattr(token_info, "log_probability", None)
        window_buffer += token_text

        if not in_target_field:
            if f'"{field_name}"' in window_buffer or field_name in window_buffer:
                if ":" in window_buffer:
                    in_target_field = True
                    window_buffer = ""
            continue

        if not in_value_block:
            if "{" in window_buffer:
                if '"value"' in window_buffer and ":" in window_buffer:
                    in_value_block = True
                    window_buffer = ""
            elif '"' in window_buffer:
                in_value_block = True
                window_buffer = ""
            elif any(delim in token_text for delim in (',', '}')):
                break
            continue

        if in_value_block:
            if any(delimiter in token_text for delimiter in (',', '}', '"')):
                if field_logprobs:
                    break
            if token_logprob is not None:
                field_logprobs.append(float(token_logprob))

    if field_logprobs:
        avg_logprob = sum(field_logprobs) / len(field_logprobs)
        probability = math.exp(avg_logprob)
        return (float(np.clip(probability, 0.0, 1.0)), LogprobStatus.EXTRACTED)

    return (float(np.clip(verbalized_confidence, 0.0, 1.0)), LogprobStatus.ALIGNMENT_FAILED)


def build_system_prompt(doc_type: DocumentType) -> str:
    """Generate isolated extraction system instructions for a specific Iraqi document type."""
    base_instructions = (
        "You are an expert KYC document extraction engine for Iraqi official onboarding documents. "
        "Extract all requested fields into the specified JSON schema with zero hallucination. "
        "For each field, provide:\n"
        "- value: The extracted string exactly as printed, or null if illegible or obscured.\n"
        "- model_confidence: A float between 0.0 and 1.0 representing your character-level certainty.\n"
        "- obscured: Boolean. Set to true if the field is partially or fully covered by glare, blur, or occlusions.\n"
        "- bbox_normalized: [ymin, xmin, ymax, xmax] coordinates on a 0-1000 scale covering the field text.\n"
        "Crucial rules:\n"
        "1. Never guess or fabricate data. If a digit or word is obscured, output value=null and obscured=true.\n"
        "2. Do not normalize or translate Arabic names. Transcribe raw Arabic text faithfully.\n"
    )

    if doc_type == DocumentType.NATIONAL_ID:
        return (
            base_instructions
            + "Target Document: Iraqi Unified National Card (البطاقة الوطنية الموحدة).\n"
            + "Fields: national_id (12 digits), full_name (4-part patronymic chain), mother_name, "
            + "expiry_date (YYYY/MM/DD or DD/MM/YYYY), issue_date, province."
        )
    if doc_type == DocumentType.BUSINESS_LICENSE:
        return (
            base_instructions
            + "Target Document: Iraqi Business License / Commercial Registry (إجازة ممارسة مهنة / سجل تجاري).\n"
            + "Fields: business_name, full_name (merchant/owner), license_number, issue_date, "
            + "business_activity, province."
        )
    if doc_type == DocumentType.TAX_CARD:
        return (
            base_instructions
            + "Target Document: Iraqi General Commission for Taxes Card (الهوية الضريبية).\n"
            + "Fields: tax_id (TIN), taxpayer_name, fiscal_year, issue_date."
        )
    raise ValueError(f"Unknown document type: {doc_type}")


class GeminiMultimodalExtractor:
    """Isolated document-parallel multimodal extraction pipeline using Gemini SDK."""

    def __init__(
        self,
        client: genai.Client | None = None,
        model_name: str = "gemini-3.8-flash",
        request_timeout_seconds: float = 25.0,
        api_key: str | None = None,
    ) -> None:
        self.request_timeout_seconds = request_timeout_seconds
        http_options = types.HttpOptions(timeout=request_timeout_seconds)
        if client is not None:
            self.client = client
        elif api_key:
            self.client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            try:
                self.client = genai.Client(http_options=http_options)
            except Exception:
                # Support offline environments/tests when GEMINI_API_KEY is not set
                self.client = genai.Client(api_key="mock_key_for_testing", http_options=http_options)
        self.model_name = model_name

    def _call_gemini_structured(
        self,
        bgr_image: np.ndarray,
        doc_type: DocumentType,
    ) -> tuple[Any, Any]:
        """Invoke Gemini multimodal API with isolated prompt and strict JSON schema."""
        pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        system_prompt = build_system_prompt(doc_type)

        schema_map = {
            DocumentType.NATIONAL_ID: RawNationalIDResponse,
            DocumentType.BUSINESS_LICENSE: RawBusinessLicenseResponse,
            DocumentType.TAX_CARD: RawTaxCardResponse,
        }
        target_schema = schema_map[doc_type]

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=target_schema,
            temperature=0.0,
            response_logprobs=True,
            logprobs=5,
            system_instruction=system_prompt,
            http_options=types.HttpOptions(timeout=self.request_timeout_seconds),
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                pil_image,
                f"Extract all fields from this front-face {doc_type.value} photograph into the structured schema.",
            ],
            config=config,
        )

        candidate = response.candidates[0] if response.candidates else None
        text_content = response.text or "{}"
        parsed_json = json.loads(text_content)
        raw_response = target_schema.model_validate(parsed_json)
        return raw_response, candidate

    def _retry_single_field_clahe(
        self,
        enhanced_crop_bgr: np.ndarray,
        doc_type: DocumentType,
        field_name: str,
    ) -> RawRetryPayload:
        """Execute targeted single-field re-extraction on CLAHE contrast-enhanced crop."""
        pil_crop = Image.fromarray(cv2.cvtColor(enhanced_crop_bgr, cv2.COLOR_BGR2RGB))
        prompt = (
            f"This is a CLAHE contrast-enhanced crop of the field '{field_name}' from an Iraqi {doc_type.value}. "
            "Carefully re-examine the text characters and digits. "
            "Output the JSON object for this field with value, model_confidence, and obscured."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RawRetryPayload,
            temperature=0.0,
            system_instruction="You are a specialized optical re-inspection tool for degraded document fields.",
            http_options=types.HttpOptions(timeout=self.request_timeout_seconds),
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[pil_crop, prompt],
            config=config,
        )
        return RawRetryPayload.model_validate(json.loads(response.text or "{}"))

    def _process_field(
        self,
        raw_payload: RawFieldPayload,
        field_name: str,
        bgr_image: np.ndarray,
        candidate: Any,
    ) -> tuple[ExtractedField[str], QualityMetrics, LogprobStatus]:
        """Convert raw payload into ExtractedField, computing calibrated composite confidence."""
        pixel_bbox = denormalize_bounding_box(raw_payload.bbox_normalized, bgr_image.shape[:2])
        quality = assess_quality(bgr_image, bbox=pixel_bbox)

        if raw_payload.obscured or raw_payload.value is None or not str(raw_payload.value).strip():
            extracted = ExtractedField[str](
                value=None,
                confidence=0.0,
                obscured=True,
                bbox=pixel_bbox,
            )
            return extracted, quality, LogprobStatus.NOT_RETURNED

        clean_value = str(raw_payload.value).strip()
        model_signal, logprob_status = resolve_model_signal(
            candidate, field_name, raw_payload.model_confidence
        )

        format_valid = validate_field_format(field_name, clean_value)
        calibrated_conf = compute_calibrated_confidence(
            model_signal=model_signal,
            clarity_factor=quality.clarity_factor,
            format_valid=format_valid,
        )

        extracted = ExtractedField[str](
            value=clean_value,
            confidence=calibrated_conf,
            obscured=False,
            bbox=pixel_bbox,
        )
        return extracted, quality, logprob_status

    def _create_escalated_fallback_schema(
        self, doc_type: DocumentType
    ) -> NationalIDSchema | BusinessLicenseSchema | TaxCardSchema:
        """Create a safe, fail-closed Pydantic schema with fresh null instances for unhandled API failures."""
        if doc_type == DocumentType.NATIONAL_ID:
            return NationalIDSchema(
                national_id=make_null_field(),
                full_name=make_null_field(),
                mother_name=make_null_field(),
                expiry_date=make_null_field(),
                issue_date=make_null_field(),
                province=make_null_field(),
            )
        if doc_type == DocumentType.BUSINESS_LICENSE:
            return BusinessLicenseSchema(
                business_name=make_null_field(),
                full_name=make_null_field(),
                license_number=make_null_field(),
                issue_date=make_null_field(),
            )
        return TaxCardSchema(
            tax_id=make_null_field(),
            taxpayer_name=make_null_field(),
            fiscal_year=make_null_field(),
            issue_date=make_null_field(),
        )

    def extract_document(
        self,
        image: Image.Image | np.ndarray,
        doc_type: DocumentType,
        enable_retry: bool = True,
    ) -> DocumentExtractionResult:
        """Extract a single document in isolated context with grounded confidence and bounded 1x retry."""
        start_time = time.perf_counter()
        bgr_image = to_cv2_bgr(image)
        image_shape = bgr_image.shape[:2]

        try:
            raw_data, candidate = self._call_gemini_structured(bgr_image, doc_type)
        except Exception as exc:
            logger.error("Primary Gemini extraction failed for %s: %s", doc_type.value, exc)
            fallback_schema = self._create_escalated_fallback_schema(doc_type)
            audit = ExtractionAuditTrail(
                document_type=doc_type,
                retry_status=RetryStatus.BYPASSED_UNRECOVERABLE,
                retry_error=f"WorkerException: {type(exc).__name__}: {exc}",
                execution_time_seconds=time.perf_counter() - start_time,
            )
            return DocumentExtractionResult(schema=fallback_schema, audit=audit)

        ordered_tier1_keys = TIER1_PRIORITY_KEYS[doc_type]
        extracted_fields: dict[str, ExtractedField[str]] = {}
        field_qualities: dict[str, QualityMetrics] = {}
        resolved_logprob_status = LogprobStatus.NOT_RETURNED

        raw_dict = raw_data.model_dump()
        for key, raw_val in raw_dict.items():
            if raw_val is None:
                continue
            payload = RawFieldPayload.model_validate(raw_val)
            field_obj, quality, logprob_stat = self._process_field(
                raw_payload=payload,
                field_name=key,
                bgr_image=bgr_image,
                candidate=candidate,
            )
            extracted_fields[key] = field_obj
            field_qualities[key] = quality
            if logprob_stat == LogprobStatus.EXTRACTED:
                resolved_logprob_status = LogprobStatus.EXTRACTED
            elif resolved_logprob_status != LogprobStatus.EXTRACTED and logprob_stat == LogprobStatus.ALIGNMENT_FAILED:
                resolved_logprob_status = LogprobStatus.ALIGNMENT_FAILED

        # Bounded 1x Retry Evaluation
        retry_status = RetryStatus.NOT_NEEDED
        retried_field_name: str | None = None
        pre_retry_confidence: float | None = None
        post_retry_confidence: float | None = None
        unrecoverable_fields: list[str] = []
        retry_error_message: str | None = None

        if enable_retry:
            recoverable_candidate: str | None = None

            # Scan all Tier 1 fields in deterministic priority order
            for key in ordered_tier1_keys:
                field_item = extracted_fields.get(key)
                if field_item is None:
                    continue

                if field_item.confidence < 0.85:
                    if field_item.obscured or not field_item.bbox:
                        unrecoverable_fields.append(key)
                        continue

                    # Direct format validation check
                    format_valid = validate_field_format(key, field_item.value)
                    if not format_valid:
                        unrecoverable_fields.append(key)
                        continue

                    quality = field_qualities[key]
                    # CLAHE is contrast-limited equalization: fixes glare and low contrast, NOT optical blur
                    is_recoverable_optical_defect = quality.has_glare or (
                        quality.clarity_factor < 0.70 and not quality.is_blurry
                    )

                    if not is_recoverable_optical_defect:
                        unrecoverable_fields.append(key)
                        continue

                    # Select highest-priority recoverable candidate
                    if recoverable_candidate is None:
                        recoverable_candidate = key

            # Execute bounded 1x retry on selected recoverable field
            if recoverable_candidate is not None:
                retried_field_name = recoverable_candidate
                candidate_field = extracted_fields[recoverable_candidate]
                pre_retry_confidence = candidate_field.confidence
                assert candidate_field.bbox is not None

                crop, _ = crop_with_padding(bgr_image, candidate_field.bbox)
                enhanced_crop = enhance_crop_clahe(crop)

                try:
                    retry_payload = self._retry_single_field_clahe(
                        enhanced_crop_bgr=enhanced_crop,
                        doc_type=doc_type,
                        field_name=recoverable_candidate,
                    )

                    # Real OpenCV clarity measurement on the enhanced crop without artificial floors
                    enhanced_quality = assess_quality(enhanced_crop)
                    format_ok = validate_field_format(recoverable_candidate, retry_payload.value)

                    new_conf = compute_calibrated_confidence(
                        model_signal=retry_payload.model_confidence,
                        clarity_factor=enhanced_quality.clarity_factor,
                        format_valid=format_ok,
                    )

                    if new_conf > candidate_field.confidence and retry_payload.value is not None:
                        extracted_fields[recoverable_candidate] = ExtractedField[str](
                            value=str(retry_payload.value).strip(),
                            confidence=new_conf,
                            obscured=False,
                            bbox=candidate_field.bbox,
                        )
                        field_qualities[recoverable_candidate] = enhanced_quality
                        post_retry_confidence = new_conf
                        retry_status = RetryStatus.APPLIED
                    else:
                        post_retry_confidence = candidate_field.confidence
                        retry_status = RetryStatus.EXHAUSTED
                except Exception as exc:
                    logger.warning("CLAHE retry failed on %s: %s", recoverable_candidate, exc)
                    retry_status = RetryStatus.EXHAUSTED
                    retry_error_message = f"{type(exc).__name__}: {exc}"
                    post_retry_confidence = candidate_field.confidence
            elif unrecoverable_fields:
                retry_status = RetryStatus.BYPASSED_UNRECOVERABLE

        # Convert any format-invalid fields to fail-closed null ExtractedField instances
        # to satisfy strict downstream Pydantic schema validation rules while preserving defect tracking in audit
        for field_key, field_obj in extracted_fields.items():
            if field_obj.value is not None and not validate_field_format(field_key, field_obj.value):
                if field_key not in unrecoverable_fields:
                    unrecoverable_fields.append(field_key)
                extracted_fields[field_key] = ExtractedField[str](
                    value=None,
                    confidence=0.0,
                    obscured=True,
                    bbox=field_obj.bbox,
                )
                if retry_status == RetryStatus.NOT_NEEDED:
                    retry_status = RetryStatus.BYPASSED_UNRECOVERABLE

        # Construct validated target document schema
        schema_instance: NationalIDSchema | BusinessLicenseSchema | TaxCardSchema
        if doc_type == DocumentType.NATIONAL_ID:
            schema_instance = NationalIDSchema.model_validate(extracted_fields)
        elif doc_type == DocumentType.BUSINESS_LICENSE:
            schema_instance = BusinessLicenseSchema.model_validate(extracted_fields)
        elif doc_type == DocumentType.TAX_CARD:
            schema_instance = TaxCardSchema.model_validate(extracted_fields)

        execution_seconds = time.perf_counter() - start_time
        audit = ExtractionAuditTrail(
            document_type=doc_type,
            retry_status=retry_status,
            retried_field=retried_field_name,
            pre_retry_confidence=pre_retry_confidence,
            post_retry_confidence=post_retry_confidence,
            unrecoverable_fields=unrecoverable_fields,
            field_qualities=field_qualities,
            logprob_status=resolved_logprob_status,
            retry_error=retry_error_message,
            execution_time_seconds=execution_seconds,
        )

        return DocumentExtractionResult(schema=schema_instance, audit=audit)

    def extract_onboarding_package_parallel(
        self,
        national_id_image: Image.Image | np.ndarray,
        business_license_image: Image.Image | np.ndarray,
        tax_card_image: Image.Image | np.ndarray,
        enable_retry: bool = True,
        package_timeout_seconds: float | None = None,
    ) -> OnboardingPackageExtractionResult:
        """Extract all 3 documents in parallel with nested timeout budgeting and non-blocking shutdown."""
        start_time = time.perf_counter()
        effective_timeout = compute_package_timeout(self.request_timeout_seconds, package_timeout_seconds)

        executor = ThreadPoolExecutor(max_workers=3)

        try:
            future_nid = executor.submit(
                self.extract_document, national_id_image, DocumentType.NATIONAL_ID, enable_retry
            )
            future_biz = executor.submit(
                self.extract_document, business_license_image, DocumentType.BUSINESS_LICENSE, enable_retry
            )
            future_tax = executor.submit(
                self.extract_document, tax_card_image, DocumentType.TAX_CARD, enable_retry
            )

            # Gather results with properly nested package-level timeout bound
            results: dict[DocumentType, DocumentExtractionResult] = {}
            for doc_type, future in [
                (DocumentType.NATIONAL_ID, future_nid),
                (DocumentType.BUSINESS_LICENSE, future_biz),
                (DocumentType.TAX_CARD, future_tax),
            ]:
                try:
                    results[doc_type] = future.result(timeout=effective_timeout)
                except FutureTimeoutError:
                    logger.error("Document extraction timed out after %s seconds: %s", effective_timeout, doc_type.value)
                    results[doc_type] = DocumentExtractionResult(
                        schema=self._create_escalated_fallback_schema(doc_type),
                        audit=ExtractionAuditTrail(
                            document_type=doc_type,
                            retry_status=RetryStatus.BYPASSED_UNRECOVERABLE,
                            retry_error=f"TimeoutError: Exceeded {effective_timeout}s package bound",
                        ),
                    )
                except Exception as exc:
                    logger.error("Unhandled worker exception on %s: %s", doc_type.value, exc)
                    results[doc_type] = DocumentExtractionResult(
                        schema=self._create_escalated_fallback_schema(doc_type),
                        audit=ExtractionAuditTrail(
                            document_type=doc_type,
                            retry_status=RetryStatus.BYPASSED_UNRECOVERABLE,
                            retry_error=f"WorkerException: {type(exc).__name__}: {exc}",
                        ),
                    )
        finally:
            # cancel_futures cancels unstarted tasks. Note: in-flight threads run until their
            # HttpOptions socket timeout releases them; wait=False ensures this coordinator returns immediately.
            executor.shutdown(wait=False, cancel_futures=True)

        total_duration = time.perf_counter() - start_time
        return OnboardingPackageExtractionResult(
            national_id=results[DocumentType.NATIONAL_ID],
            business_license=results[DocumentType.BUSINESS_LICENSE],
            tax_card=results[DocumentType.TAX_CARD],
            total_execution_time_seconds=total_duration,
        )
