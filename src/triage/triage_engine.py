"""Deterministic KYC onboarding triage engine and exception dossier generator.

Implements the top-level verification seam coordinating document-parallel extraction,
tiered field confidence policies (Tier 1 >= 0.85, Tier 2 >= 0.70), cross-document
identity reconciliation with bipartite fallback, and auditable Arabic exception
reporting for human operations officers.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
import logging
from PIL import Image

from src.extraction.gemini_extractor import (
    DocumentExtractionResult,
    GeminiMultimodalExtractor,
    OnboardingPackageExtractionResult,
)
from src.matching.arabic_matcher import (
    CrossDocumentReconciliationResult,
    IdentityMatchResult,
    MatchBand,
    TokenMatchDetail,
    match_arabic_names,
    reconcile_onboarding_identities,
)
from src.models.schemas import (
    BusinessLicenseSchema,
    ExtractedField,
    NationalIDSchema,
    TaxCardSchema,
)

logger = logging.getLogger(__name__)


class TriageLifecycle(str, Enum):
    """Deterministic lifecycle triage outcome for an onboarding package."""

    AUTO_PASS = "AUTO_PASS"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
    HARD_MISMATCH = "HARD_MISMATCH"


# Arabic translations for document field names
FIELD_NAMES_AR: dict[str, str] = {
    # Unified National Card
    "national_id": "رقم البطاقة الوطنية",
    "full_name": "الاسم الرباعي واللقب",
    "mother_name": "اسم الأم الثلاثي",
    "expiry_date": "تاريخ انتهاء النفاذ",
    "issue_date": "تاريخ الإصدار",
    "province": "المحافظة",
    # Business License
    "business_name": "اسم المنشأة / الاسم التجاري",
    "license_number": "رقم إجازة المهنة",
    "business_activity": "نوع النشاط التجاري",
    # Tax Card
    "tax_id": "الرقم الضريبي (TIN)",
    "taxpayer_name": "اسم المكلف الضريبي",
    "fiscal_year": "السنة المالية",
}

# Arabic translations for document types (including pair-split aliases)
DOC_TYPES_AR: dict[str, str] = {
    # Full schema keys
    "national_id": "البطاقة الوطنية الموحدة",
    "business_license": "إجازة ممارسة المهنة / السجل التجاري",
    "tax_card": "الهوية الضريبية",
    "package": "حزمة الوثائق الكاملة",
    # Short pair-split keys
    "national": "البطاقة الوطنية الموحدة",
    "business": "إجازة ممارسة المهنة / السجل التجاري",
    "tax": "الهوية الضريبية",
}

DOC_TYPES_EN: dict[str, str] = {
    "national_id": "National ID",
    "business_license": "Business License",
    "tax_card": "Tax Card",
    "package": "Full Document Package",
    "national": "National ID",
    "business": "Business License",
    "tax": "Tax Card",
}

AR_TO_EN_DOC_TYPES: dict[str, str] = {
    "البطاقة الوطنية الموحدة": "National ID",
    "إجازة ممارسة المهنة / السجل التجاري": "Business License",
    "إجازة ممارسة المهنة": "Business License",
    "الهوية الضريبية": "Tax Card",
    "حزمة الوثائق الكاملة": "Full Document Package",
}


def format_pair_label_en(pair_key: str, doc_a_name: str, doc_b_name: str) -> str:
    """Format a pairwise cross-document comparison label in English."""
    doc_a_en = AR_TO_EN_DOC_TYPES.get(doc_a_name, doc_a_name.replace("_", " ").title())
    doc_b_en = AR_TO_EN_DOC_TYPES.get(doc_b_name, doc_b_name.replace("_", " ").title())
    if "_vs_" in pair_key:
        k_a, k_b = pair_key.split("_vs_")
        doc_a_en = DOC_TYPES_EN.get(k_a, doc_a_en)
        doc_b_en = DOC_TYPES_EN.get(k_b, doc_b_en)
    return f"{doc_a_en} vs {doc_b_en}"


TIER1_CONFIDENCE_THRESHOLD = 0.85
TIER2_CONFIDENCE_THRESHOLD = 0.70


@dataclass(frozen=True)
class FieldAnomaly:
    """Detailed record of a field confidence anomaly or extraction defect.

    Attributes:
        bbox: Normalized coordinates [ymin, xmin, ymax, xmax] in 0-1000 scale.
    """

    document_type: str
    document_type_ar: str
    field_name: str
    field_name_ar: str
    tier: int
    confidence: float
    threshold: float
    value: str | None
    obscured: bool
    bbox: list[int] | tuple[int, int, int, int] | None
    reason: str
    reason_ar: str

    def __post_init__(self) -> None:
        """Validate bounding box coordinate format and geometric integrity."""
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"Normalized bbox must contain exactly 4 coordinates, got {self.bbox}")
            ymin, xmin, ymax, xmax = self.bbox
            if not all(0 <= coord <= 1000 for coord in (ymin, xmin, ymax, xmax)):
                raise ValueError(f"Normalized coordinates must be within [0, 1000], got {self.bbox}")
            if ymin > ymax or xmin > xmax:
                raise ValueError(f"Invalid coordinate geometry: ymin<=ymax and xmin<=xmax required, got {self.bbox}")


@dataclass(frozen=True)
class NameMismatchDetail:
    """Pairwise cross-document name comparison detail with slot-level breakdown."""

    pair_key: str
    doc_a_name: str
    doc_b_name: str
    raw_name_a: str
    raw_name_b: str
    similarity_score: float
    triage_band: MatchBand
    token_details: list[TokenMatchDetail] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
    audit_notes_ar: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OnboardingDossier:
    """Auditable onboarding decision dossier containing triage state and exception details."""

    lifecycle_outcome: TriageLifecycle
    overall_confidence: float
    min_tier1_confidence: float
    min_matching_score: float | None = None
    tier1_passed: bool = True
    tier1_anomalies: list[FieldAnomaly] = field(default_factory=list)
    tier2_warnings: list[FieldAnomaly] = field(default_factory=list)
    cross_matching_result: CrossDocumentReconciliationResult | None = None
    name_mismatches: list[NameMismatchDetail] = field(default_factory=list)
    audit_trail: list[str] = field(default_factory=list)
    audit_trail_ar: list[str] = field(default_factory=list)
    actionable_summary_ar: str = ""
    extraction_package: OnboardingPackageExtractionResult | None = None

    def to_markdown_report(self) -> str:
        """Generate a structured, presentation-ready Arabic Markdown exception report."""
        lines: list[str] = []
        lines.append("# تقرير تدقيق التحقق من وثائق الانضمام (KYC Onboarding Dossier)")
        lines.append("")

        # Decision Header
        if self.lifecycle_outcome == TriageLifecycle.AUTO_PASS:
            lines.append("## القرار النهائي: **قبول تلقائي (AUTO_PASS)**")
            lines.append("> تم التحقق بنجاح من كافة الوثائق والحقول الأساسية ومطابقة الهوية.")
        elif self.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION:
            lines.append("## القرار النهائي: **مراجعة يدوية مطلوبة (HUMAN_ESCALATION)**")
            lines.append("> يتطلب هذا الملف مراجعة موظف العمليات المختص للأسباب المبينة أدناه.")
        else:
            lines.append("## القرار النهائي: **رفض بسبب عدم تطابق جوهري (HARD_MISMATCH)**")
            lines.append("> تم رصد اختلاف جوهري غير قابل للتسوية في الهوية عبر الوثائق المقدمة.")

        lines.append("")
        min_match_str = f"`{self.min_matching_score:.1%}`" if self.min_matching_score is not None else "*(غير متوفر)*"
        lines.append(f"- **معدل الثقة الإجمالي**: `{self.overall_confidence:.1%}`")
        lines.append(f"- **أدنى ثقة في الحقول الأساسية (Worst Tier 1)**: `{self.min_tier1_confidence:.1%}` (الحد الأدنى: 85%)")
        lines.append(f"- **أدنى نسبة تطابق بين الأسماء (Min Identity Match)**: {min_match_str} (عتبة القبول: 88%)")
        lines.append(f"- **سلامة الحقول الأساسية (Tier 1)**: `{'ناجح' if self.tier1_passed else 'فاشل'}`")
        lines.append("")

        # Actionable Summary
        lines.append("### ملخص القرار التوجيهي")
        lines.append(self.actionable_summary_ar)
        lines.append("")

        # Tier 1 Anomalies Table
        if self.tier1_anomalies:
            lines.append("### الحقول الأساسية المتعثرة (Tier 1 - تتطلب ثقة >= 85%)")
            lines.append("| الوثيقة | الحقل | القيمة المستخرجة | الثقة المحسوبة | العتبة | الإحداثيات البصرية | سبب التعثر |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for anomaly in self.tier1_anomalies:
                val = f"`{anomaly.value}`" if anomaly.value else "*(محجوب / غير مقروء)*"
                bbox_str = (
                    f"`ymin={anomaly.bbox[0]}, xmin={anomaly.bbox[1]}, ymax={anomaly.bbox[2]}, xmax={anomaly.bbox[3]}`"
                    if anomaly.bbox
                    else "*(غير متوفر)*"
                )
                lines.append(
                    f"| {anomaly.document_type_ar} | {anomaly.field_name_ar} | {val} | "
                    f"`{anomaly.confidence:.1%}` | `{anomaly.threshold:.0%}` | {bbox_str} | {anomaly.reason_ar} |"
                )
            lines.append("")

        # Cross-Document Name Comparisons
        if self.name_mismatches:
            lines.append("### نتائج مطابقة الأسماء عبر الوثائق (Cross-Document Identity)")
            lines.append("| طرفي المقارنة | الاسم في الوثيقة الأولى | الاسم في الوثيقة الثانية | نسبة التطابق | التصنيف | ملاحظات التدقيق |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for mismatch in self.name_mismatches:
                notes_str = " - ".join(mismatch.audit_notes_ar) if mismatch.audit_notes_ar else "تطابق تام"
                lines.append(
                    f"| {mismatch.doc_a_name} مقابل {mismatch.doc_b_name} | `{mismatch.raw_name_a}` | "
                    f"`{mismatch.raw_name_b}` | `{mismatch.similarity_score:.1%}` | `{mismatch.triage_band.value}` | {notes_str} |"
                )
            lines.append("")

            # Render Aligned Patronymic Slot Diff Table (sorted by discrepancy)
            has_slots = any(bool(mismatch.token_details) for mismatch in self.name_mismatches)
            if has_slots:
                lines.append("#### جدول الفروقات التفصيلي لسلاسل النسب (Patronymic Slot Diff)")
                lines.append("| طرفي المقارنة | الرتبة النسبية | المقطع في الوثيقة الأولى | المقطع في الوثيقة الثانية | نسبة التشابه | الوزن | الأثر النسبي |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                slot_names_ar = {
                    "given": "الاسم الأول (Given)",
                    "father": "اسم الأب (Father)",
                    "grandfather": "اسم الجد (Grandfather)",
                    "surname": "اللقب / العشيرة (Surname)",
                }
                for mismatch in self.name_mismatches:
                    pair_label = f"{mismatch.doc_a_name} مقابل {mismatch.doc_b_name}"
                    sorted_slots = sorted(mismatch.token_details, key=lambda slot: slot.similarity)
                    is_escalated_or_mismatch = mismatch.triage_band in (
                        MatchBand.HUMAN_ESCALATION,
                        MatchBand.HARD_MISMATCH,
                    )
                    for slot in sorted_slots:
                        role_key = slot.role.value
                        role_str = slot_names_ar.get(role_key, role_key)
                        tok_a = f"`{slot.token_a}`" if slot.token_a else "*(غير مذكور)*"
                        tok_b = f"`{slot.token_b}`" if slot.token_b else "*(غير مذكور)*"

                        # Only apply alarmist bolding if the pair actually escalated or hard-mismatched
                        if is_escalated_or_mismatch and slot.similarity < 0.88:
                            impact = "**تباين مؤثر**"
                            sim_str = f"**{slot.similarity:.1%}**"
                        elif slot.similarity >= 0.88:
                            impact = "مطابق"
                            sim_str = f"{slot.similarity:.1%}"
                        else:
                            impact = "مقبول ضمن النطاق الكلي"
                            sim_str = f"{slot.similarity:.1%}"

                        lines.append(
                            f"| {pair_label} | {role_str} | {tok_a} | {tok_b} | {sim_str} | `{slot.weight:.2f}` | {impact} |"
                        )

                    # Only emit discrepancy driver callout if the pair actually escalated or hard-mismatched
                    if is_escalated_or_mismatch:
                        diverging_slots = [slot for slot in mismatch.token_details if slot.similarity < 0.88]
                        if diverging_slots:
                            worst_slot = min(diverging_slots, key=lambda slot: slot.similarity)
                            role_ar = slot_names_ar.get(worst_slot.role.value, worst_slot.role.value)
                            lines.append("")
                            lines.append(
                                f"> **السبب الأبرز للتباين ({pair_label})**: `{role_ar}` بنسبة تشابه "
                                f"`{worst_slot.similarity:.1%}` ووزن `{worst_slot.weight:.2f}`."
                            )
                lines.append("")

        # Tier 2 Warnings
        if self.tier2_warnings:
            lines.append("### تنبيهات الحقول الثانوية (Tier 2 - غير معطلة للقبول)")
            lines.append("| الوثيقة | الحقل | القيمة | الثقة المحسوبة | ملاحظة |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for warning in self.tier2_warnings:
                val = f"`{warning.value}`" if warning.value else "*(محجوب)*"
                lines.append(
                    f"| {warning.document_type_ar} | {warning.field_name_ar} | {val} | `{warning.confidence:.1%}` | {warning.reason_ar} |"
                )
            lines.append("")

        # Audit Log (Arabic for UI)
        if self.audit_trail_ar:
            lines.append("### السجل الكامل لعمليات الفحص")
            for entry in self.audit_trail_ar:
                lines.append(f"- {entry}")
            lines.append("")

        return "\n".join(lines)

    def to_markdown_report_en(self) -> str:
        """Generate a structured, presentation-ready English Markdown exception report."""
        lines: list[str] = []
        lines.append("# KYC Onboarding Verification & Audit Dossier")
        lines.append("")

        # Decision Header
        if self.lifecycle_outcome == TriageLifecycle.AUTO_PASS:
            lines.append("## Final Decision: **Approved (AUTO_PASS)**")
            lines.append("> All onboarding documents, mandatory fields, and identity cross-checks successfully verified.")
        elif self.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION:
            lines.append("## Final Decision: **Manual Review Required (HUMAN_ESCALATION)**")
            lines.append("> Manual inspection by a compliance officer is required for the reasons detailed below.")
        else:
            lines.append("## Final Decision: **Rejected (HARD_MISMATCH)**")
            lines.append("> Critical irreconcilable identity discrepancy detected across submitted documents.")

        lines.append("")
        min_match_str = f"`{self.min_matching_score:.1%}`" if self.min_matching_score is not None else "*(N/A)*"
        lines.append(f"- **Overall Calibrated Confidence**: `{self.overall_confidence:.1%}`")
        lines.append(f"- **Worst Mandatory Field (Tier 1)**: `{self.min_tier1_confidence:.1%}` (Threshold: 85%)")
        lines.append(f"- **Minimum Name Match Score**: {min_match_str} (Auto-pass threshold: 88%)")
        lines.append(f"- **Mandatory Fields Integrity**: `{'PASSED' if self.tier1_passed else 'FAILED'}`")
        lines.append("")

        # Actionable Summary
        if self.lifecycle_outcome == TriageLifecycle.AUTO_PASS:
            actionable_summary_en = (
                "Automatic approval granted. All mandatory (Tier 1) fields extracted with >= 85% confidence, "
                "and applicant name alignment across all documents exceeds the automatic pass threshold (88%)."
            )
        elif self.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION:
            actionable_summary_en = (
                "File referred for manual compliance review. Document contains degraded mandatory fields "
                "or partial name alignment variance requiring human verification."
            )
        else:
            actionable_summary_en = (
                "Application rejected due to an irreconcilable identity conflict across submitted documents "
                "(name matching similarity < 70%)."
            )
        lines.append("### Summary & Recommended Action")
        lines.append(actionable_summary_en)
        lines.append("")

        # Tier 1 Anomalies Table
        if self.tier1_anomalies:
            lines.append("### Deficient Mandatory Fields (Tier 1 — Requires >= 85% Confidence)")
            lines.append("| Document | Field | Extracted Value | Confidence | Threshold | Coordinates | Defect Reason |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for anomaly in self.tier1_anomalies:
                val = f"`{anomaly.value}`" if anomaly.value else "*(Obscured / Illegible)*"
                bbox_str = (
                    f"`ymin={anomaly.bbox[0]}, xmin={anomaly.bbox[1]}, ymax={anomaly.bbox[2]}, xmax={anomaly.bbox[3]}`"
                    if anomaly.bbox
                    else "*(N/A)*"
                )
                doc_name_en = anomaly.document_type.replace("_", " ").title()
                field_name_en = anomaly.field_name.replace("_", " ").title()
                lines.append(
                    f"| {doc_name_en} | {field_name_en} | {val} | "
                    f"`{anomaly.confidence:.1%}` | `{anomaly.threshold:.0%}` | {bbox_str} | {anomaly.reason} |"
                )
            lines.append("")

        # Cross-Document Name Comparisons
        if self.name_mismatches:
            lines.append("### Cross-Document Identity Reconciliation")
            lines.append("| Comparison Pair | Name in First Document | Name in Second Document | Similarity | Classification | Inspection Notes |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for mismatch in self.name_mismatches:
                notes_str = " - ".join(mismatch.audit_notes) if mismatch.audit_notes else "Full Match"
                pair_label = format_pair_label_en(mismatch.pair_key, mismatch.doc_a_name, mismatch.doc_b_name)
                lines.append(
                    f"| {pair_label} | `{mismatch.raw_name_a}` | "
                    f"`{mismatch.raw_name_b}` | `{mismatch.similarity_score:.1%}` | `{mismatch.triage_band.value}` | {notes_str} |"
                )
            lines.append("")

            # Render Aligned Patronymic Slot Diff Table
            has_slots = any(bool(mismatch.token_details) for mismatch in self.name_mismatches)
            if has_slots:
                lines.append("#### Detailed Patronymic Slot Breakdown")
                lines.append("| Comparison Pair | Slot | Name in Doc (A) | Name in Doc (B) | Similarity | Weight | Impact |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                slot_names_en = {
                    "given": "Given Name",
                    "father": "Father's Name",
                    "grandfather": "Grandfather's Name",
                    "surname": "Surname / Clan",
                }
                for mismatch in self.name_mismatches:
                    pair_label = format_pair_label_en(mismatch.pair_key, mismatch.doc_a_name, mismatch.doc_b_name)
                    sorted_slots = sorted(mismatch.token_details, key=lambda slot: slot.similarity)
                    is_escalated_or_mismatch = mismatch.triage_band in (
                        MatchBand.HUMAN_ESCALATION,
                        MatchBand.HARD_MISMATCH,
                    )
                    for slot in sorted_slots:
                        role_key = slot.role.value
                        role_str = slot_names_en.get(role_key, role_key.title())
                        tok_a = f"`{slot.token_a}`" if slot.token_a else "*(Omitted)*"
                        tok_b = f"`{slot.token_b}`" if slot.token_b else "*(Omitted)*"

                        if is_escalated_or_mismatch and slot.similarity < 0.88:
                            impact = "**Divergent**"
                            sim_str = f"**{slot.similarity:.1%}**"
                        elif slot.similarity >= 0.88:
                            impact = "Match"
                            sim_str = f"{slot.similarity:.1%}"
                        else:
                            impact = "Acceptable Variance"
                            sim_str = f"{slot.similarity:.1%}"

                        lines.append(
                            f"| {pair_label} | {role_str} | {tok_a} | {tok_b} | {sim_str} | `{slot.weight:.2f}` | {impact} |"
                        )
                lines.append("")

        # Tier 2 Warnings
        if self.tier2_warnings:
            lines.append("### Secondary Field Warnings (Tier 2 — Non-Blocking)")
            lines.append("| Document | Field | Value | Confidence | Note |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for warning in self.tier2_warnings:
                val = f"`{warning.value}`" if warning.value else "*(Obscured)*"
                doc_name_en = warning.document_type.replace("_", " ").title()
                field_name_en = warning.field_name.replace("_", " ").title()
                lines.append(
                    f"| {doc_name_en} | {field_name_en} | {val} | `{warning.confidence:.1%}` | {warning.reason} |"
                )
            lines.append("")

        # System Audit Log
        if self.audit_trail:
            lines.append("### Chronological System Audit Trail")
            for entry in self.audit_trail:
                lines.append(f"- {entry}")
            lines.append("")

        return "\n".join(lines)


def _inspect_document_fields(
    doc_type_key: str,
    doc_schema: NationalIDSchema | BusinessLicenseSchema | TaxCardSchema,
) -> tuple[list[FieldAnomaly], list[FieldAnomaly], list[float], list[float]]:
    """Scan document schema fields and separate into Tier 1 anomalies, Tier 2 warnings, and confidences."""
    tier1_anomalies: list[FieldAnomaly] = []
    tier2_warnings: list[FieldAnomaly] = []
    confidences: list[float] = []
    tier1_confidences: list[float] = []

    doc_name_ar = DOC_TYPES_AR.get(doc_type_key, doc_type_key)

    # 1. Inspect Tier 1 Critical Fields
    tier1_fields = doc_schema.get_tier1_fields()
    for field_key, field_obj in tier1_fields.items():
        confidences.append(field_obj.confidence)
        tier1_confidences.append(field_obj.confidence)
        field_name_ar = FIELD_NAMES_AR.get(field_key, field_key)

        if field_obj.obscured or field_obj.value is None:
            tier1_anomalies.append(
                FieldAnomaly(
                    document_type=doc_type_key,
                    document_type_ar=doc_name_ar,
                    field_name=field_key,
                    field_name_ar=field_name_ar,
                    tier=1,
                    confidence=field_obj.confidence,
                    threshold=TIER1_CONFIDENCE_THRESHOLD,
                    value=None,
                    obscured=True,
                    bbox=field_obj.bbox,
                    reason=f"Field '{field_key}' is obscured or unreadable.",
                    reason_ar=f"الحقل '{field_name_ar}' محجوب أو غير مقروء في الصورة.",
                )
            )
        elif field_obj.confidence < TIER1_CONFIDENCE_THRESHOLD:
            tier1_anomalies.append(
                FieldAnomaly(
                    document_type=doc_type_key,
                    document_type_ar=doc_name_ar,
                    field_name=field_key,
                    field_name_ar=field_name_ar,
                    tier=1,
                    confidence=field_obj.confidence,
                    threshold=TIER1_CONFIDENCE_THRESHOLD,
                    value=field_obj.value,
                    obscured=False,
                    bbox=field_obj.bbox,
                    reason=f"Tier 1 confidence {field_obj.confidence:.3f} below 0.85 threshold.",
                    reason_ar=f"نسبة الثقة في الحقل الأساسي ({field_obj.confidence:.1%}) أقل من الحد الأدنى المطلوب (85%).",
                )
            )

    # 2. Inspect Tier 2 Contextual Fields
    tier2_fields = doc_schema.get_tier2_fields()
    for field_key, field_obj in tier2_fields.items():
        confidences.append(field_obj.confidence)
        field_name_ar = FIELD_NAMES_AR.get(field_key, field_key)

        if field_obj.obscured or field_obj.confidence < TIER2_CONFIDENCE_THRESHOLD:
            tier2_warnings.append(
                FieldAnomaly(
                    document_type=doc_type_key,
                    document_type_ar=doc_name_ar,
                    field_name=field_key,
                    field_name_ar=field_name_ar,
                    tier=2,
                    confidence=field_obj.confidence,
                    threshold=TIER2_CONFIDENCE_THRESHOLD,
                    value=field_obj.value,
                    obscured=field_obj.obscured,
                    bbox=field_obj.bbox,
                    reason=f"Tier 2 field '{field_key}' confidence below 0.70 threshold.",
                    reason_ar=f"الحقل الثانوي '{field_name_ar}' ثقته ({field_obj.confidence:.1%}) أقل من العتبة (70%).",
                )
            )

    return tier1_anomalies, tier2_warnings, confidences, tier1_confidences


def evaluate_package_triage(
    extraction_package: OnboardingPackageExtractionResult,
    reconciliation_result: CrossDocumentReconciliationResult | None = None,
) -> OnboardingDossier:
    """Evaluate deterministic triage rules and compile the onboarding dossier.

    Implements strict rule precedence:
    1. HARD_MISMATCH: Any cross-document match score < 0.70.
    2. HUMAN_ESCALATION: Any Tier 1 field < 0.85, obscured, or match 0.70-0.88.
    3. AUTO_PASS: All Tier 1 fields >= 0.85 and match >= 0.88.
    Tier 2 warnings (< 0.70) are logged in the audit trail without blocking AUTO_PASS.
    """
    tier1_anomalies: list[FieldAnomaly] = []
    tier2_warnings: list[FieldAnomaly] = []
    all_confidences: list[float] = []
    tier1_confidences: list[float] = []
    audit_trail: list[str] = []
    audit_trail_ar: list[str] = []

    # 1. Scan All Documents
    documents_to_scan = [
        ("national_id", "National ID", extraction_package.national_id),
        ("business_license", "Business License", extraction_package.business_license),
        ("tax_card", "Tax Card", extraction_package.tax_card),
    ]
    for doc_key, doc_label, doc_result in documents_to_scan:
        doc_t1, doc_t2, doc_conf, doc_t1_conf = _inspect_document_fields(doc_key, doc_result.schema)
        tier1_anomalies.extend(doc_t1)
        tier2_warnings.extend(doc_t2)
        all_confidences.extend(doc_conf)
        tier1_confidences.extend(doc_t1_conf)
        if doc_result.audit.unrecoverable_fields:
            doc_name_ar = DOC_TYPES_AR.get(doc_key, doc_key)
            audit_trail.append(f"{doc_label} has unrecoverable fields: {doc_result.audit.unrecoverable_fields}")
            audit_trail_ar.append(f"{doc_name_ar}: حقول غير قابلة للاسترجاع: {doc_result.audit.unrecoverable_fields}")

    # Calculate metrics
    overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    min_tier1_confidence = min(tier1_confidences) if tier1_confidences else 0.0

    # 4. Process Cross-Document Name Matching
    name_mismatches: list[NameMismatchDetail] = []
    has_hard_mismatch = False
    has_matching_escalation = False
    min_matching_score: float | None = None

    if reconciliation_result is not None:
        audit_trail.extend(reconciliation_result.audit_summary)
        audit_trail_ar.extend(reconciliation_result.audit_summary_ar)
        min_matching_score = reconciliation_result.min_similarity_score

        if reconciliation_result.overall_band == MatchBand.HARD_MISMATCH:
            has_hard_mismatch = True
        elif reconciliation_result.overall_band == MatchBand.HUMAN_ESCALATION:
            has_matching_escalation = True

        for pair_key, match_res in reconciliation_result.pairwise_matches.items():
            doc_a, doc_b = pair_key.split("_vs_") if "_vs_" in pair_key else (pair_key, "")
            name_mismatches.append(
                NameMismatchDetail(
                    pair_key=pair_key,
                    doc_a_name=DOC_TYPES_AR.get(doc_a, doc_a),
                    doc_b_name=DOC_TYPES_AR.get(doc_b, doc_b),
                    raw_name_a=match_res.raw_name_a,
                    raw_name_b=match_res.raw_name_b,
                    similarity_score=match_res.similarity_score,
                    triage_band=match_res.triage_band,
                    token_details=list(match_res.token_details),
                    audit_notes=match_res.audit_notes,
                    audit_notes_ar=match_res.audit_notes_ar,
                )
            )

    # 5. Determine Final Lifecycle Outcome
    tier1_passed = len(tier1_anomalies) == 0

    if has_hard_mismatch:
        lifecycle_outcome = TriageLifecycle.HARD_MISMATCH
        actionable_summary_ar = (
            "تم رصد عدم تطابق جوهري في بيانات الهوية بين الوثائق المقدمة (نسبة التطابق أقل من 70%). "
            "يجب رفض الطلب أو إعادة التقديم بوثائق رسمية مطابقة."
        )
    elif not tier1_passed or has_matching_escalation:
        lifecycle_outcome = TriageLifecycle.HUMAN_ESCALATION
        reasons_ar: list[str] = []
        if not tier1_passed:
            failing_names = ", ".join({f"'{a.field_name_ar}' ({a.document_type_ar})" for a in tier1_anomalies})
            reasons_ar.append(f"انخفاض دقة استخراج حقول أساسية: {failing_names}")
        if has_matching_escalation:
            reasons_ar.append("تباين جزئي في الاسم عبر الوثائق (كنية غير رسمية أو نقص في اسم الجد/اللقب)")
        actionable_summary_ar = (
            "يتطلب الملف مراجعة يدوية سريعة للأسباب التالية: "
            + "؛ ".join(reasons_ar)
            + ". يرجى مراجعة الجدول التفصيلي والإحداثيات البصرية أدناه لاتخاذ قرار الاعتماد."
        )
    else:
        lifecycle_outcome = TriageLifecycle.AUTO_PASS
        actionable_summary_ar = (
            "تمت الموافقة التلقائية على الطلب. كافة الحقول الأساسية (Tier 1) مستخرجة بدقة تفوق 85% "
            "وتطابق الأسماء عبر جميع الوثائق يتجاوز عتبة القبول التلقائي (88%)."
        )

    return OnboardingDossier(
        lifecycle_outcome=lifecycle_outcome,
        overall_confidence=round(overall_confidence, 4),
        min_tier1_confidence=round(min_tier1_confidence, 4),
        min_matching_score=round(min_matching_score, 4) if min_matching_score is not None else None,
        tier1_passed=tier1_passed,
        tier1_anomalies=tier1_anomalies,
        tier2_warnings=tier2_warnings,
        cross_matching_result=reconciliation_result,
        name_mismatches=name_mismatches,
        audit_trail=audit_trail,
        audit_trail_ar=audit_trail_ar,
        actionable_summary_ar=actionable_summary_ar,
        extraction_package=extraction_package,
    )


def verify_onboarding_package(
    national_id_image: Image.Image,
    business_license_image: Image.Image,
    tax_card_image: Image.Image,
    extractor: GeminiMultimodalExtractor | None = None,
) -> OnboardingDossier:
    """Execute the complete end-to-end verification and triaging pipeline.

    Coordinates document-parallel extraction with bounded 1x CLAHE retry,
    evaluates cross-document identity consistency (with bipartite fallback on missing NID),
    applies tiered confidence thresholds, and compiles the auditable onboarding decision dossier.
    Guarantees a safe fallback to HUMAN_ESCALATION if any unexpected system exception occurs.
    """
    try:
        active_extractor = extractor or GeminiMultimodalExtractor()

        # 1. Parallel Document Extraction
        extraction_package = active_extractor.extract_onboarding_package_parallel(
            national_id_image=national_id_image,
            business_license_image=business_license_image,
            tax_card_image=tax_card_image,
        )

        # 2. Extract Document Names for Cross-Reconciliation
        nid_schema: NationalIDSchema = extraction_package.national_id.schema  # type: ignore[assignment]
        biz_schema: BusinessLicenseSchema = extraction_package.business_license.schema  # type: ignore[assignment]
        tax_schema: TaxCardSchema = extraction_package.tax_card.schema  # type: ignore[assignment]

        nid_name = nid_schema.full_name.value
        biz_name = biz_schema.full_name.value
        tax_name = tax_schema.taxpayer_name.value

        # 3. Cross-Document Identity Reconciliation with Bipartite Fallback
        reconciliation_result: CrossDocumentReconciliationResult | None = None
        degraded_audit_notes: list[str] = []
        degraded_audit_notes_ar: list[str] = []

        if nid_name and (biz_name or tax_name):
            reconciliation_result = reconcile_onboarding_identities(
                national_card_name=nid_name,
                business_license_name=biz_name,
                tax_card_name=tax_name,
            )
        elif biz_name and tax_name:
            match_biz_tax = match_arabic_names(biz_name, tax_name)
            reconciliation_result = CrossDocumentReconciliationResult(
                overall_band=match_biz_tax.triage_band,
                min_similarity_score=match_biz_tax.similarity_score,
                pairwise_matches={"business_vs_tax": match_biz_tax},
                audit_summary=[
                    "Cross-matching degraded: National ID name missing; evaluated business license vs tax card directly.",
                    *match_biz_tax.audit_notes,
                ],
                audit_summary_ar=[
                    "تدقيق جزئي لمطابقة الأسماء: تعذر قراءة اسم صاحب البطاقة الوطنية؛ تم إجراء مطابقة مباشرة بين إجازة المهنة والهوية الضريبية.",
                    *match_biz_tax.audit_notes_ar,
                ],
            )
        else:
            degraded_audit_notes.append("Cross-matching skipped: Insufficient legible applicant names across documents.")
            degraded_audit_notes_ar.append("تم تخطي مطابقة الأسماء عبر الوثائق لعدم توفر أسماء مقروءة كافية عبر الوثائق.")

        # 4. Evaluate Deterministic Triage Policy
        dossier = evaluate_package_triage(
            extraction_package=extraction_package,
            reconciliation_result=reconciliation_result,
        )

        if degraded_audit_notes:
            dossier = replace(
                dossier,
                audit_trail=[*dossier.audit_trail, *degraded_audit_notes],
                audit_trail_ar=[*dossier.audit_trail_ar, *degraded_audit_notes_ar],
            )

        return dossier

    except Exception:
        logger.exception("Unexpected system fault during onboarding package verification")
        safe_reason = "A technical system fault occurred during automated document extraction and processing."
        safe_reason_ar = "حدث خطأ تقني في مسار المعالجة الآلية للوثائق (تم تسجيل التفاصيل الفنية في سجلات النظام)."
        return OnboardingDossier(
            lifecycle_outcome=TriageLifecycle.HUMAN_ESCALATION,
            overall_confidence=0.0,
            min_tier1_confidence=0.0,
            min_matching_score=None,
            tier1_passed=False,
            tier1_anomalies=[
                FieldAnomaly(
                    document_type="package",
                    document_type_ar=DOC_TYPES_AR["package"],
                    field_name="verification_pipeline",
                    field_name_ar="مسار المعالجة الشامل",
                    tier=1,
                    confidence=0.0,
                    threshold=TIER1_CONFIDENCE_THRESHOLD,
                    value=None,
                    obscured=True,
                    bbox=None,
                    reason=safe_reason,
                    reason_ar=safe_reason_ar,
                )
            ],
            tier2_warnings=[],
            cross_matching_result=None,
            name_mismatches=[],
            audit_trail=[safe_reason],
            audit_trail_ar=[safe_reason_ar],
            actionable_summary_ar="تعذر استكمال المعالجة الآلية بسبب خطأ فني داخلي. تم تسجيل تفاصيل الخطأ للتحقيق الفني وإحالة المعاملة للمراجعة اليدوية المباشرة.",
            extraction_package=None,
        )
