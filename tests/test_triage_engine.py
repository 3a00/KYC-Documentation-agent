"""Comprehensive test suite for the deterministic triage engine and exception dossier.

Verifies Tier 1 hard escalation policies, Tier 2 audit-warning non-blocking policies,
cross-document name reconciliation integration with bipartite fallback, fail-safe
exception seam defense, 3-way lifecycle outcomes, and patronymic slot diff rendering.
"""

from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from src.extraction.gemini_extractor import (
    DocumentExtractionResult,
    DocumentType,
    ExtractionAuditTrail,
    OnboardingPackageExtractionResult,
    RetryStatus,
)
from src.matching.arabic_matcher import (
    CrossDocumentReconciliationResult,
    IdentityMatchResult,
    MatchBand,
    PatronymicRole,
    TokenMatchDetail,
)
from src.models.schemas import (
    BusinessLicenseSchema,
    ExtractedField,
    NationalIDSchema,
    TaxCardSchema,
)
from src.triage.triage_engine import (
    FieldAnomaly,
    OnboardingDossier,
    TriageLifecycle,
    evaluate_package_triage,
    verify_onboarding_package,
)


def _make_field(
    val: str | None,
    conf: float,
    obscured: bool = False,
    bbox: list[int] | None = None,
) -> ExtractedField[str]:
    """Helper to construct ExtractedField preserving biconditional zero-hallucination."""
    return ExtractedField[str](
        value=None if obscured else val,
        confidence=conf,
        obscured=obscured,
        bbox=bbox or [100, 100, 200, 400],
    )


def _make_clean_national_id() -> NationalIDSchema:
    return NationalIDSchema(
        national_id=_make_field("199012345678", 0.95),
        full_name=_make_field("محمد جاسم كريم الزبيدي", 0.96),
        mother_name=_make_field("فاطمة حسن علي", 0.92),
        expiry_date=_make_field("2030-01-01", 0.94),
        issue_date=_make_field("2020-01-01", 0.91),
        province=_make_field("بغداد", 0.93),
    )


def _make_clean_business_license() -> BusinessLicenseSchema:
    return BusinessLicenseSchema(
        business_name=_make_field("شركة الرافدين للتجارة العامة", 0.94),
        full_name=_make_field("محمد جاسم كريم الزبيدي", 0.95),
        license_number=_make_field("BL-987654", 0.93),
        issue_date=_make_field("2021-05-15", 0.89),
        business_activity=_make_field("تجارة عامة", 0.90),
        province=_make_field("بغداد", 0.92),
    )


def _make_clean_tax_card() -> TaxCardSchema:
    return TaxCardSchema(
        tax_id=_make_field("TAX-44556677", 0.96),
        taxpayer_name=_make_field("محمد جاسم كريم الزبيدي", 0.95),
        fiscal_year=_make_field("2023", 0.92),
        issue_date=_make_field("2022-03-10", 0.91),
    )


def _make_package_result(
    nid: NationalIDSchema,
    biz: BusinessLicenseSchema,
    tax: TaxCardSchema,
) -> OnboardingPackageExtractionResult:
    return OnboardingPackageExtractionResult(
        national_id=DocumentExtractionResult(
            schema=nid,
            audit=ExtractionAuditTrail(
                document_type=DocumentType.NATIONAL_ID,
                retry_status=RetryStatus.NOT_NEEDED,
            ),
        ),
        business_license=DocumentExtractionResult(
            schema=biz,
            audit=ExtractionAuditTrail(
                document_type=DocumentType.BUSINESS_LICENSE,
                retry_status=RetryStatus.NOT_NEEDED,
            ),
        ),
        tax_card=DocumentExtractionResult(
            schema=tax,
            audit=ExtractionAuditTrail(
                document_type=DocumentType.TAX_CARD,
                retry_status=RetryStatus.NOT_NEEDED,
            ),
        ),
        total_execution_time_seconds=1.25,
    )


def _make_reconciliation_result(
    band: MatchBand,
    score: float = 0.98,
    slots: list[TokenMatchDetail] | None = None,
) -> CrossDocumentReconciliationResult:
    default_slots = [
        TokenMatchDetail(role=PatronymicRole.GIVEN, token_a="محمد", token_b="محمد", similarity=1.0, weight=0.35, weighted_score=0.35),
        TokenMatchDetail(role=PatronymicRole.FATHER, token_a="جاسم", token_b="جاسم", similarity=1.0, weight=0.30, weighted_score=0.30),
        TokenMatchDetail(role=PatronymicRole.GRANDFATHER, token_a="كريم", token_b="كريم", similarity=1.0, weight=0.25, weighted_score=0.25),
        TokenMatchDetail(role=PatronymicRole.SURNAME, token_a="الزبيدي", token_b="الزبيدي", similarity=1.0, weight=0.10, weighted_score=0.10),
    ]
    match_res = IdentityMatchResult(
        raw_name_a="محمد جاسم كريم الزبيدي",
        raw_name_b="محمد جاسم كريم الزبيدي",
        normalized_name_a="محمد جاسم كريم الزبيدي",
        normalized_name_b="محمد جاسم كريم الزبيدي",
        similarity_score=score,
        triage_band=band,
        token_details=slots or default_slots,
        audit_notes=["Match validated."],
        audit_notes_ar=["تمت المطابقة بنجاح."],
    )
    return CrossDocumentReconciliationResult(
        overall_band=band,
        min_similarity_score=score,
        pairwise_matches={"national_vs_business": match_res, "national_vs_tax": match_res},
        audit_summary=["All identity checks passed."],
        audit_summary_ar=["تمت مطابقة الهوية بنجاح."],
    )


class TestDeterministicTriageEngine:
    """Test suite validating deterministic triage policy decisions and dossier output."""

    def test_clean_onboarding_package_yields_auto_pass(self) -> None:
        """When all Tier 1 fields >= 0.85 and matching >= 0.88, result must be AUTO_PASS."""
        pkg = _make_package_result(_make_clean_national_id(), _make_clean_business_license(), _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.98)

        dossier = evaluate_package_triage(pkg, recon)

        assert dossier.lifecycle_outcome == TriageLifecycle.AUTO_PASS
        assert dossier.tier1_passed is True
        assert len(dossier.tier1_anomalies) == 0
        assert len(dossier.tier2_warnings) == 0
        assert dossier.overall_confidence > 0.90
        assert dossier.min_tier1_confidence >= 0.85
        assert dossier.min_matching_score == 0.98
        assert "الموافقة التلقائية" in dossier.actionable_summary_ar

    def test_auto_pass_with_non_uniform_slots_does_not_flag_impactful_discrepancy(self) -> None:
        """Non-uniform slots in AUTO_PASS must NOT trigger impactful discrepancy bolding or driver callout."""
        pkg = _make_package_result(_make_clean_national_id(), _make_clean_business_license(), _make_clean_tax_card())
        non_uniform_slots = [
            TokenMatchDetail(role=PatronymicRole.GIVEN, token_a="محمد", token_b="محمد", similarity=1.0, weight=0.35, weighted_score=0.35),
            TokenMatchDetail(role=PatronymicRole.FATHER, token_a="جاسم", token_b="جاسم", similarity=1.0, weight=0.30, weighted_score=0.30),
            TokenMatchDetail(role=PatronymicRole.GRANDFATHER, token_a="كريم", token_b="كامل", similarity=0.75, weight=0.25, weighted_score=0.1875),
            TokenMatchDetail(role=PatronymicRole.SURNAME, token_a="الزبيدي", token_b="الزبيدي", similarity=1.0, weight=0.10, weighted_score=0.10),
        ]
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.9375, slots=non_uniform_slots)

        dossier = evaluate_package_triage(pkg, recon)
        report_md = dossier.to_markdown_report()

        assert dossier.lifecycle_outcome == TriageLifecycle.AUTO_PASS
        assert "**تباين مؤثر**" not in report_md
        assert "السبب الأبرز للتباين" not in report_md
        assert "مقبول ضمن النطاق الكلي" in report_md

    def test_tier1_confidence_drop_triggers_human_escalation(self) -> None:
        """Any Tier 1 field below 0.85 (e.g. National ID at 0.81) triggers HUMAN_ESCALATION."""
        nid = _make_clean_national_id()
        nid.national_id = _make_field("199012345678", 0.81, bbox=[120, 50, 160, 300])

        pkg = _make_package_result(nid, _make_clean_business_license(), _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.95)

        dossier = evaluate_package_triage(pkg, recon)

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert dossier.tier1_passed is False
        assert len(dossier.tier1_anomalies) == 1
        assert dossier.tier1_anomalies[0].field_name == "national_id"
        assert dossier.tier1_anomalies[0].field_name_ar == "رقم البطاقة الوطنية"
        assert dossier.tier1_anomalies[0].bbox == [120, 50, 160, 300]
        assert dossier.min_tier1_confidence == 0.81
        assert "رقم البطاقة الوطنية" in dossier.actionable_summary_ar

    def test_tier1_obscured_field_triggers_human_escalation(self) -> None:
        """Tier 1 field marked obscured (value=None) triggers HUMAN_ESCALATION with bbox."""
        biz = _make_clean_business_license()
        biz.license_number = _make_field(None, 0.0, obscured=True, bbox=[200, 150, 240, 450])

        pkg = _make_package_result(_make_clean_national_id(), biz, _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.95)

        dossier = evaluate_package_triage(pkg, recon)

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert dossier.tier1_passed is False
        assert len(dossier.tier1_anomalies) == 1
        assert dossier.tier1_anomalies[0].field_name == "license_number"
        assert dossier.tier1_anomalies[0].obscured is True

    def test_tier2_confidence_drop_does_not_block_auto_pass(self) -> None:
        """Tier 2 field (mother_name) dropping below 0.70 emits audit warning but AUTO_PASSes."""
        nid = _make_clean_national_id()
        nid.mother_name = _make_field("فاطمة حسن علي", 0.62)

        pkg = _make_package_result(nid, _make_clean_business_license(), _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.96)

        dossier = evaluate_package_triage(pkg, recon)

        assert dossier.lifecycle_outcome == TriageLifecycle.AUTO_PASS
        assert dossier.tier1_passed is True
        assert len(dossier.tier1_anomalies) == 0
        assert len(dossier.tier2_warnings) == 1
        assert dossier.tier2_warnings[0].field_name == "mother_name"
        assert dossier.tier2_warnings[0].field_name_ar == "اسم الأم الثلاثي"

    def test_cross_document_matching_escalation_triggers_human_escalation(self) -> None:
        """When names match with HUMAN_ESCALATION band (0.70 <= score < 0.88), dossier escalates."""
        pkg = _make_package_result(_make_clean_national_id(), _make_clean_business_license(), _make_clean_tax_card())
        escalation_slots = [
            TokenMatchDetail(role=PatronymicRole.GIVEN, token_a="محمد", token_b="محمد", similarity=1.0, weight=0.35, weighted_score=0.35),
            TokenMatchDetail(role=PatronymicRole.FATHER, token_a="جاسم", token_b="جاسم", similarity=1.0, weight=0.30, weighted_score=0.30),
            TokenMatchDetail(role=PatronymicRole.GRANDFATHER, token_a="كريم", token_b="خضير", similarity=0.40, weight=0.25, weighted_score=0.10),
            TokenMatchDetail(role=PatronymicRole.SURNAME, token_a="الزبيدي", token_b="الزبيدي", similarity=1.0, weight=0.10, weighted_score=0.10),
        ]
        recon = _make_reconciliation_result(MatchBand.HUMAN_ESCALATION, score=0.85, slots=escalation_slots)

        dossier = evaluate_package_triage(pkg, recon)
        report_md = dossier.to_markdown_report()

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert len(dossier.name_mismatches) == 2
        assert dossier.min_matching_score == 0.85
        assert "تباين جزئي في الاسم" in dossier.actionable_summary_ar
        assert "**تباين مؤثر**" in report_md
        assert "السبب الأبرز للتباين" in report_md

    def test_cross_document_matching_hard_mismatch_overrides_to_hard_mismatch(self) -> None:
        """Score < 0.70 triggers HARD_MISMATCH, taking precedence over other states."""
        pkg = _make_package_result(_make_clean_national_id(), _make_clean_business_license(), _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.HARD_MISMATCH, score=0.45)

        dossier = evaluate_package_triage(pkg, recon)

        assert dossier.lifecycle_outcome == TriageLifecycle.HARD_MISMATCH
        assert dossier.min_matching_score == 0.45
        assert "عدم تطابق جوهري" in dossier.actionable_summary_ar

    def test_missing_nid_name_triggers_bipartite_biz_vs_tax_reconciliation(self) -> None:
        """When NID name is obscured/null, verify seam reconciles biz vs tax directly."""
        mock_extractor = MagicMock()
        nid = _make_clean_national_id()
        nid.full_name = _make_field(None, 0.0, obscured=True)
        biz = _make_clean_business_license()
        tax = _make_clean_tax_card()
        pkg = _make_package_result(nid, biz, tax)
        mock_extractor.extract_onboarding_package_parallel.return_value = pkg

        dummy_img = Image.new("RGB", (100, 100), color="white")
        dossier = verify_onboarding_package(dummy_img, dummy_img, dummy_img, extractor=mock_extractor)

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert dossier.tier1_passed is False
        assert len(dossier.name_mismatches) == 1
        assert dossier.name_mismatches[0].pair_key == "business_vs_tax"
        assert any("تدقيق جزئي" in note for note in dossier.audit_trail_ar)

    def test_bipartite_fallback_can_never_reach_auto_pass(self) -> None:
        """Explicit invariance test: degraded bipartite path can NEVER reach AUTO_PASS."""
        mock_extractor = MagicMock()
        nid = _make_clean_national_id()
        nid.full_name = _make_field(None, 0.0, obscured=True)
        biz = _make_clean_business_license()
        tax = _make_clean_tax_card()
        pkg = _make_package_result(nid, biz, tax)
        mock_extractor.extract_onboarding_package_parallel.return_value = pkg

        dummy_img = Image.new("RGB", (100, 100), color="white")
        dossier = verify_onboarding_package(dummy_img, dummy_img, dummy_img, extractor=mock_extractor)

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert dossier.tier1_passed is False
        assert dossier.lifecycle_outcome != TriageLifecycle.AUTO_PASS

    def test_missing_nid_name_and_biz_tax_hard_mismatch_yields_hard_mismatch(self) -> None:
        """Missing NID name + severe Biz/Tax discrepancy (<0.70) must yield HARD_MISMATCH."""
        mock_extractor = MagicMock()
        nid = _make_clean_national_id()
        nid.full_name = _make_field(None, 0.0, obscured=True)
        biz = _make_clean_business_license()
        biz.full_name = _make_field("علي حسن خضير الكعبي", 0.95)
        tax = _make_clean_tax_card()
        tax.taxpayer_name = _make_field("مصطفى طارق سلمان الدوري", 0.95)
        pkg = _make_package_result(nid, biz, tax)
        mock_extractor.extract_onboarding_package_parallel.return_value = pkg

        dummy_img = Image.new("RGB", (100, 100), color="white")
        dossier = verify_onboarding_package(dummy_img, dummy_img, dummy_img, extractor=mock_extractor)

        assert dossier.lifecycle_outcome == TriageLifecycle.HARD_MISMATCH
        assert dossier.min_matching_score is not None
        assert dossier.min_matching_score < 0.70

    def test_missing_all_names_logs_explicit_degraded_audit_trail(self) -> None:
        """When all name fields are obscured, verify seam logs explicit degradation notice."""
        mock_extractor = MagicMock()
        nid = _make_clean_national_id()
        nid.full_name = _make_field(None, 0.0, obscured=True)
        biz = _make_clean_business_license()
        biz.full_name = _make_field(None, 0.0, obscured=True)
        tax = _make_clean_tax_card()
        tax.taxpayer_name = _make_field(None, 0.0, obscured=True)
        pkg = _make_package_result(nid, biz, tax)
        mock_extractor.extract_onboarding_package_parallel.return_value = pkg

        dummy_img = Image.new("RGB", (100, 100), color="white")
        dossier = verify_onboarding_package(dummy_img, dummy_img, dummy_img, extractor=mock_extractor)

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert any("تم تخطي مطابقة الأسماء" in note for note in dossier.audit_trail_ar)

    def test_field_anomaly_bbox_geometry_validation(self) -> None:
        """FieldAnomaly raises ValueError when bbox geometry or coordinates are out of bounds."""
        with pytest.raises(ValueError, match="within \\[0, 1000\\]"):
            FieldAnomaly(
                document_type="national_id",
                document_type_ar="البطاقة الوطنية",
                field_name="test",
                field_name_ar="تجربة",
                tier=1,
                confidence=0.5,
                threshold=0.85,
                value=None,
                obscured=True,
                bbox=[100, 200, 1200, 400],
                reason="out of range",
                reason_ar="خارج النطاق",
            )
        with pytest.raises(ValueError, match="Invalid coordinate geometry"):
            FieldAnomaly(
                document_type="national_id",
                document_type_ar="البطاقة الوطنية",
                field_name="test",
                field_name_ar="تجربة",
                tier=1,
                confidence=0.5,
                threshold=0.85,
                value=None,
                obscured=True,
                bbox=[500, 200, 100, 400],
                reason="ymin > ymax",
                reason_ar="خطأ في الإحداثيات",
            )

    def test_verify_onboarding_package_seam_handles_extractor_exception_safely(self) -> None:
        """Top-level seam must catch unhandled extractor exceptions and safely escalate."""
        mock_extractor = MagicMock()
        mock_extractor.extract_onboarding_package_parallel.side_effect = RuntimeError("Upstream API token secret timeout")

        dummy_img = Image.new("RGB", (100, 100), color="white")
        with patch("logging.Logger.exception") as mock_log:
            dossier = verify_onboarding_package(dummy_img, dummy_img, dummy_img, extractor=mock_extractor)
            mock_log.assert_called_once()

        assert dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION
        assert dossier.tier1_passed is False
        assert "Upstream API token secret timeout" not in dossier.tier1_anomalies[0].reason
        assert "technical system fault occurred" in dossier.tier1_anomalies[0].reason
        assert "خطأ فني داخلي" in dossier.actionable_summary_ar

    def test_markdown_report_formatting_with_patronymic_slot_diff_and_arabic_pair_labels(self) -> None:
        """Report generator outputs valid Arabic Markdown containing decision headers and tables."""
        nid = _make_clean_national_id()
        pkg = _make_package_result(nid, _make_clean_business_license(), _make_clean_tax_card())
        recon = _make_reconciliation_result(MatchBand.AUTO_PASS, score=0.98)
        dossier = evaluate_package_triage(pkg, recon)
        report_md = dossier.to_markdown_report()

        assert "# تقرير تدقيق التحقق من وثائق الانضمام" in report_md
        assert "البطاقة الوطنية الموحدة مقابل إجازة ممارسة المهنة / السجل التجاري" in report_md
        assert "business مقابل tax" not in report_md
        assert "national مقابل business" not in report_md
        assert "جدول الفروقات التفصيلي لسلاسل النسب" in report_md
        assert "الاسم الأول (Given)" in report_md
        assert "أدنى ثقة في الحقول الأساسية" in report_md
        assert "أدنى نسبة تطابق بين الأسماء" in report_md
