"""Unit tests for deterministic Arabic normalization and patronymic cross-matching.

Verifies:
1. Six-stage orthographic normalization pipeline.
2. Compound name spacing unification (Abdullah, Amatullah, Din, Abu, Umm).
3. Article-debiased stem similarity on tribal titles.
4. Auto-Pass (>= 0.88) on legal Iraqi patronymic variations and compound given names.
5. Human Escalation (0.70 <= score < 0.88) on kunyas (Abu/Umm), 2-part names, and omitted/conflicting surnames.
6. Guard against false-positive kunya popping on genuine Alif-Meem given names (Amal, Amina).
7. Hard Mismatch (< 0.70) on non-matching identities and mismatched 2-part names.
8. Extended patronymic chain audit notes (> 4 tokens).
9. Tripartite cross-document onboarding reconciliation.
"""

import pytest
from src.matching.arabic_matcher import (
    MatchBand,
    compute_token_similarity,
    is_kunya_token,
    jaro_winkler_similarity,
    match_arabic_names,
    normalize_arabic_name,
    reconcile_onboarding_identities,
    strip_definite_article_token,
    strip_kunya_marker,
    strip_tashkeel,
    standardize_hamza,
    unify_compound_names,
    unify_taa_marbuta,
    unify_yaa,
)


# ---------------------------------------------------------------------------
# Normalization Pipeline Tests
# ---------------------------------------------------------------------------

def test_stage1_strip_tashkeel_and_tatweel():
    """Verify diacritics, shadda, and tatweel are completely removed."""
    raw = "مُحَمَّدٌ عَـبْدُ اللَّهِ"
    assert strip_tashkeel(raw) == "محمد عبد الله"


def test_stage2_standardize_hamza():
    """Verify Hamza variants (إ, أ, آ, ء) unify to bare Alif (ا)."""
    assert standardize_hamza("إبراهيم") == "ابراهيم"
    assert standardize_hamza("أحمد") == "احمد"
    assert standardize_hamza("آلاء") == "الاا"
    assert standardize_hamza("بهاء") == "بهاا"


def test_stage3_unify_yaa_and_alif_maqsura():
    """Verify Alif Maqsura (ى) and Persian Yeh unify to Arabic Yaa (ي)."""
    assert unify_yaa("مصطفى") == "مصطفي"
    assert unify_yaa("مهدى") == "مهدي"
    assert unify_yaa("علي\u06CC") == "عليي"


def test_stage4_unify_taa_marbuta():
    """Verify Taa Marbuta (ة) unifies to Haa (ه)."""
    assert unify_taa_marbuta("حمزة") == "حمزه"
    assert unify_taa_marbuta("فاطمة") == "فاطمه"


def test_stage6_unify_compound_names():
    """Verify compound names (عبد, أمة, الدين, الله, ابو, ام) unify without spaces."""
    assert unify_compound_names("عبد الله") == "عبدالله"
    assert unify_compound_names("أمة الله") == "امهالله"
    assert unify_compound_names("أمة الرحمن") == "امهالرحمن"
    assert unify_compound_names("عبد الرحمن") == "عبدالرحمن"
    assert unify_compound_names("عبد العزيز") == "عبدالعزيز"
    assert unify_compound_names("نور الدين") == "نورالدين"
    assert unify_compound_names("صلاح الدين") == "صلاحالدين"
    assert unify_compound_names("فضل الله") == "فضلالله"
    assert unify_compound_names("ابو بكر") == "ابوبكر"
    assert unify_compound_names("أم كلثوم") == "أمكلثوم"


def test_full_normalization_pipeline():
    """Verify complete 6-stage pipeline across heavily decorated Arabic strings."""
    input_text = "  مُحَمَّدٌ   عَبْدُ اللَّهِ   كَـاظِمٌ   الزُّبَيْدِيُّ  "
    expected = "محمد عبدالله كاظم الزبيدي"
    assert normalize_arabic_name(input_text) == expected


# ---------------------------------------------------------------------------
# Article De-biasing & Stem Similarity Tests
# ---------------------------------------------------------------------------

def test_article_stem_stripping():
    """Verify 'ال' is stripped on valid roots (>= 5 chars) but preserved on short words."""
    assert strip_definite_article_token("الزبيدي") == "زبيدي"
    assert strip_definite_article_token("الخفاجي") == "خفاجي"
    assert strip_definite_article_token("الم") == "الم"


def test_article_debiased_token_similarity():
    """Verify article-stripping prevents inflated Jaro-Winkler scores on distinct tribes."""
    assert compute_token_similarity("الزبيدي", "زبيدي") == 1.0
    assert compute_token_similarity("الزبيدي", "الزبيدي") == 1.0

    stem_sim = compute_token_similarity("الزبيدي", "الجبوري")
    assert stem_sim == pytest.approx(0.600, abs=0.01)
    assert stem_sim < 0.70


def test_kunya_guard_and_marker_stripping():
    """Verify is_kunya_token and strip_kunya_marker guard real Alif-Meem given names."""
    assert not is_kunya_token("امل")
    assert not is_kunya_token("امينه")
    assert not is_kunya_token("اميره")
    assert not is_kunya_token("امجد")
    assert strip_kunya_marker("امل") == "امل"
    assert strip_kunya_marker("امينه") == "امينه"

    assert is_kunya_token("ابوفهد")
    assert is_kunya_token("امفهد")
    assert is_kunya_token("ابو")
    assert is_kunya_token("ام")
    assert strip_kunya_marker("ابوفهد") == "فهد"
    assert strip_kunya_marker("امفهد") == "فهد"


# ---------------------------------------------------------------------------
# Triage Band 1: Auto-Pass (>= 0.88)
# ---------------------------------------------------------------------------

def test_autopass_exact_four_part_match():
    """Identical 4-part patronymic name achieves 1.000 Auto-Pass."""
    name_a = "محمد عبد الله كاظم الزبيدي"
    name_b = "محمد عبدالله كاظم الزبيدي"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score == 1.000
    assert result.triage_band == MatchBand.AUTO_PASS


def test_autopass_omitted_tribal_surname():
    """4-part National Card matching 3-part Business License (omitted surname) achieves Auto-Pass."""
    name_card = "محمد عبد الله كاظم الزبيدي"
    name_license = "محمد عبدالله كاظم"
    result = match_arabic_names(name_card, name_license)
    assert result.similarity_score >= 0.88
    assert result.similarity_score == pytest.approx(0.900, abs=0.01)
    assert result.triage_band == MatchBand.AUTO_PASS
    assert any("omitted" in note.lower() for note in result.audit_notes)


def test_autopass_female_theophoric_name_amat_allah():
    """Female theophoric compound name (Amatullah) unifies spaces and auto-passes."""
    name_card = "أمة الله كاظم جاسم الزبيدي"
    name_license = "امةالله كاظم جاسم"
    result = match_arabic_names(name_card, name_license)
    assert result.similarity_score >= 0.88
    assert result.similarity_score == pytest.approx(0.900, abs=0.01)
    assert result.triage_band == MatchBand.AUTO_PASS


def test_autopass_with_orthographic_variations():
    """Tashkeel, Hamza, Alif Maqsura, and Taa Marbuta differences cleanly Auto-Pass."""
    name_a = "أحمد عبد الرزاق مهدى حمزة التميمي"
    name_b = "احمد عبدالرزاق مهدي حمزه التميمي"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score == 1.000
    assert result.triage_band == MatchBand.AUTO_PASS


def test_autopass_three_part_names():
    """Both documents with matching 3-part names cleanly Auto-Pass."""
    name_a = "علي جاسم كريم"
    name_b = "علي جاسم كريم"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score == 1.000
    assert result.triage_band == MatchBand.AUTO_PASS


def test_autopass_compound_given_names_abu_and_umm():
    """Compound given names (Abu Bakr, Umm Kulthum) match cleanly without false kunya-popping."""
    res_abu = match_arabic_names("أبو بكر جاسم كريم", "ابوبكر جاسم كريم")
    assert res_abu.similarity_score == 1.000
    assert res_abu.triage_band == MatchBand.AUTO_PASS

    res_umm = match_arabic_names("أم كلثوم جاسم كريم", "ام كلثوم جاسم كريم")
    assert res_umm.similarity_score == 1.000
    assert res_umm.triage_band == MatchBand.AUTO_PASS


def test_autopass_female_name_starting_with_alif_meem():
    """Female names starting with Alif-Meem (e.g. Amal) auto-pass when matching."""
    res = match_arabic_names("أمل محمد عبدالله", "امل محمد عبدالله")
    assert res.similarity_score == 1.000
    assert res.triage_band == MatchBand.AUTO_PASS


# ---------------------------------------------------------------------------
# Triage Band 2: Human Escalation (0.70 <= score < 0.88)
# ---------------------------------------------------------------------------

def test_escalation_matching_two_part_names():
    """Identical 2-part names renormalize and cap at 0.850 for Human Escalation."""
    name_a = "محمد جاسم"
    name_b = "محمد جاسم"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score == 0.850
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("insufficient patronymic chain length" in note.lower() for note in result.audit_notes)


def test_escalation_partially_matching_two_part_names():
    """2-part name with exact given name and minor father typo escalates without being deflated."""
    result = match_arabic_names("محمد جاسم", "محمد ياسم")
    assert 0.70 <= result.similarity_score < 0.88
    assert result.triage_band == MatchBand.HUMAN_ESCALATION


def test_escalation_omitted_grandfather():
    """Given, Father, and Surname match, but Grandfather is omitted -> Escalation (0.750)."""
    name_a = "محمد عبد الله كاظم الزبيدي"
    name_b = "محمد عبدالله الزبيدي"
    result = match_arabic_names(name_a, name_b)
    assert 0.70 <= result.similarity_score < 0.88
    assert result.similarity_score == pytest.approx(0.750, abs=0.01)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("grandfather" in note.lower() for note in result.audit_notes)


def test_escalation_conflicting_tribal_surnames():
    """Core patronymics match, but explicit tribal titles conflict -> Escalation (0.820)."""
    name_a = "محمد عبد الله كاظم الزبيدي"
    name_b = "محمد عبدالله كاظم الجبوري"
    result = match_arabic_names(name_a, name_b)
    assert 0.70 <= result.similarity_score < 0.88
    assert result.similarity_score == pytest.approx(0.820, abs=0.01)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("conflicting" in note.lower() for note in result.audit_notes)


def test_escalation_informal_male_teknonym_prefix():
    """Informal male kunya ('أبو فهد') prefixed to full legal chain -> Escalation (0.850)."""
    name_a = "أبو فهد محمد عبدالله كاظم"
    name_b = "محمد عبدالله كاظم الزبيدي"
    result = match_arabic_names(name_a, name_b)
    assert 0.70 <= result.similarity_score < 0.88
    assert result.similarity_score == pytest.approx(0.850, abs=0.01)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("teknonym" in note.lower() for note in result.audit_notes)


def test_escalation_informal_female_teknonym_prefix():
    """Informal female kunya ('أم فهد') prefixed to full legal chain -> Escalation (0.850)."""
    name_a = "أم فهد فاطمة حسن كريم"
    name_b = "فاطمة حسن كريم"
    result = match_arabic_names(name_a, name_b)
    assert 0.70 <= result.similarity_score < 0.88
    assert result.similarity_score == pytest.approx(0.850, abs=0.01)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("teknonym" in note.lower() for note in result.audit_notes)


def test_escalation_kunya_as_operative_given_name():
    """Kunya used interchangeably with base name in Given slot ('أبو فهد' vs 'فهد') -> Escalation (0.850)."""
    name_a = "أبو فهد جاسم كريم"
    name_b = "فهد جاسم كريم"
    result = match_arabic_names(name_a, name_b)
    assert 0.70 <= result.similarity_score < 0.88
    assert result.similarity_score == pytest.approx(0.850, abs=0.01)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert any("kunya variation" in note.lower() for note in result.audit_notes)


def test_escalation_ambiguous_alignment():
    """Near-equal similarity of 3rd token to Grandfather and Surname triggers escalation with note."""
    # When 3rd token matches both Grandfather and Surname slots identically in a 4-part name
    name_a = "محمد عبدالله كريم الزبيدي"
    # Secondary document has 3-part name where 3rd token has identical distance to both
    name_b = "محمد عبدالله كريم كريم"  # 4 tokens
    # Alternatively: 4 tokens vs 3 tokens
    # Long: "محمد عبدالله كاظم كاظم" (grandfather and surname are identical or tied)
    name_long = "محمد عبدالله كاظم كاظم"
    name_short = "محمد عبدالله كاظم"
    result = match_arabic_names(name_long, name_short)
    assert result.triage_band == MatchBand.HUMAN_ESCALATION
    assert result.similarity_score <= 0.85
    assert any("ambiguous alignment" in note.lower() for note in result.audit_notes)


# ---------------------------------------------------------------------------
# Triage Band 3: Hard Mismatch (< 0.70)
# ---------------------------------------------------------------------------

def test_hard_mismatch_two_part_names():
    """Mismatched 2-part names fail closed to Hard Mismatch."""
    result = match_arabic_names("محمد جاسم", "علي كريم")
    assert result.similarity_score < 0.70
    assert result.triage_band == MatchBand.HARD_MISMATCH


def test_hard_mismatch_female_name_starting_with_alif_meem():
    """Amal ('أمل') must NOT be treated as a kunya prefix and popped when compared to Muhammad."""
    name_a = "أمل محمد عبدالله"
    name_b = "محمد عبدالله كاظم"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score < 0.70
    assert result.triage_band == MatchBand.HARD_MISMATCH
    assert "teknonym" not in " ".join(result.audit_notes).lower()


def test_hard_mismatch_amina_vs_muhammad():
    """Amina ('أمينة') must NOT be popped as a kunya prefix."""
    name_a = "أمينة محمد عبدالله"
    name_b = "محمد عبدالله كاظم"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score < 0.70
    assert result.triage_band == MatchBand.HARD_MISMATCH


def test_hard_mismatch_completely_different_person():
    """Completely different applicant names yield Hard Mismatch."""
    name_a = "محمد عبد الله كاظم الزبيدي"
    name_b = "علي حسن جاسم الخفاجي"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score < 0.70
    assert result.triage_band == MatchBand.HARD_MISMATCH


def test_hard_mismatch_same_given_name_different_family():
    """Same Given name but different Father and Grandfather yields Hard Mismatch."""
    name_a = "محمد عبد الله كاظم الزبيدي"
    name_b = "محمد جاسم صادق التميمي"
    result = match_arabic_names(name_a, name_b)
    assert result.similarity_score < 0.70
    assert result.similarity_score == pytest.approx(0.674, abs=0.01)
    assert result.triage_band == MatchBand.HARD_MISMATCH


def test_hard_mismatch_empty_strings():
    """Empty strings fail closed to Hard Mismatch with 0.0 score."""
    result = match_arabic_names("", "محمد عبدالله")
    assert result.similarity_score == 0.0
    assert result.triage_band == MatchBand.HARD_MISMATCH


# ---------------------------------------------------------------------------
# Extended Chains & Tripartite Cross-Document Reconciliation Tests
# ---------------------------------------------------------------------------

def test_extended_patronymic_chain_note():
    """Names with > 4 tokens log an audit warning noting extended patronymic evaluation."""
    name_a = "علي جاسم كريم علي الزبيدي"  # 5 tokens
    name_b = "علي جاسم كريم الزبيدي"      # 4 tokens
    result = match_arabic_names(name_a, name_b)
    assert any("extended patronymic chain" in note.lower() for note in result.audit_notes)


def test_cross_document_reconciliation_all_pass():
    """All 3 documents matching yields overall AUTO_PASS."""
    id_name = "محمد عبد الله كاظم الزبيدي"
    license_name = "محمد عبدالله كاظم"
    tax_name = "محمد عبد الله كاظم الزبيدي"

    result = reconcile_onboarding_identities(id_name, license_name, tax_name)
    assert result.overall_band == MatchBand.AUTO_PASS
    assert result.min_similarity_score >= 0.88
    assert len(result.pairwise_matches) == 3


def test_cross_document_reconciliation_one_escalated():
    """One document with missing grandfather escalates the entire application."""
    id_name = "محمد عبد الله كاظم الزبيدي"
    license_name = "محمد عبدالله الزبيدي"  # Omitted grandfather -> 0.750
    tax_name = "محمد عبد الله كاظم الزبيدي"

    result = reconcile_onboarding_identities(id_name, license_name, tax_name)
    assert result.overall_band == MatchBand.HUMAN_ESCALATION
    assert result.min_similarity_score == pytest.approx(0.750, abs=0.01)


def test_cross_document_reconciliation_hard_mismatch():
    """One mismatched document triggers overall HARD_MISMATCH."""
    id_name = "محمد عبد الله كاظم الزبيدي"
    license_name = "محمد عبدالله كاظم"
    tax_name = "عمر طارق سلمان الدليمي"  # Fraud or wrong card uploaded

    result = reconcile_onboarding_identities(id_name, license_name, tax_name)
    assert result.overall_band == MatchBand.HARD_MISMATCH
    assert result.min_similarity_score < 0.70
