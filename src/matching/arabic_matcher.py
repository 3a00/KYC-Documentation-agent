"""Deterministic Arabic normalizer and patronymic cross-document identity matcher.

Provides an auditable 6-stage orthographic normalization pipeline, article-debiased
pure-Python Jaro-Winkler similarity, alignment-aware kunya handling with Alif-Meem
name protection, semantic patronymic token alignment, and deterministic triage
classification across Iraqi onboarding documents.
"""

from enum import Enum
import re
from pydantic import BaseModel, ConfigDict


class MatchBand(str, Enum):
    """Deterministic triage classification bands for identity matching."""

    AUTO_PASS = "AUTO_PASS"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
    HARD_MISMATCH = "HARD_MISMATCH"


class PatronymicRole(str, Enum):
    """Semantic slot roles in an Iraqi patronymic name chain."""

    GIVEN = "given"
    FATHER = "father"
    GRANDFATHER = "grandfather"
    SURNAME = "surname"


class TokenMatchDetail(BaseModel):
    """Detailed similarity score and weight for an individual patronymic token slot."""

    model_config = ConfigDict(frozen=True)

    role: PatronymicRole
    token_a: str | None
    token_b: str | None
    similarity: float
    weight: float
    weighted_score: float


class IdentityMatchResult(BaseModel):
    """Comprehensive comparison result between two Arabic name strings."""

    model_config = ConfigDict(frozen=True)

    raw_name_a: str
    raw_name_b: str
    normalized_name_a: str
    normalized_name_b: str
    similarity_score: float
    triage_band: MatchBand
    token_details: list[TokenMatchDetail]
    audit_notes: list[str]
    audit_notes_ar: list[str]


class CrossDocumentReconciliationResult(BaseModel):
    """Consolidated identity reconciliation across all provided onboarding documents."""

    model_config = ConfigDict(frozen=True)

    overall_band: MatchBand
    min_similarity_score: float
    pairwise_matches: dict[str, IdentityMatchResult]
    audit_summary: list[str]
    audit_summary_ar: list[str]


# ---------------------------------------------------------------------------
# Pure-Python Jaro & Jaro-Winkler Implementation (100% Deterministic)
# ---------------------------------------------------------------------------

def jaro_similarity(s1: str, s2: str) -> float:
    """Compute exact Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    match_index = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[match_index]:
            match_index += 1
        if s1[i] != s2[match_index]:
            transpositions += 1
        match_index += 1

    half_transpositions = transpositions // 2
    return (
        matches / len1
        + matches / len2
        + (matches - half_transpositions) / matches
    ) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity with a prefix bonus up to 4 characters."""
    jaro_dist = jaro_similarity(s1, s2)
    prefix_length = 0
    for char1, char2 in zip(s1[:4], s2[:4]):
        if char1 == char2:
            prefix_length += 1
        else:
            break

    return jaro_dist + (prefix_length * prefix_weight * (1.0 - jaro_dist))


# ---------------------------------------------------------------------------
# Six-Stage Arabic Orthographic Normalization Pipeline
# ---------------------------------------------------------------------------

# Stage 1: Tashkeel (harakaat, tanween, shadda, sukun, dagger alif) & Tatweel/Kashida
_TASHKEEL_TATWEEL_PATTERN = re.compile(r"[\u064B-\u0655\u0670\u0640]")

# Stage 2: Hamza standardization (إ, أ, آ, ء -> ا)
_HAMZA_PATTERN = re.compile(r"[إأآء]")

# Stage 3: Yaa and Alif Maqsura unification (ى and Persian Yeh -> ي)
_YAA_PATTERN = re.compile(r"[ى\u06CC]")

# Stage 4: Taa Marbuta and Haa unification (ة -> ه)
_TAA_MARBUTA_PATTERN = re.compile(r"[ة]")

# Stage 5: Definite article prefix normalization (الـ -> ال)
_ARTICLE_TATWEEL_PATTERN = re.compile(r"الـ+")

# Stage 6: Compound name spacing patterns
# 6a. عبد + name (e.g. عبد الله -> عبدالله, عبد الرحمن -> عبدالرحمن)
_ABD_PATTERN = re.compile(r"\bعبد\s+(ال\w+|الله)\b")

# 6b. أمة / امة + name (e.g. أمة الله -> امهالله, أمة الرحمن -> امهالرحمن)
_AMAT_PATTERN = re.compile(r"\b(امة|أمة|امه|أمه)\s+(ال\w+|الله)\b")

# 6c. ... + الدين (e.g. نور الدين -> نورالدين, علاء الدين -> علاءالدين)
_DIN_PATTERN = re.compile(
    r"\b(نور|علاء|علا|صلاح|ضياء|ضيا|حسام|سيف|عماد|بهاء|بها|شمس|تقي|جمال|جلال|كمال|زين)\s+(ال)?دين\b"
)

# 6d. ... + الله (e.g. فضل الله -> فضلالله, نعمة الله -> نعمةالله)
_ALLAH_PATTERN = re.compile(
    r"\b(فضل|نعمة|نعمه|حبيب|عطاء|عطا|رحمة|رحمه|فتح|نصر|جار|لطف)\s+الله\b"
)

# 6e. Abu and Umm kunya markers (e.g. ابو بكر -> ابوبكر, أم كلثوم -> امكلثوم, أبو فهد -> ابوفهد)
_KUNYA_PREFIX_PATTERN = re.compile(r"\b(ابو|أبو|ابي|أبي|ابا|أبا|ام|أم)\s+(\w+)\b")


def strip_tashkeel(text: str) -> str:
    """Stage 1: Strip Arabic diacritics (harakaat, tanween, shadda) and Tatweel/Kashida."""
    return _TASHKEEL_TATWEEL_PATTERN.sub("", text)


def standardize_hamza(text: str) -> str:
    """Stage 2: Standardize Hamza variants (إ, أ, آ, ء) to bare Alif (ا)."""
    return _HAMZA_PATTERN.sub("ا", text)


def unify_yaa(text: str) -> str:
    """Stage 3: Unify Alif Maqsura (ى) and Persian Yeh to standard Arabic Yaa (ي)."""
    return _YAA_PATTERN.sub("ي", text)


def unify_taa_marbuta(text: str) -> str:
    """Stage 4: Unify Taa Marbuta (ة) to Haa (ه)."""
    return _TAA_MARBUTA_PATTERN.sub("ه", text)


def normalize_definite_article_prefix(text: str) -> str:
    """Stage 5: Normalize elongated definite article prefixes (الـ -> ال)."""
    return _ARTICLE_TATWEEL_PATTERN.sub("ال", text)


def unify_compound_names(text: str) -> str:
    """Stage 6: Remove interior spaces in compound names for unified single-token representation."""
    text = _ABD_PATTERN.sub(r"عبد\1", text)
    text = _AMAT_PATTERN.sub(r"امه\2", text)
    text = _DIN_PATTERN.sub(r"\1الدين", text)
    text = _ALLAH_PATTERN.sub(r"\1الله", text)
    text = _KUNYA_PREFIX_PATTERN.sub(r"\1\2", text)
    return text


def normalize_arabic_name(text: str) -> str:
    """Run full 6-stage orthographic normalization on an Arabic name string."""
    if not text or not text.strip():
        return ""
    text = strip_tashkeel(text)
    text = standardize_hamza(text)
    text = unify_yaa(text)
    text = unify_taa_marbuta(text)
    text = normalize_definite_article_prefix(text)
    text = unify_compound_names(text)
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------------
# Article De-biasing & Token Stem Similarity
# ---------------------------------------------------------------------------

def strip_definite_article_token(token: str) -> str:
    """Strip 'ال' prefix if token is at least 5 characters (ensuring a 3+ letter root)."""
    if token.startswith("ال") and len(token) >= 5:
        return token[2:]
    return token


def compute_token_similarity(token_a: str, token_b: str) -> float:
    """Compute Jaro-Winkler similarity, de-biasing shared 'ال' prefixes on tribal roots."""
    if token_a == token_b:
        return 1.0

    stem_a = strip_definite_article_token(token_a)
    stem_b = strip_definite_article_token(token_b)

    # If both or either had an article stripped, compare their stems to remove prefix bias
    if stem_a != token_a or stem_b != token_b:
        return jaro_winkler_similarity(stem_a, stem_b)

    return jaro_winkler_similarity(token_a, token_b)


# ---------------------------------------------------------------------------
# Arabic Onomastic Lexicons: Distinguishing Names from Kunyas
# ---------------------------------------------------------------------------

# Reference living lexicon: common Arabic given names starting with Alif-Meem that must
# never be misclassified as informal kunyas or popped away from the applicant's legal name.
NON_KUNYA_AM_NAMES = {
    "امل", "امال", "امين", "امينه", "امير", "اميره", "امجد",
    "اماني", "اميمه", "امامه", "امتياز", "امنيه"
}


def is_kunya_token(token: str) -> bool:
    """Check if token is an informal kunya marker or compound, protecting real Alif-Meem names."""
    if token in ("ابو", "ام", "ابي", "ابا"):
        return True
    if token.startswith("ابو") and len(token) >= 6:
        return True
    if token.startswith("ام") and len(token) >= 5:
        if token in NON_KUNYA_AM_NAMES:
            return False
        return True
    return False


def strip_kunya_marker(token: str) -> str:
    """Strip leading 'ابو' or 'ام' prefix to extract operative base name, protecting real names."""
    if token.startswith("ابو") and len(token) >= 6:
        return token[3:]
    if token.startswith("ام") and len(token) >= 5:
        if token in NON_KUNYA_AM_NAMES:
            return token
        return token[2:]
    return token


# ---------------------------------------------------------------------------
# Token-Weighted Patronymic Alignment & Matching
# ---------------------------------------------------------------------------

# Slot weights prioritizing Given and Father names over optional tribal titles
WEIGHT_GIVEN = 0.35
WEIGHT_FATHER = 0.30
WEIGHT_GRANDFATHER = 0.25
WEIGHT_SURNAME = 0.10


def match_arabic_names(name_a: str, name_b: str) -> IdentityMatchResult:
    """Compare two Arabic names with patronymic alignment and classify into triage bands.

    Triage Bands:
    - Score >= 0.88: AUTO_PASS
    - 0.70 <= Score < 0.88: HUMAN_ESCALATION
    - Score < 0.70: HARD_MISMATCH
    """
    normalized_a = normalize_arabic_name(name_a)
    normalized_b = normalize_arabic_name(name_b)

    tokens_a = normalized_a.split()
    tokens_b = normalized_b.split()

    audit_notes: list[str] = []
    audit_notes_ar: list[str] = []
    kunya_detected = False

    # Audit note for extended patronymic chains (> 4 tokens)
    if len(tokens_a) > 4:
        audit_notes.append(f"Extended patronymic chain ({len(tokens_a)} tokens); evaluated primary 4 slots.")
        audit_notes_ar.append(f"سلسلة نسب مطولة ({len(tokens_a)} أجزاء)؛ تم تقييم الحقول الأربعة الأساسية.")
    if len(tokens_b) > 4:
        audit_notes.append(f"Extended patronymic chain ({len(tokens_b)} tokens); evaluated primary 4 slots.")
        audit_notes_ar.append(f"سلسلة نسب مطولة ({len(tokens_b)} أجزاء)؛ تم تقييم الحقول الأربعة الأساسية.")

    # Guarded alignment-aware teknonym prefix detection (Abu / Umm):
    # Detects extraneous informal prefixes added before a full legal chain,
    # strictly guarded by is_kunya_token to prevent misclassifying real Alif-Meem names.
    if len(tokens_a) >= 3 and len(tokens_b) >= 2:
        if is_kunya_token(tokens_a[0]) and compute_token_similarity(tokens_a[0], tokens_b[0]) < 0.70:
            if compute_token_similarity(tokens_a[1], tokens_b[0]) >= 0.70:
                kunya_val = tokens_a.pop(0)
                kunya_detected = True
                audit_notes.append(
                    f"Informal teknonym prefix detected in primary name ('{kunya_val}'); patronymic tokens shifted."
                )
                audit_notes_ar.append(
                    f"تم رصد كنية غير رسمية سابقة للاسم في الوثيقة الأولى ('{kunya_val}')؛ تمت محاذاة سلسلة النسب."
                )

    if len(tokens_b) >= 3 and len(tokens_a) >= 2:
        if is_kunya_token(tokens_b[0]) and compute_token_similarity(tokens_b[0], tokens_a[0]) < 0.70:
            if compute_token_similarity(tokens_b[1], tokens_a[0]) >= 0.70:
                kunya_val = tokens_b.pop(0)
                kunya_detected = True
                audit_notes.append(
                    f"Informal teknonym prefix detected in secondary name ('{kunya_val}'); patronymic tokens shifted."
                )
                audit_notes_ar.append(
                    f"تم رصد كنية غير رسمية سابقة للاسم في الوثيقة الثانية ('{kunya_val}')؛ تمت محاذاة سلسلة النسب."
                )

    len_a = len(tokens_a)
    len_b = len(tokens_b)

    # Guard: empty names
    if len_a == 0 or len_b == 0:
        return IdentityMatchResult(
            raw_name_a=name_a,
            raw_name_b=name_b,
            normalized_name_a=normalized_a,
            normalized_name_b=normalized_b,
            similarity_score=0.0,
            triage_band=MatchBand.HARD_MISMATCH,
            token_details=[],
            audit_notes=["One or both names are empty after normalization."],
            audit_notes_ar=["أحد الاسمين أو كلاهما فارغ بعد المعالجة."],
        )

    # 1. Given Name (Slot 0): check direct match, then kunya-stem variation
    sim_given = compute_token_similarity(tokens_a[0], tokens_b[0])
    if sim_given < 0.70:
        stem_a = strip_kunya_marker(tokens_a[0])
        stem_b = strip_kunya_marker(tokens_b[0])
        if stem_a != tokens_a[0] or stem_b != tokens_b[0]:
            kunya_given_sim = compute_token_similarity(stem_a, stem_b)
            if kunya_given_sim >= 0.70:
                sim_given = kunya_given_sim
                kunya_detected = True
                audit_notes.append(
                    f"Kunya variation detected on Given Name ('{tokens_a[0]}' vs '{tokens_b[0]}')."
                )
                audit_notes_ar.append(
                    f"تم رصد اختلاف كنية في الاسم الأول ('{tokens_a[0]}' مقابل '{tokens_b[0]}')."
                )

    # 2. Father's Name (Slot 1)
    sim_father = 0.0
    tok_father_a = tokens_a[1] if len_a > 1 else None
    tok_father_b = tokens_b[1] if len_b > 1 else None
    if tok_father_a and tok_father_b:
        sim_father = compute_token_similarity(tok_father_a, tok_father_b)

    # -----------------------------------------------------------------------
    # Branch 1: Short Names (1 or 2 tokens) - Renormalized
    # -----------------------------------------------------------------------
    if len_a < 3 or len_b < 3:
        token_details = [
            TokenMatchDetail(
                role=PatronymicRole.GIVEN,
                token_a=tokens_a[0],
                token_b=tokens_b[0],
                similarity=sim_given,
                weight=WEIGHT_GIVEN,
                weighted_score=sim_given * WEIGHT_GIVEN,
            )
        ]
        weight_sum = WEIGHT_GIVEN
        weighted_val = WEIGHT_GIVEN * sim_given

        if tok_father_a and tok_father_b:
            token_details.append(
                TokenMatchDetail(
                    role=PatronymicRole.FATHER,
                    token_a=tok_father_a,
                    token_b=tok_father_b,
                    similarity=sim_father,
                    weight=WEIGHT_FATHER,
                    weighted_score=sim_father * WEIGHT_FATHER,
                )
            )
            weight_sum += WEIGHT_FATHER
            weighted_val += WEIGHT_FATHER * sim_father

        norm_score = weighted_val / weight_sum
        # Under KYC rules, 2-part names cannot Auto-Pass (requires >= 3 parts),
        # so matching 2-part names cap at 0.850 to trigger HUMAN_ESCALATION
        final_score = round(min(norm_score, 0.85), 4)
        band = MatchBand.HUMAN_ESCALATION if final_score >= 0.70 else MatchBand.HARD_MISMATCH
        audit_notes.append("Insufficient patronymic chain length (fewer than 3 tokens); routed to human review.")
        audit_notes_ar.append("عدد أجزاء الاسم غير كافٍ للتحقق التلقائي (أقل من ثلاثة أسماء)؛ تتطلب مراجعة بشرية.")

        return IdentityMatchResult(
            raw_name_a=name_a,
            raw_name_b=name_b,
            normalized_name_a=normalized_a,
            normalized_name_b=normalized_b,
            similarity_score=final_score,
            triage_band=band,
            token_details=token_details,
            audit_notes=audit_notes,
            audit_notes_ar=audit_notes_ar,
        )

    # -----------------------------------------------------------------------
    # Branch 2: 3-Part vs 3-Part Names - Renormalized over 3 slots
    # -----------------------------------------------------------------------
    if len_a == 3 and len_b == 3:
        tok_gf_a = tokens_a[2]
        tok_gf_b = tokens_b[2]
        sim_gf = compute_token_similarity(tok_gf_a, tok_gf_b)

        weight_sum = WEIGHT_GIVEN + WEIGHT_FATHER + WEIGHT_GRANDFATHER
        score_3 = (
            WEIGHT_GIVEN * sim_given
            + WEIGHT_FATHER * sim_father
            + WEIGHT_GRANDFATHER * sim_gf
        ) / weight_sum

        token_details = [
            TokenMatchDetail(role=PatronymicRole.GIVEN, token_a=tokens_a[0], token_b=tokens_b[0], similarity=sim_given, weight=WEIGHT_GIVEN, weighted_score=sim_given * WEIGHT_GIVEN),
            TokenMatchDetail(role=PatronymicRole.FATHER, token_a=tokens_a[1], token_b=tokens_b[1], similarity=sim_father, weight=WEIGHT_FATHER, weighted_score=sim_father * WEIGHT_FATHER),
            TokenMatchDetail(role=PatronymicRole.GRANDFATHER, token_a=tok_gf_a, token_b=tok_gf_b, similarity=sim_gf, weight=WEIGHT_GRANDFATHER, weighted_score=sim_gf * WEIGHT_GRANDFATHER),
        ]

        if kunya_detected:
            final_score = round(min(score_3, 0.85), 4)
        else:
            final_score = round(score_3, 4)

        if final_score >= 0.88:
            band = MatchBand.AUTO_PASS
        elif final_score >= 0.70:
            band = MatchBand.HUMAN_ESCALATION
            audit_notes.append("Sub-threshold 3-part patronymic similarity.")
            audit_notes_ar.append("تشابه جزئي في الاسم الثلاثي يتطلب مراجعة بشرية.")
        else:
            band = MatchBand.HARD_MISMATCH
            audit_notes.append("Irreconcilable patronymic name mismatch.")
            audit_notes_ar.append("عدم تطابق جوهري في الاسم الثلاثي.")

        return IdentityMatchResult(
            raw_name_a=name_a,
            raw_name_b=name_b,
            normalized_name_a=normalized_a,
            normalized_name_b=normalized_b,
            similarity_score=final_score,
            triage_band=band,
            token_details=token_details,
            audit_notes=audit_notes,
            audit_notes_ar=audit_notes_ar,
        )

    # -----------------------------------------------------------------------
    # Branch 3: 4-Part vs 4-Part or 4-Part vs 3-Part Names
    # -----------------------------------------------------------------------
    sim_gf = 0.0
    tok_gf_a = None
    tok_gf_b = None
    sim_sur = 0.0
    tok_sur_a = None
    tok_sur_b = None
    has_conflicting_surname = False
    ambiguous_alignment = False

    if len_a >= 4 and len_b >= 4:
        # Both documents have full 4-part patronymic chains
        tok_gf_a = tokens_a[2]
        tok_gf_b = tokens_b[2]
        sim_gf = compute_token_similarity(tok_gf_a, tok_gf_b)

        tok_sur_a = tokens_a[3]
        tok_sur_b = tokens_b[3]
        sim_sur = compute_token_similarity(tok_sur_a, tok_sur_b)

        if sim_sur < 0.70:
            has_conflicting_surname = True
            audit_notes.append(
                f"Conflicting tribal surnames detected: '{tok_sur_a}' vs '{tok_sur_b}'."
            )
            audit_notes_ar.append(
                f"اختلاف في اللقب العشائري بين الوثائق: '{tok_sur_a}' مقابل '{tok_sur_b}'."
            )

    else:
        # 4-token vs 3-token: evaluate whether the 3rd token of shorter name is Grandfather or Surname
        tok_long = tokens_a if len_a >= 4 else tokens_b
        tok_short = tokens_b if len_a >= 4 else tokens_a

        sim_to_gf = compute_token_similarity(tok_short[2], tok_long[2])
        sim_to_sur = compute_token_similarity(tok_short[2], tok_long[3])

        # Flag ambiguous alignment if 3rd token is near-tie to both slots
        if abs(sim_to_gf - sim_to_sur) < 0.05 and sim_to_gf >= 0.50:
            ambiguous_alignment = True
            audit_notes.append("Ambiguous alignment: 3rd token has near-equal similarity to Grandfather and Surname.")
            audit_notes_ar.append("تشابه متقارب بين اسم الجد واللقب في الاسم الثلاثي؛ تعذر الحسم التلقائي.")

        if sim_to_gf >= sim_to_sur:
            # 3rd token matches Grandfather; Surname was omitted in shorter name
            tok_gf_a = tokens_a[2]
            tok_gf_b = tokens_b[2]
            sim_gf = sim_to_gf

            tok_sur_a = tokens_a[3] if len_a >= 4 else None
            tok_sur_b = tokens_b[3] if len_b >= 4 else None
            sim_sur = 0.0

            audit_notes.append("Tribal surname omitted in secondary document.")
            audit_notes_ar.append("اللقب العشائري محذوف في إحدى الوثائق.")
        else:
            # 3rd token matches Surname; Grandfather was omitted in shorter name
            tok_gf_a = tokens_a[2] if len_a >= 4 else None
            tok_gf_b = tokens_b[2] if len_b >= 4 else None
            sim_gf = 0.0

            tok_sur_a = tokens_a[3] if len_a >= 4 else tokens_a[2]
            tok_sur_b = tokens_b[3] if len_b >= 4 else tokens_b[2]
            sim_sur = sim_to_sur

            audit_notes.append("Grandfather name omitted in secondary document.")
            audit_notes_ar.append("اسم الجد محذوف في إحدى الوثائق.")

    # Calculate 4-part weighted score
    weighted_given = WEIGHT_GIVEN * sim_given
    weighted_father = WEIGHT_FATHER * sim_father
    weighted_gf = WEIGHT_GRANDFATHER * sim_gf
    weighted_sur = WEIGHT_SURNAME * sim_sur
    raw_score = weighted_given + weighted_father + weighted_gf + weighted_sur

    token_details = [
        TokenMatchDetail(role=PatronymicRole.GIVEN, token_a=tokens_a[0], token_b=tokens_b[0], similarity=sim_given, weight=WEIGHT_GIVEN, weighted_score=weighted_given),
        TokenMatchDetail(role=PatronymicRole.FATHER, token_a=tok_father_a, token_b=tok_father_b, similarity=sim_father, weight=WEIGHT_FATHER, weighted_score=weighted_father),
        TokenMatchDetail(role=PatronymicRole.GRANDFATHER, token_a=tok_gf_a, token_b=tok_gf_b, similarity=sim_gf, weight=WEIGHT_GRANDFATHER, weighted_score=weighted_gf),
        TokenMatchDetail(role=PatronymicRole.SURNAME, token_a=tok_sur_a, token_b=tok_sur_b, similarity=sim_sur, weight=WEIGHT_SURNAME, weighted_score=weighted_sur),
    ]

    # Enforce policy adjustments:
    # 1. Conflicting explicit tribal surnames cap score to 0.82 (forcing escalation)
    # 2. Informal kunya or ambiguous slot alignment caps score to 0.85 (forcing escalation)
    if has_conflicting_surname:
        final_score = min(raw_score, 0.82)
    elif kunya_detected or ambiguous_alignment:
        final_score = min(raw_score, 0.85)
    else:
        final_score = raw_score

    final_score = round(final_score, 4)

    # Classify triage band
    if final_score >= 0.88:
        band = MatchBand.AUTO_PASS
    elif final_score >= 0.70:
        band = MatchBand.HUMAN_ESCALATION
    else:
        band = MatchBand.HARD_MISMATCH
        audit_notes.append("Cross-document patronymic similarity falls below 0.70 threshold.")
        audit_notes_ar.append("نسبة تطابق الهوية بين الوثائق أقل من حد التطابق الأدنى (0.70).")

    return IdentityMatchResult(
        raw_name_a=name_a,
        raw_name_b=name_b,
        normalized_name_a=normalized_a,
        normalized_name_b=normalized_b,
        similarity_score=final_score,
        triage_band=band,
        token_details=token_details,
        audit_notes=audit_notes,
        audit_notes_ar=audit_notes_ar,
    )


# ---------------------------------------------------------------------------
# Cross-Document Tripartite Reconciliation
# ---------------------------------------------------------------------------

def reconcile_onboarding_identities(
    national_card_name: str,
    business_license_name: str | None = None,
    tax_card_name: str | None = None,
) -> CrossDocumentReconciliationResult:
    """Reconcile identity names across all provided onboarding documents.

    Evaluates pairwise matches between National ID, Business License, and Tax Card.
    Returns overall triage band (AUTO_PASS, HUMAN_ESCALATION, HARD_MISMATCH).
    """
    pairwise_matches: dict[str, IdentityMatchResult] = {}
    audit_summary: list[str] = []
    audit_summary_ar: list[str] = []

    pairs_to_test: list[tuple[str, str, str]] = []
    if business_license_name:
        pairs_to_test.append(("national_vs_business", national_card_name, business_license_name))
    if tax_card_name:
        pairs_to_test.append(("national_vs_tax", national_card_name, tax_card_name))
    if business_license_name and tax_card_name:
        pairs_to_test.append(("business_vs_tax", business_license_name, tax_card_name))

    if not pairs_to_test:
        raise ValueError("At least two document names are required for cross-document reconciliation.")

    min_score = 1.0
    has_hard_mismatch = False
    has_escalation = False

    for pair_key, name_1, name_2 in pairs_to_test:
        match_res = match_arabic_names(name_1, name_2)
        pairwise_matches[pair_key] = match_res
        if match_res.similarity_score < min_score:
            min_score = match_res.similarity_score

        if match_res.triage_band == MatchBand.HARD_MISMATCH:
            has_hard_mismatch = True
            audit_summary.append(f"Hard mismatch in pair '{pair_key}' ({match_res.similarity_score:.3f}).")
            audit_summary_ar.append(f"عدم تطابق جوهري بين الوثائق ({pair_key}): النسبة {match_res.similarity_score:.3f}.")
        elif match_res.triage_band == MatchBand.HUMAN_ESCALATION:
            has_escalation = True
            audit_summary.extend(match_res.audit_notes)
            audit_summary_ar.extend(match_res.audit_notes_ar)

    if has_hard_mismatch:
        overall_band = MatchBand.HARD_MISMATCH
    elif has_escalation:
        overall_band = MatchBand.HUMAN_ESCALATION
    else:
        overall_band = MatchBand.AUTO_PASS
        audit_summary.append("All cross-document identity checks passed above 0.88 threshold.")
        audit_summary_ar.append("تمت مطابقة جميع وثائق الهوية بنجاح أعلى من عتبة القبول التلقائي (0.88).")

    return CrossDocumentReconciliationResult(
        overall_band=overall_band,
        min_similarity_score=round(min_score, 4),
        pairwise_matches=pairwise_matches,
        audit_summary=audit_summary,
        audit_summary_ar=audit_summary_ar,
    )
