# 05: Deterministic Triage Engine & Exception Dossier Generator

**What to build:** The top-level verification seam that coordinates document extraction, applies tiered field confidence policies (Tier 1 $\ge 0.85$, Tier 2 $\ge 0.70$), evaluates cross-document identity consistency, and compiles an actionable Arabic exception report for human operations officers.

**Blocked by:** 02 (OpenCV Diagnostics & Pydantic Schemas), 03 (Arabic Normalizer & Matcher), 04 (Gemini Extractor & Bounded Retry).

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] Implements the top-level testing seam:
  `verify_onboarding_package(national_id_image, business_license_image, tax_card_image) -> OnboardingDossier`
- [ ] Tier 1 Policy: National ID Number, Full Name, Expiration Date strictly require $\ge 0.85$ confidence. Any drop immediately triggers `HUMAN_ESCALATION`.
- [ ] Tier 2 Policy: Mother's Name, Issue Date, Province require $\ge 0.70$ confidence. Drops below 0.70 append audit warnings to the dossier without blocking `AUTO_PASS` if Tier 1 and cross-matching pass.
- [ ] Incorporates deterministic cross-document matching results across National ID, Business License, and Tax Card.
- [ ] Emits one of three clear lifecycle outcomes: `AUTO_PASS`, `HUMAN_ESCALATION`, or `HARD_MISMATCH`.
- [ ] Generates an auditable Arabic exception dossier for escalated files, highlighting the failing field name, confidence scores, visual coordinates, and side-by-side string diffs.
