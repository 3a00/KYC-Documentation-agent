# 03: Deterministic Arabic Normalizer & Cross-Document Matcher

**What to build:** An auditable Arabic identity reconciliation engine that normalizes patronymic name variations, strips orthographic noise, and computes token-weighted fuzzy similarity across onboarding documents to deterministically classify matches into Auto-Pass, Escalation, or Hard Mismatch.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] Six-stage Arabic orthographic normalization pipeline:
  - Diacritics/Tashkeel stripping.
  - Hamza standardization (`إ`, `أ`, `آ`, `ء` $\rightarrow$ `ا`).
  - Yaa and Alif Maqsura unification (`ى` $\rightarrow$ `ي`).
  - Taa Marbuta and Haa unification (`ة` $\rightarrow$ `ه`).
  - Definite article prefix normalization (`الـ`).
  - Compound name spacing unification (`عبد الله` $\leftrightarrow$ `عبدالله`).
- [ ] Token-weighted patronymic similarity comparison prioritizing Given and Father names over optional tribal titles.
- [ ] Deterministic triage bands based on normalized similarity score:
  - Score $\ge 0.88$: `AUTO_PASS` match.
  - $0.70 \le \text{Score} < 0.88$: `HUMAN_ESCALATION` (suspected informal kunya or missing surname).
  - Score $< 0.70$: `HARD_MISMATCH`.
- [ ] Unit test suite verifying edge cases (e.g. `محمد عبد الله كاظم الزبيدي` matching `محمد عبدالله كاظم`, and rejecting non-matching names).
