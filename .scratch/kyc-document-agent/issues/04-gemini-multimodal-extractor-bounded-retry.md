# 04: Gemini Multimodal Extractor with Bounded 1x Retry

**What to build:** An isolated document-parallel multimodal extraction pipeline using Gemini that extracts structured Pydantic schemas, verifies confidence signals with immediate logprob probing (falling back to verbalized confidence if unavailable), and triggers a strictly bounded single retry on recoverable optical defects.

**Blocked by:** 01 (Synthetic Document Generator), 02 (OpenCV Diagnostics & Pydantic Schemas).

**Status:** ready-for-agent

## Acceptance Criteria

- [x] Early spike: Probe Gemini API token logprobs on structured multimodal extraction; establish working fallback to verbalized field-level confidence if logprobs are unsupported or sparse.
- [x] Multimodal extraction runs in parallel, isolated contexts per document (National ID, Business License, Tax Card) to prevent token contamination across documents.
- [x] Structured extraction enforces Pydantic models with bounding box predictions and field confidence.
- [x] Composite confidence calculation integrates model extraction signal, local OpenCV clarity metrics, and format checks.
- [x] Bounded 1x retry loop: If a Tier 1 field exhibits low confidence ($<0.85$) and the localized bounding box suffers from recoverable optical defects (glare/low contrast), the agent invokes a re-crop and CLAHE contrast enhancement tool.
- [x] Retries are strictly capped at maximum one attempt per document.
- [x] Unrecoverable errors (structural format invalidity, illegible handwriting) bypass the retry tool and route directly to human escalation.
