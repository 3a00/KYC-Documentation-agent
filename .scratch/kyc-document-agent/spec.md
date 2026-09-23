# Spec: KYC Document Agent

**Status:** ready-for-agent  
**Scope:** Front-side document onboarding triaging for ZainCash  
**Target Delivery:** Bounded multi-document verification pipeline and held-out empirical evaluation  

---

## Problem Statement

ZainCash operations officers are overwhelmed by manual onboarding reviews for merchants and individual customers. Applicants submit phone camera photos of their Unified National Card, Business License, and Tax Card. These photos are frequently compromised by poor lighting, steep angles, plastic-laminate glare, and handwritten annotations in Arabic script.

Human review of every document is slow, expensive, and inconsistent. Reviewers spend most of their time repeatedly verifying routine, legible documents while struggling to read degraded text on edge cases. Existing automated systems fail because pure computer vision reject-gates trigger false alarms on valid cards with minor reflections, while naive LLM extraction produces overconfident hallucinations on obscured fields.

## Solution

The KYC Document Agent provides an autonomous, auditable verification and triaging pipeline that processes front-side photos of all three required onboarding documents in parallel.

The system uses computer vision solely as a diagnostic sensor to detect blur and glare without rejecting legible cards. A multimodal foundation model extracts structured identity attributes into strict schemas with calibrated field-level confidence scores. When low confidence on a critical field is caused by a recoverable optical defect, the agent triggers a bounded single-retry tool call to enhance and re-extract that specific region. 

All cross-document identity reconciliation and final routing decisions are executed by deterministic, transparent Python business logic using comprehensive Arabic orthographic normalization. The system automatically clears valid applicants (Auto-Pass), routes ambiguous cases to an operations officer with an actionable Arabic exception report (Human Escalation), and flags irreconcilable identity discrepancies (Hard Mismatch).

---

## User Stories

1. As a ZainCash KYC operations officer, I want fully legible, valid document sets to Auto-Pass automatically, so that I can focus my limited time exclusively on ambiguous or high-risk applications.
2. As a ZainCash KYC operations officer, I want every escalated application to arrive with an actionable Arabic exception note explaining exactly which field failed and why, so that I can resolve discrepancies in under 15 seconds.
3. As a ZainCash KYC operations officer, I want unreadable or obscured document fields to be recorded as null and flagged rather than guessed, so that the bank never accepts hallucinated applicant credentials.
4. As a ZainCash KYC operations officer, I want to see both applicant name strings side-by-side when names partially mismatch due to informal kunyas or missing tribal surnames, so that I can make a 1-click approval decision.
5. As a ZainCash KYC operations officer, I want secondary field warnings (like minor blur on mother's name or issue date) logged in the audit trail without blocking onboarding when core legal identity matches, so that legitimate customers are not unnecessarily delayed.
6. As a ZainCash merchant applicant, I want slight phone reflections or paper folds not to cause immediate rejection of my application, so that I do not have to repeatedly retake photos.
7. As a ZainCash compliance auditor, I want all cross-document matching rules and Arabic text normalizations to be deterministic and inspectable, so that automated onboarding decisions are fully traceable.
8. As a ZainCash compliance auditor, I want extraction confidence scores to be mathematically calibrated, so that a 90% confidence score accurately reflects a 90% probability of ground-truth correctness.
9. As a ZainCash system administrator, I want re-extraction retry loops to be strictly capped at one attempt per document, so that system latency, API costs, and token usage remain strictly bounded.
10. As a ZainCash system administrator, I want document extraction to occur in parallel contexts for each card, so that issues on a business license photo do not corrupt or re-trigger processing of a valid national ID.
11. As an evaluation judge, I want to see quantitative performance metrics measured on a held-out synthetic test set with realistic physical noise, so that system reliability is proven with empirical evidence rather than speculative claims.
12. As an evaluation judge, I want to see at least three documented failure modes and clear fallback pathways, so that I know how the system behaves safely on the worst-quality inputs.

---

## Implementation Decisions

### 1. Document Extraction Seam
- The primary interface for testing and execution is a unified onboarding package verifier:
  `verify_onboarding_package(national_id_image, business_license_image, tax_card_image) -> OnboardingDossier`
- The pipeline processes the front face of three document types: Unified National Card, Business License, and Tax Card.
- Internal extraction runs document-by-document in isolated parallel contexts to avoid cross-card context contamination.

### 2. OpenCV Diagnostic Layer
- Computer vision algorithms calculate image-level and bounding-box quality indicators: Laplacian blur variance and HSV color-space specular glare ratios.
- These indicators are emitted as structured diagnostic metadata into the confidence evaluation engine rather than acting as hard rejection gates.

### 3. Multimodal Extraction Schemas
- Multimodal extraction enforces strict Pydantic models for each document type:
  - `NationalIDSchema`: 12-digit national ID number, full name, mother's name, expiration date, issue date, province.
  - `BusinessLicenseSchema`: Business entity name, authorized merchant name, license number, issue date, activity type.
  - `TaxCardSchema`: Tax identification number, taxpayer name, fiscal year, issue date.
- Every extracted field is paired with a calibrated confidence float ($0.0 - 1.0$) and an `obscured` boolean flag.
- Unreadable or missing fields strictly output `value: null` with `obscured: true`.

### 4. Bounded 1x Retry Engine
- The agent is equipped with an image inspection and crop-enhancement tool.
- If a critical Tier 1 field exhibits low confidence and OpenCV identifies an optical defect (localized glare or low contrast) in that field's bounding box, the agent calls the enhancement tool once to obtain a contrast-adjusted crop and re-evaluates.
- Retries are strictly capped at maximum one attempt per document.
- Unrecoverable defects (e.g. illegible handwriting or structural digit count errors) bypass the retry tool and route immediately to Human Escalation.

### 5. Calibrated Field Confidence Formula
- Field confidence is calculated as a composite score:
  `Field Confidence = Model Extraction Signal × Local Clarity Factor × Format Validity Factor`
- National ID numbers undergo deterministic format validation (verifying exact 12-digit numeric length and date ranges; no mathematical checksum algorithm is assumed).

### 6. Tiered Criticality Thresholds
- **Tier 1 (Critical):** National ID Number, Full Name, Expiration Date. Threshold is strictly $\ge 0.85$. Any drop below 0.85 triggers Human Escalation.
- **Tier 2 (Contextual):** Mother's Name, Issue Date, Province/Address. Threshold is $\ge 0.70$. Drops below 0.70 append audit warnings to the dossier but do not block Auto-Pass if Tier 1 and cross-matching pass.

### 7. Deterministic Arabic Normalization & Matching
- Cross-document applicant matching applies a 6-stage orthographic normalizer:
  1. Tashkeel / diacritics stripping.
  2. Hamza standardization (`إ`, `أ`, `آ`, `ء` $\rightarrow$ `ا`).
  3. Yaa and Alif Maqsura unification (`ى` $\rightarrow$ `ي`).
  4. Taa Marbuta and Haa unification (`ة` $\rightarrow$ `ه`).
  5. Definite article prefix normalization (`الـ`).
  6. Compound name spacing standardization (`عبد الله` $\leftrightarrow$ `عبدالله`).
- Patronymic token comparison uses token-weighted Jaro-Winkler similarity:
  - $\ge 0.88$: **Auto-Pass** (exact or standard phonetic match).
  - $0.70 \le \text{Score} < 0.88$: **Human Escalation** (suspected informal kunya or omitted tribal surname).
  - $< 0.70$: **Hard Mismatch**.

### 8. Final Triage Policy
- The final triage state (`Auto-Pass`, `Human Escalation`, `Hard Mismatch`) is calculated strictly by deterministic Python logic outside the agent extraction loop.
- Escalations produce a human-readable Arabic exception summary indicating the failing field, confidence scores, and visual coordinates.

---

## Testing Decisions

### 1. Seam-Level Black-Box Testing
- Tests must target the external interface (`verify_onboarding_package`) rather than internal prompt variations or intermediate LLM tokens.
- Inputs are sets of 3 document images; outputs are asserted against the expected `OnboardingDossier` triage status and extracted field values.

### 2. Synthetic Test Dataset Generation
- Ground-truth synthetic Iraqi documents are generated programmatically using PIL and authentic Arabic Noto fonts, with exact bounding box coordinates recorded.
- Defect injection includes localized specular glare, optical blur, perspective tilt, and shadow occlusions.
- A physical subset of documents is printed on cardstock/laminate and photographed under varied phone camera angles and ambient lighting to ensure real-world optical validity.

### 3. Quantitative Evaluation Metrics
- **Field Extraction Accuracy:** Percentage of correctly extracted characters and numbers on legible fields.
- **Expected Calibration Error (ECE):** Deviation between predicted confidence scores and empirical correctness across confidence bins.
- **False Auto-Pass Rate:** Must be $<0.5\%$ on documents with planted errors or obscured critical fields.
- **Escalation Accuracy:** Appropriate triaging of ambiguous cases to human review without unnecessary false alarms on minor defects.

---

## Out of Scope

- Facial biometrics, selfie-to-ID matching, and facial liveness detection.
- Physical document forgery and fraud forensics (e.g. UV ink validation, microprint inspection).
- Integration with external government or banking production APIs.
- Reverse-side document analysis (scope is restricted to front faces).
- Mathematical checksum validation for Iraqi National IDs (until an official public specification is published).

---

## Further Notes

- All synthetic generation and testing must comply strictly with the hackathon rule: **zero real customer or identity records**.
- Once the pipeline and evaluation suite are verified, measured results will directly populate the two Round 2 submission slides.
