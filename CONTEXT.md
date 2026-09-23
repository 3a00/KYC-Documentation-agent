# KYC Document Agent

Autonomous verification and triaging domain for ZainCash merchant and customer onboarding documents.

## Language

### Documents

**Unified National Card**:
The primary Iraqi identity document (البطاقة الوطنية الموحدة) bearing the 12-digit National ID number, bearer full name, family details, and expiration date (front face only).
_Avoid_: Citizen ID, civil status ID, jinsiya

**Business License**:
The municipal or commercial registry document (إجازة ممارسة مهنة / سجل تجاري) establishing a merchant's business entity name and designated business owner.
_Avoid_: Trade permit, merchant certificate

**Tax Card**:
The official Iraqi General Commission for Taxes identification card (الهوية الضريبية) certifying taxpayer registration and tax identification number.
_Avoid_: Tax certificate, fiscal card

### Onboarding Outcomes

**Auto-Pass**:
The automated clearance state emitted when all required document fields exceed confidence thresholds and pass deterministic cross-validation.
_Avoid_: Auto-approval, instant verified

**Human Escalation**:
The triaging state that routes an application to an operations officer alongside an auditable Arabic exception report detailing specific discrepancies.
_Avoid_: Rejection, manual review flag, error drop

**Hard Mismatch**:
The state where cross-document identity tokens fall below the irreconcilable threshold (<0.70 similarity), indicating distinct legal identities.
_Avoid_: Fraud flag, forgery detection

### Identity & Extraction

**Patronymic Name Chain**:
The standardized 4-part Iraqi naming convention (Given, Father, Grandfather, Surname/Tribe) evaluated under deterministic orthographic normalization.
_Avoid_: First and last name, single name string

**Tier 1 Fields**:
Critical identity attributes (National ID Number, Full Name, Expiration Date) requiring strict $\ge 0.85$ calibrated confidence to avoid human escalation.
_Avoid_: Primary fields, core keys

**Tier 2 Fields**:
Secondary verification attributes (Mother's Name, Issue Date, Province) requiring $\ge 0.70$ confidence; sub-threshold values append audit warnings to the dossier without blocking auto-pass.
_Avoid_: Optional fields, minor fields

**Calibrated Field Confidence**:
The composite score reflecting multimodal extraction certainty grounded in physical image quality and deterministic format validity.
_Avoid_: Verbalized confidence, self-rated probability

**Deterministic Format Validation**:
Structural integrity checks verifying exact digit counts (e.g., 12-digit National ID) and calendar date consistency, independent of unverified checksum algorithms.
_Avoid_: Checksum validation, hash check
