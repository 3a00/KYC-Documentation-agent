# 02: OpenCV Quality Diagnostics & Pydantic Schemas

**What to build:** An image quality diagnostic module that extracts blur variance and glare ratios at full-image and bounding-box levels, alongside formal Pydantic extraction schemas enforcing calibrated confidence and zero-hallucination (`null` + `obscured_flag=True`) semantics.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] OpenCV diagnostic functions compute Laplacian blur variance and HSV color-space specular glare ratios for whole images and localized bounding boxes.
- [ ] Quality diagnostics output structured metrics rather than acting as hard reject gates.
- [ ] Formal Pydantic schemas defined for each document:
  - `NationalIDSchema` (12-digit ID, full name, mother's name, expiry date, issue date, province).
  - `BusinessLicenseSchema` (business entity name, merchant full name, license number, issue date).
  - `TaxCardSchema` (tax ID, taxpayer name, fiscal year, issue date).
- [ ] Each extracted field is wrapped in a container providing `value`, `confidence` (float 0.0–1.0), and `obscured` (boolean).
- [ ] Schema validation enforces `value: null` when `obscured: true`, preventing hallucinated content from being accepted.
- [ ] Deterministic format validators confirm 12-digit numeric constraint for Iraqi National ID.
