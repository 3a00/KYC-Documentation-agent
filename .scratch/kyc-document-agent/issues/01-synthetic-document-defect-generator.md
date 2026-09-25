# 01: Synthetic Iraqi Document & Defect Generator

**What to build:** A programmatic generation pipeline producing paired synthetic Iraqi identity documents (ground-truth metadata + rendered image with exact field bounding boxes), capable of injecting localized optical defects (specular glare, blur, tilt) to test model behavior under adverse capture conditions.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Acceptance Criteria

- [x] Generates realistic synthetic front-face templates for Unified National Card, Business License, and Tax Card using authentic Iraqi layout proportions and Arabic Noto typography.
- [x] Reshapes and reorders Arabic text properly using `arabic_reshaper` and `python-bidi` so compound names and ligatures render correctly.
- [x] Synthesizes fictional Iraqi identity data (4-part patronymic name chains, 12-digit numeric National IDs, valid date ranges, Iraqi provinces) conforming strictly to zero real-customer data rules.
- [x] Records exact pixel bounding box coordinates for each drawn text field during rendering.
- [x] Injects localized optical defects: specular glare patches placed specifically over targeted field coordinates, Gaussian blur, and perspective tilt.
- [x] Produces paired outputs: clean baseline image + degraded image + ground-truth JSON metadata.
