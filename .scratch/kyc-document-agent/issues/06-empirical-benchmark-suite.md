# 06: Empirical Benchmark Suite (Accuracy, ECE & Triage Distribution)

**What to build:** A quantitative evaluation runner that executes the verification pipeline across a held-out synthetic test set with planted defects and mismatches, outputting verified extraction accuracy, Expected Calibration Error (ECE), and triage distribution metrics ready for manual placement into Round 2 submission slides.

**Blocked by:** 05 (Deterministic Triage Engine & Exception Dossier Generator).

**Status:** ready-for-agent

## Acceptance Criteria

- [x] Runs end-to-end evaluation across a held-out test split of synthetic Iraqi document sets (clean, specular glare, blurred, tilted, and planted identity mismatch cases).
- [x] Computes field-level extraction accuracy for character and numeric fields against ground-truth labels.
- [x] Computes Expected Calibration Error (ECE) and tabulates calibration curve data (empirical accuracy vs predicted confidence across bins).
- [x] Measures False Auto-Pass Rate (confirming it is $<0.5\%$ on invalid/obscured test cases) and average human escalation rate.
- [x] Outputs a concise quantitative summary report providing the verified empirical numbers to directly populate the Round 2 pitch slides.
