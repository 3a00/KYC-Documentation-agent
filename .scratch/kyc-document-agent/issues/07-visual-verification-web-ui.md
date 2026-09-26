# 07: Visual Verification Web UI & Drag-and-Drop Sample Suite

**What to build:** An interactive Streamlit web dashboard for visual end-to-end verification of onboarding packages (Unified National Card, Business License, Tax Card), rendering side-by-side document views with bounding-box anomaly overlays, live triage lifecycle banners, an Arabic RTL exception dossier, and patronymic slot diffs, accompanied by an export utility providing augmented synthetic sample images for easy drag-and-drop testing.

**Blocked by:** 05 (Deterministic Triage Engine & Exception Dossier Generator), 06 (Empirical Benchmark Suite).

**Status:** ready-for-agent

## Acceptance Criteria

- [ ] Add `streamlit` and `python-dotenv` to dependencies and load `GEMINI_API_KEY` securely from `.env` or system environment.
- [ ] Implement `src/ui/app.py` and root launcher `app.py` executed via `streamlit run app.py`.
- [ ] Provide dual document input modes:
  - File uploaders accepting front-face images for Unified National Card, Business License, and Tax Card.
  - 1-click Preset Scenario dropdown loading synthetic packages directly into the verification pipeline.
- [ ] Provide a sample generator utility (`scripts/generate_demo_samples.py`) that exports augmented synthetic document image sets (`clean`, `glare`, `blur`, `mismatch`) to a dedicated local directory (`demo_samples/`) for immediate drag-and-drop testing.
- [ ] Render 3-column side-by-side document preview with a toggleable checkbox to overlay diagnostic bounding boxes (red for failed Tier 1 fields/obscurations, amber for Tier 2/glare) directly on the images.
- [ ] Prominently display the final Triage Lifecycle outcome (`AUTO_PASS` in green, `HUMAN_ESCALATION` in amber, `HARD_MISMATCH` in red) along with overall confidence and worst Tier 1 confidence metrics.
- [ ] Multi-tab dossier viewer:
  - Tab 1: Actionable Arabic Exception Dossier with RTL styling (`dir="rtl"`).
  - Tab 2: Patronymic Name Chain alignment and slot-level discrepancy diff table.
  - Tab 3: Detailed audit log trail and raw extracted JSON schemas.
- [ ] Add automated test suite in `tests/test_ui.py` validating app initialization, preset execution, and bounding-box annotation rendering.
