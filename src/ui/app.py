"""Streamlit visual verification web dashboard for KYC Document Agent.

Provides dual document input modes (file upload and 1-click synthetic presets),
side-by-side 3-document preview with diagnostic bounding-box overlays,
live triage lifecycle outcome banners, and a multi-tab Arabic exception dossier.
"""

import html
import logging
import os
from typing import Any
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.benchmark.benchmark_suite import (
    BenchmarkScenario,
    generate_synthetic_benchmark_split,
)
from src.data.synthetic_generator import FONT_BOLD
from src.extraction.gemini_extractor import (
    GeminiMultimodalExtractor,
    ensure_ipv4_socket_resolution,
)
from src.triage.triage_engine import (
    FieldAnomaly,
    OnboardingDossier,
    TriageLifecycle,
    format_pair_label_en,
    verify_onboarding_package,
)

logger = logging.getLogger(__name__)

# Bounding box color constants
COLOR_TIER1_FAIL = (220, 38, 38)     # Bold Red
COLOR_TIER2_WARN = (217, 119, 6)     # Amber / Orange


def get_badge_font(size: int = 14) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load bold TrueType font from synthetic generator path with loud fallback."""
    try:
        return ImageFont.truetype(FONT_BOLD, size=size)
    except (OSError, ImportError):
        logger.warning("Arabic TrueType font not found at %s; using default font", FONT_BOLD)
        return ImageFont.load_default()


def annotate_document_image(
    image: Image.Image,
    anomalies: list[FieldAnomaly],
    doc_type: str,
) -> Image.Image:
    """Draw diagnostic bounding boxes and confidence badges on document image.

    Unconditionally maps normalized coordinates [ymin, xmin, ymax, xmax] in 0-1000 scale
    to actual pixel coordinates. Highlights Tier 1 failures in red and Tier 2 in amber.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    img_width, img_height = annotated.size
    font = get_badge_font(size=14)

    # Filter anomalies relevant to this document
    doc_anomalies = [a for a in anomalies if a.document_type == doc_type and a.bbox is not None]

    for anomaly in doc_anomalies:
        if anomaly.bbox is None:
            continue

        ymin, xmin, ymax, xmax = anomaly.bbox
        # Coordinates are guaranteed normalized in [0, 1000] scale by FieldAnomaly
        py_min = int(ymin * img_height / 1000)
        px_min = int(xmin * img_width / 1000)
        py_max = int(ymax * img_height / 1000)
        px_max = int(xmax * img_width / 1000)

        # Ensure valid box geometry
        py_min, py_max = min(py_min, py_max), max(py_min, py_max)
        px_min, px_max = min(px_min, px_max), max(px_min, px_max)

        color = COLOR_TIER1_FAIL if anomaly.tier == 1 else COLOR_TIER2_WARN
        line_width = 4 if anomaly.tier == 1 else 3

        draw.rectangle([px_min, py_min, px_max, py_max], outline=color, width=line_width)

        # Construct label badge text
        label = f"{anomaly.field_name}: {anomaly.confidence:.0%}"
        if anomaly.obscured:
            label += " [OBSCURED]"

        # Compute accurate badge bounding box using text metrics
        text_bbox = draw.textbbox((px_min + 4, 0), label, font=font)
        text_w = (text_bbox[2] - text_bbox[0]) + 8
        text_h = (text_bbox[3] - text_bbox[1]) + 4

        badge_y_min = max(0, py_min - text_h)
        badge_y_max = badge_y_min + text_h
        badge_x_max = min(img_width, px_min + text_w)

        draw.rectangle([px_min, badge_y_min, badge_x_max, badge_y_max], fill=color)
        draw.text((px_min + 4, badge_y_min + 2), label, fill=(255, 255, 255), font=font)

    return annotated


def load_preset_scenario(
    scenario: BenchmarkScenario,
    seed: int = 42,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    """Load a synthetic document package deterministically for the selected scenario."""
    packages = generate_synthetic_benchmark_split(sample_count_per_scenario=1, seed=seed)
    matching_packages = [p for p in packages if p.scenario == scenario]
    if not matching_packages:
        raise ValueError(f"No package found for scenario {scenario}")

    pkg = matching_packages[0]
    return (
        pkg.national_id_image,
        pkg.business_license_image,
        pkg.tax_card_image,
        pkg.ground_truth,
    )


def render_outcome_banner(dossier: OnboardingDossier) -> None:
    """Display prominent triage outcome banner with color-coding and metrics."""
    if dossier.lifecycle_outcome == TriageLifecycle.AUTO_PASS:
        bg_color = "#dcfce7"
        border_color = "#22c55e"
        text_color = "#15803d"
        title = "AUTO_PASS — Approved"
        description = "All onboarding criteria successfully passed. Mandatory Tier 1 fields meet or exceed 85% calibrated confidence and applicant identity aligns across all documents."
    elif dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION:
        bg_color = "#fef3c7"
        border_color = "#f59e0b"
        text_color = "#b45309"
        title = "HUMAN_ESCALATION — Manual Review Required"
        description = "Manual review required by an operations officer due to low field confidence or partial name discrepancy."
    else:
        bg_color = "#fee2e2"
        border_color = "#ef4444"
        text_color = "#b91c1c"
        title = "HARD_MISMATCH — Rejected (Identity Conflict)"
        description = "Critical irreconcilable identity discrepancy detected across submitted documents (name similarity < 70%)."

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-left: 8px solid {border_color};
            border-radius: 8px;
            padding: 18px 24px;
            margin-bottom: 20px;
            direction: ltr;
            text-align: left;
            font-family: sans-serif;
        ">
            <h2 style="color: {text_color}; margin: 0 0 8px 0; font-size: 22px;">{title}</h2>
            <p style="color: {text_color}; margin: 0; font-size: 15px; font-weight: 500;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Overall Confidence",
            value=f"{dossier.overall_confidence:.1%}",
        )
    with m2:
        t1_delta = "Passed (>= 85%)" if dossier.tier1_passed else "Deficient (< 85%)"
        st.metric(
            label="Worst Mandatory Field (Tier 1)",
            value=f"{dossier.min_tier1_confidence:.1%}",
            delta=t1_delta,
            delta_color="normal" if dossier.tier1_passed else "inverse",
        )
    with m3:
        match_val = f"{dossier.min_matching_score:.1%}" if dossier.min_matching_score is not None else "N/A"
        match_delta = "Full Match" if (dossier.min_matching_score or 0) >= 0.88 else "Review"
        st.metric(
            label="Min Name Match",
            value=match_val,
            delta=match_delta if dossier.min_matching_score is not None else None,
        )
    with m4:
        anomaly_count = len(dossier.tier1_anomalies)
        st.metric(
            label="Tier 1 Anomalies",
            value=str(anomaly_count),
            delta="Zero Defects" if anomaly_count == 0 else f"{anomaly_count} Defect{'s' if anomaly_count > 1 else ''}",
            delta_color="normal" if anomaly_count == 0 else "inverse",
        )


def main() -> None:
    """Streamlit application main entrypoint."""
    load_dotenv()
    ensure_ipv4_socket_resolution()
    st.set_page_config(
        page_title="KYC Document Verification Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        [data-testid="stHeaderActionElements"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("KYC Document Verification Agent")
    st.markdown(
        "**Automated optical verification and compliance triage platform for Iraqi onboarding documents (Unified National Card, Business License, and Tax Card).**"
    )

    # 1. Sidebar Controls
    with st.sidebar:
        st.header("System Settings")

        env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        has_env_key = bool(env_api_key)

        # Zero Secret Echoing: Leave input field blank by default, never echo actual secret into client DOM
        custom_key = st.text_input(
            "Gemini API Key (Override)",
            value="",
            type="password",
            placeholder="Enter key to override .env..." if has_env_key else "Enter API key...",
            help="Leave blank to use GEMINI_API_KEY loaded securely from environment or .env file.",
        )
        api_key = custom_key.strip() if custom_key.strip() else env_api_key
        has_active_key = bool(api_key)

        if custom_key.strip():
            st.success("Manual API Key Active (Live API Mode)")
        elif has_env_key:
            st.success("Environment API Key Active (Live API Mode)")
        else:
            st.error("No API Key detected. Live Gemini API is strictly required.")

        st.divider()
        st.subheader("Document Input Mode")
        input_mode = st.radio(
            "Choose Input Source:",
            options=["1-Click Preset Scenario", "Upload Custom Documents"],
            index=0,
        )

        preset_scenario: BenchmarkScenario | None = None
        uploaded_nid = None
        uploaded_biz = None
        uploaded_tax = None

        if "1-Click" in input_mode:
            scenario_name = st.selectbox(
                "Select Test Scenario:",
                options=[
                    "Clean Baseline — Auto-Pass",
                    "Specular Glare — Human Escalation",
                    "Gaussian Blur — Human Escalation",
                    "Perspective Tilt — Auto-Pass",
                    "Identity Mismatch — Hard Mismatch",
                ],
            )
            scenario_map = {
                "Clean Baseline — Auto-Pass": BenchmarkScenario.CLEAN,
                "Specular Glare — Human Escalation": BenchmarkScenario.SPECULAR_GLARE,
                "Gaussian Blur — Human Escalation": BenchmarkScenario.BLURRED,
                "Perspective Tilt — Auto-Pass": BenchmarkScenario.TILTED,
                "Identity Mismatch — Hard Mismatch": BenchmarkScenario.IDENTITY_MISMATCH,
            }
            preset_scenario = scenario_map[scenario_name]
        else:
            st.markdown("**Upload the three document images (PNG / JPG / JPEG):**")
            uploaded_nid = st.file_uploader("1. Unified National Card", type=["png", "jpg", "jpeg"])
            uploaded_biz = st.file_uploader("2. Business License / Commercial Registry", type=["png", "jpg", "jpeg"])
            uploaded_tax = st.file_uploader("3. Tax Card (TIN)", type=["png", "jpg", "jpeg"])

        st.divider()
        run_btn = st.button("Run Verification", type="primary", use_container_width=True)

    # 2. Acquire Working Images with Input Signature Tracking
    img_nid: Image.Image | None = None
    img_biz: Image.Image | None = None
    img_tax: Image.Image | None = None
    ground_truth: dict[str, Any] = {}

    if "1-Click" in input_mode and preset_scenario is not None:
        active_input_id = f"preset_{preset_scenario.value}"
        img_nid, img_biz, img_tax, ground_truth = load_preset_scenario(preset_scenario)
    else:
        nid_sig = f"{uploaded_nid.name}_{uploaded_nid.size}" if uploaded_nid else "none"
        biz_sig = f"{uploaded_biz.name}_{uploaded_biz.size}" if uploaded_biz else "none"
        tax_sig = f"{uploaded_tax.name}_{uploaded_tax.size}" if uploaded_tax else "none"
        active_input_id = f"upload_{nid_sig}_{biz_sig}_{tax_sig}"

        if uploaded_nid and uploaded_biz and uploaded_tax:
            try:
                img_nid = Image.open(uploaded_nid).convert("RGB")
                img_biz = Image.open(uploaded_biz).convert("RGB")
                img_tax = Image.open(uploaded_tax).convert("RGB")
            except (Image.UnidentifiedImageError, OSError) as err:
                st.error(f"Cannot read uploaded image files: {err}")
                return

    # Check for input selection changes to prevent stale results
    evaluated_input_id = st.session_state.get("evaluated_input_id")
    is_input_changed = evaluated_input_id is not None and evaluated_input_id != active_input_id

    if is_input_changed:
        st.session_state.pop("dossier", None)
        st.session_state.pop("current_images", None)

    # 3. Pipeline Execution & State Management
    if run_btn:
        if not has_active_key:
            st.error("Gemini API key is required to run verification. Please enter an API key in the sidebar or in .env.")
            return

        if img_nid is None or img_biz is None or img_tax is None:
            st.error("Please upload all three documents or select a preset scenario to proceed.")
            return

        with st.spinner("Extracting document data and verifying compliance via Gemini API..."):
            extractor = GeminiMultimodalExtractor(api_key=api_key)

            dossier = verify_onboarding_package(
                national_id_image=img_nid,
                business_license_image=img_biz,
                tax_card_image=img_tax,
                extractor=extractor,
            )
            st.session_state["dossier"] = dossier
            st.session_state["current_images"] = (img_nid, img_biz, img_tax)
            st.session_state["evaluated_input_id"] = active_input_id
            is_input_changed = False

    # 4. Display Results
    dossier: OnboardingDossier | None = st.session_state.get("dossier")
    current_images = st.session_state.get("current_images")

    if current_images is not None and not is_input_changed:
        disp_nid, disp_biz, disp_tax = current_images
    else:
        disp_nid, disp_biz, disp_tax = img_nid, img_biz, img_tax

    # Side-by-side Document Previews
    st.subheader("Side-by-Side Document Previews")
    if disp_nid is not None and disp_biz is not None and disp_tax is not None:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**1. Unified National Card (National ID)**")
            st.image(disp_nid, use_container_width=True)
        with c2:
            st.markdown("**2. Business License**")
            st.image(disp_biz, use_container_width=True)
        with c3:
            st.markdown("**3. Tax Card (TIN)**")
            st.image(disp_tax, use_container_width=True)
    else:
        st.info("Select a test scenario or upload documents from the sidebar to begin.")

    if is_input_changed:
        st.info("Document input changed. Click 'Run Verification' to evaluate the new package.")

    if dossier is not None and not is_input_changed:
        if dossier.extraction_package is not None:
            pkg = dossier.extraction_package
            api_errors = []
            for doc_title, doc_res in [
                ("Unified National Card", pkg.national_id),
                ("Business License", pkg.business_license),
                ("Tax Card", pkg.tax_card),
            ]:
                if doc_res.audit.retry_error and "WorkerException" in doc_res.audit.retry_error:
                    api_errors.append(f"{doc_title}: {doc_res.audit.retry_error}")
            if api_errors:
                st.error("Warning: Document extraction failed on some documents due to an upstream API error:")
                for err in api_errors:
                    st.caption(err)

        st.divider()
        render_outcome_banner(dossier)

        # Multi-tab dossier viewer
        tab1, tab2, tab3 = st.tabs([
            "Verification & Audit Dossier",
            "Patronymic Name Alignment",
            "Audit Trail & Raw Schemas",
        ])

        with tab1:
            st.markdown(dossier.to_markdown_report_en(), unsafe_allow_html=False)

        with tab2:
            st.markdown("### Cross-Document Patronymic Name Alignment Analysis")
            if dossier.name_mismatches:
                slot_names_en = {
                    "given": "Given Name",
                    "father": "Father's Name",
                    "grandfather": "Grandfather's Name",
                    "surname": "Surname / Clan",
                }
                for mismatch in dossier.name_mismatches:
                    pair_label = html.escape(format_pair_label_en(mismatch.pair_key, mismatch.doc_a_name, mismatch.doc_b_name))
                    safe_name_a = html.escape(mismatch.raw_name_a)
                    safe_name_b = html.escape(mismatch.raw_name_b)

                    with st.expander(
                        f"Comparison: {pair_label} (Similarity: {mismatch.similarity_score:.1%})",
                        expanded=True,
                    ):
                        st.markdown(f"**Name in First Document:** `{safe_name_a}`")
                        st.markdown(f"**Name in Second Document:** `{safe_name_b}`")
                        st.markdown(f"**Triage Classification:** `{mismatch.triage_band.value}`")

                        if mismatch.token_details:
                            st.markdown("**Detailed Patronymic Slot Breakdown:**")
                            slot_rows = []
                            for slot in mismatch.token_details:
                                if slot.similarity >= 0.88:
                                    status_icon = "[MATCH]"
                                elif slot.similarity >= 0.70:
                                    status_icon = "[REVIEW]"
                                else:
                                    status_icon = "[MISMATCH]"
                                slot_rows.append({
                                    "Status": status_icon,
                                    "Patronymic Slot": slot_names_en.get(slot.role.value, slot.role.value.title()),
                                    "Name in (A)": slot.token_a or "—",
                                    "Name in (B)": slot.token_b or "—",
                                    "Similarity": f"{slot.similarity:.1%}",
                                    "Weight": f"{slot.weight:.2f}",
                                })
                            st.table(slot_rows)

                        if mismatch.audit_notes:
                            st.markdown("##### Inspection Notes:")
                            for note in mismatch.audit_notes:
                                st.markdown(f"- {note}")
            else:
                st.success("Complete identity match across all documents with zero discrepancies.")

        with tab3:
            st.markdown("### Chronological System Audit Trail")
            for entry in dossier.audit_trail:
                st.markdown(f"- {entry}")

            st.divider()
            st.markdown("### Extracted Raw Schemas (JSON)")
            if dossier.extraction_package:
                pkg = dossier.extraction_package
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Unified National Card Schema**")
                    st.json(pkg.national_id.schema.model_dump())
                with col_b:
                    st.markdown("**Business License Schema**")
                    st.json(pkg.business_license.schema.model_dump())
                with col_c:
                    st.markdown("**Tax Card Schema**")
                    st.json(pkg.tax_card.schema.model_dump())
            else:
                st.info("Raw schema data is not available for the current evaluation package.")


if __name__ == "__main__":
    main()
