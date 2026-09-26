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
    SimulatedBenchmarkExtractor,
    generate_synthetic_benchmark_split,
)
from src.data.synthetic_generator import FONT_BOLD
from src.extraction.gemini_extractor import GeminiMultimodalExtractor
from src.triage.triage_engine import (
    FieldAnomaly,
    OnboardingDossier,
    TriageLifecycle,
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
        title = "AUTO_PASS - قبول تلقائي"
        description = "تم اجتياز كافة معايير الفحص بنجاح. الحقول الأساسية موثوقة بنسبة تفوق 85% والأسماء متطابقة عبر الوثائق."
    elif dossier.lifecycle_outcome == TriageLifecycle.HUMAN_ESCALATION:
        bg_color = "#fef3c7"
        border_color = "#f59e0b"
        text_color = "#b45309"
        title = "HUMAN_ESCALATION - مراجعة يدوية مطلوبة"
        description = "يتطلب الملف مراجعة موظف العمليات المختص بسبب تدني ثقة حقول أساسية أو تباين جزئي في الاسم."
    else:
        bg_color = "#fee2e2"
        border_color = "#ef4444"
        text_color = "#b91c1c"
        title = "HARD_MISMATCH - رفض بسبب عدم تطابق جوهري"
        description = "تم رصد اختلاف جوهري غير قابل للتسوية في الهوية عبر الوثائق المقدمة (نسبة التطابق أقل من 70%)."

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-right: 8px solid {border_color};
            border-radius: 8px;
            padding: 18px 24px;
            margin-bottom: 20px;
            direction: rtl;
            text-align: right;
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
            label="معدل الثقة الإجمالي (Overall Confidence)",
            value=f"{dossier.overall_confidence:.1%}",
        )
    with m2:
        t1_delta = "ناجح (>= 85%)" if dossier.tier1_passed else "متعثر (< 85%)"
        st.metric(
            label="أدنى ثقة حقل أساسي (Worst Tier 1)",
            value=f"{dossier.min_tier1_confidence:.1%}",
            delta=t1_delta,
            delta_color="normal" if dossier.tier1_passed else "inverse",
        )
    with m3:
        match_val = f"{dossier.min_matching_score:.1%}" if dossier.min_matching_score is not None else "N/A"
        match_delta = "تطابق تام" if (dossier.min_matching_score or 0) >= 0.88 else "مراجعة"
        st.metric(
            label="أدنى تطابق أسماء (Min Name Match)",
            value=match_val,
            delta=match_delta if dossier.min_matching_score is not None else None,
        )
    with m4:
        anomaly_count = len(dossier.tier1_anomalies)
        st.metric(
            label="الحقول الأساسية المتعثرة (Tier 1 Anomalies)",
            value=str(anomaly_count),
            delta="خال من العيوب" if anomaly_count == 0 else f"{anomaly_count} عيب",
            delta_color="normal" if anomaly_count == 0 else "inverse",
        )


def main() -> None:
    """Streamlit application main entrypoint."""
    load_dotenv()
    st.set_page_config(
        page_title="ZainCash KYC Document Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Scoped RTL Stylesheet: targets tab panels safely without requiring unsafe_allow_html on user text
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
        }
        .stTabs [data-baseweb="tab-panel"] table {
            direction: rtl;
            text-align: right;
            width: 100%;
        }
        .stTabs [data-baseweb="tab-panel"] th,
        .stTabs [data-baseweb="tab-panel"] td {
            text-align: right !important;
            padding: 8px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("ZainCash KYC Document Verification Agent")
    st.markdown(
        "**منظومة التدقيق والتحقق البصري لوثائق الانضمام - البطاقة الوطنية الموحدة، إجازة المهنة، والهوية الضريبية.**"
    )

    # 1. Sidebar Controls
    with st.sidebar:
        st.header("إعدادات المنظومة (System Settings)")

        env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        has_env_key = bool(env_api_key)

        # Zero Secret Echoing: Leave input field blank by default, never echo actual secret into client DOM
        custom_key = st.text_input(
            "Gemini API Key (Override)",
            value="",
            type="password",
            placeholder="Enter key to override .env..." if has_env_key else "AIzaSy...",
            help="Leave blank to use GEMINI_API_KEY loaded securely from environment or .env file.",
        )
        api_key = custom_key.strip() if custom_key.strip() else env_api_key
        has_active_key = bool(api_key)

        if custom_key.strip():
            st.success("Manual API Key Override Active")
        elif has_env_key:
            st.success("Key Loaded Securely from Environment")
        else:
            st.info("No API Key: Offline Simulation Mode Enabled")

        st.divider()
        st.subheader("طريقة إدخال الوثائق (Input Mode)")
        input_mode = st.radio(
            "اختر المصدر:",
            options=["1-Click Preset Scenario (حالات تجريبية جاهزة)", "Upload Custom Documents (رفع وثائق)"],
            index=0,
        )

        preset_scenario: BenchmarkScenario | None = None
        uploaded_nid = None
        uploaded_biz = None
        uploaded_tax = None

        if "1-Click" in input_mode:
            scenario_name = st.selectbox(
                "اختر الحالة الاختبارية:",
                options=[
                    "Clean Baseline - Auto-Pass (حزمة نظامية نظيفة)",
                    "Specular Glare - Human Escalation (انعكاس ضوئي ساطع)",
                    "Gaussian Blur - Human Escalation (ضبابية بصرية)",
                    "Perspective Tilt - Auto-Pass (انحراف زاوي مقروء)",
                    "Identity Mismatch - Hard Mismatch (عدم تطابق في الهوية)",
                ],
            )
            scenario_map = {
                "Clean Baseline - Auto-Pass (حزمة نظامية نظيفة)": BenchmarkScenario.CLEAN,
                "Specular Glare - Human Escalation (انعكاس ضوئي ساطع)": BenchmarkScenario.SPECULAR_GLARE,
                "Gaussian Blur - Human Escalation (ضبابية بصرية)": BenchmarkScenario.BLURRED,
                "Perspective Tilt - Auto-Pass (انحراف زاوي مقروء)": BenchmarkScenario.TILTED,
                "Identity Mismatch - Hard Mismatch (عدم تطابق في الهوية)": BenchmarkScenario.IDENTITY_MISMATCH,
            }
            preset_scenario = scenario_map[scenario_name]
        else:
            st.markdown("**ارفع صور الوثائق الثلاث (JPG / PNG):**")
            uploaded_nid = st.file_uploader("1. البطاقة الوطنية الموحدة", type=["png", "jpg", "jpeg"])
            uploaded_biz = st.file_uploader("2. إجازة ممارسة المهنة", type=["png", "jpg", "jpeg"])
            uploaded_tax = st.file_uploader("3. الهوية الضريبية", type=["png", "jpg", "jpeg"])

        st.divider()
        st.subheader("خيارات العرض التشخيصي")
        show_bboxes = st.checkbox("إظهار مربعات الإحاطة التشخيصية (Overlay Bounding Boxes)", value=True)
        force_simulation = st.checkbox("تشغيل وضع المحاكاة السريع (Force Offline Simulator)", value=not has_active_key)

        run_btn = st.button("تشغيل الفحص والتحقق (Run Verification)", type="primary", use_container_width=True)

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
                st.error(f"يتعذر قراءة ملفات الصور المرفوعة: {err}")
                return

    # Check for input selection changes to prevent stale results
    evaluated_input_id = st.session_state.get("evaluated_input_id")
    is_input_changed = evaluated_input_id is not None and evaluated_input_id != active_input_id

    if is_input_changed:
        st.session_state.pop("dossier", None)
        st.session_state.pop("current_images", None)

    # 3. Pipeline Execution & State Management
    if run_btn:
        if img_nid is None or img_biz is None or img_tax is None:
            st.error("يرجى تحميل جميع الوثائق الثلاث أو اختيار حالة تجريبية جاهزة للمتابعة.")
            return

        with st.spinner("جاري استخراج البيانات وفحص التوافقية ومطابقة الهوية عبر الوثائق..."):
            extractor = None
            if force_simulation or not has_active_key:
                extractor = SimulatedBenchmarkExtractor()
            else:
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

    # 4. Display Results
    dossier: OnboardingDossier | None = st.session_state.get("dossier")
    current_images = st.session_state.get("current_images")

    if current_images is not None and not is_input_changed:
        disp_nid, disp_biz, disp_tax = current_images
    else:
        disp_nid, disp_biz, disp_tax = img_nid, img_biz, img_tax

    # Side-by-side Document Previews
    st.subheader("معاينة الوثائق المفحوصة (Side-by-Side Documents)")
    if disp_nid is not None and disp_biz is not None and disp_tax is not None:
        c1, c2, c3 = st.columns(3)

        anomalies = dossier.tier1_anomalies + dossier.tier2_warnings if dossier else []

        if show_bboxes and dossier and not is_input_changed:
            annotated_nid = annotate_document_image(disp_nid, anomalies, "national_id")
            annotated_biz = annotate_document_image(disp_biz, anomalies, "business_license")
            annotated_tax = annotate_document_image(disp_tax, anomalies, "tax_card")
        else:
            annotated_nid = disp_nid
            annotated_biz = disp_biz
            annotated_tax = disp_tax

        with c1:
            st.markdown("##### 1. البطاقة الوطنية الموحدة (National ID)")
            st.image(annotated_nid, use_container_width=True)
        with c2:
            st.markdown("##### 2. إجازة ممارسة المهنة (Business License)")
            st.image(annotated_biz, use_container_width=True)
        with c3:
            st.markdown("##### 3. الهوية الضريبية (Tax Card)")
            st.image(annotated_tax, use_container_width=True)
    else:
        st.info("اختر حالة اختبارية أو قم برفع الوثائق من القائمة الجانبية لبدء الفحص.")

    if is_input_changed:
        st.info("تم تغيير إدخال الوثائق. اضغط على زر 'تشغيل الفحص والتحقق' لفحص الحزمة الجديدة.")

    if dossier is not None and not is_input_changed:
        st.divider()
        render_outcome_banner(dossier)

        # Multi-tab dossier viewer
        tab1, tab2, tab3 = st.tabs([
            "تقرير التدقيق والاستثناءات (Arabic Dossier)",
            "مطابقة سلسلة الأسماء (Patronymic Alignment)",
            "سجل التدقيق والبيانات التقنية (Audit Trail & JSON)",
        ])

        with tab1:
            # Safe Native Markdown Rendering without unsafe_allow_html
            st.markdown(dossier.to_markdown_report(), unsafe_allow_html=False)

        with tab2:
            st.markdown("### تحليل مطابقة الاسم الرباعي واللقب عبر الوثائق")
            if dossier.name_mismatches:
                for mismatch in dossier.name_mismatches:
                    safe_doc_a = html.escape(mismatch.doc_a_name)
                    safe_doc_b = html.escape(mismatch.doc_b_name)
                    safe_name_a = html.escape(mismatch.raw_name_a)
                    safe_name_b = html.escape(mismatch.raw_name_b)

                    with st.expander(
                        f"مقارنة: {safe_doc_a} - {safe_doc_b} (نسبة التشابه: {mismatch.similarity_score:.1%})",
                        expanded=True,
                    ):
                        st.markdown(f"**الاسم في الوثيقة الأولى:** `{safe_name_a}`")
                        st.markdown(f"**الاسم في الوثيقة الثانية:** `{safe_name_b}`")
                        st.markdown(f"**تصنيف المطابقة:** `{mismatch.triage_band.value}`")

                        if mismatch.token_details:
                            st.markdown("##### جدول تفصيل الأجزاء الأربعة (Patronymic Slots Breakdown):")
                            slot_rows = []
                            for slot in mismatch.token_details:
                                if slot.similarity >= 0.88:
                                    status_icon = "[MATCH]"
                                elif slot.similarity >= 0.70:
                                    status_icon = "[REVIEW]"
                                else:
                                    status_icon = "[MISMATCH]"
                                slot_rows.append({
                                    "الحالة": status_icon,
                                    "الجزء": slot.role.value,
                                    "الاسم في (أ)": slot.token_a or "—",
                                    "الاسم في (ب)": slot.token_b or "—",
                                    "نسبة التشابه": f"{slot.similarity:.1%}",
                                    "وزن الجزء": f"{slot.weight:.2f}",
                                })
                            st.table(slot_rows)

                        if mismatch.audit_notes_ar:
                            st.markdown("##### ملاحظات الفحص:")
                            for note in mismatch.audit_notes_ar:
                                st.markdown(f"- {note}")
            else:
                st.success("تطابق كامل في الأسماء عبر كافة الوثائق دون تسجيل أي تباين.")

        with tab3:
            st.markdown("### السجل الكامل لعمليات الفحص (Chronological Audit Trail)")
            for entry in dossier.audit_trail_ar:
                st.markdown(f"- {entry}")

            st.divider()
            st.markdown("### البيانات المستخرجة الخام (Extracted JSON Schemas)")
            if dossier.extraction_package:
                pkg = dossier.extraction_package
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**National ID Schema**")
                    st.json(pkg.national_id.schema.model_dump())
                with col_b:
                    st.markdown("**Business License Schema**")
                    st.json(pkg.business_license.schema.model_dump())
                with col_c:
                    st.markdown("**Tax Card Schema**")
                    st.json(pkg.tax_card.schema.model_dump())
            else:
                st.info("البيانات الخام غير متوفرة لحزمة الفحص الحالية.")


if __name__ == "__main__":
    main()
