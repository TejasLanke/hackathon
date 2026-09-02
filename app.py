
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import CLASS_DISPLAY_NAMES, DEVICE, MODEL_NAME, MODEL_PATH
from src.gradcam import generate_gradcam
from src.inference import predict
from src.model import load_model


# ============================================================
# RETINAX — PREMIUM STREAMLIT UI
# UI layer only: existing ML / inference modules are preserved.
# ============================================================

st.set_page_config(
    page_title="RetinaX | DR Screening",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- CSS ------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --ink: #102033;
    --muted: #6d7d90;
    --line: #e5ebf2;
    --blue: #246bfd;
    --blue-dark: #1553d6;
    --cyan: #0ea5c7;
    --green: #0e9f6e;
    --amber: #d98a00;
    --red: #d9534f;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 95% 0%, rgba(36,107,253,.07), transparent 22%),
        radial-gradient(circle at 0% 45%, rgba(14,165,199,.045), transparent 25%),
        var(--bg);
    color: var(--ink);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: rgba(244,247,251,.88);
}
div[data-testid="stToolbar"] { visibility: hidden; }

.block-container {
    max-width: 1540px;
    padding: 1.25rem 2.1rem 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c1a2b;
    border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] > div {
    padding: 1.2rem 1rem;
}
section[data-testid="stSidebar"] * {
    color: #d9e5f2 !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #8ea4bb !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.08);
}
.side-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 4px 18px;
}
.side-logo {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: linear-gradient(135deg,#2b73ff,#13b6c8);
    color: white !important;
    font-size: 21px;
    box-shadow: 0 8px 24px rgba(36,107,253,.25);
}
.side-title {
    color: white !important;
    font: 800 19px "Manrope", sans-serif;
}
.side-subtitle {
    color: #8fa4bb !important;
    font-size: 11px;
    margin-top: 1px;
}
.side-section {
    color: #718ba5 !important;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: 10px;
    font-weight: 800;
    margin: 19px 3px 9px;
}
.side-step {
    padding: 9px 10px;
    margin: 4px 0;
    border-radius: 9px;
    color: #a8bbcf !important;
    font-size: 13px;
}
.side-step.active {
    background: rgba(36,107,253,.15);
    color: #eaf2ff !important;
}
.side-status {
    padding: 11px 12px;
    margin-top: 7px;
    border-radius: 12px;
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.07);
    font-size: 12px;
}
.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c993;
    margin-right: 7px;
}

/* Header */
.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.crumb {
    color: #7a8999;
    font-size: 12px;
    font-weight: 600;
}
.secure {
    color: #177d61;
    background: #e8f8f2;
    border: 1px solid #c9eee1;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 31px 36px;
    color: white;
    background:
        radial-gradient(circle at 90% 20%, rgba(75,184,255,.20), transparent 26%),
        linear-gradient(118deg, #0d2239 0%, #102c4d 52%, #123b5a 100%);
    box-shadow: 0 18px 48px rgba(16,42,70,.16);
}
.hero:after {
    content: "◉";
    position: absolute;
    right: 55px;
    bottom: -55px;
    font-size: 190px;
    color: rgba(255,255,255,.025);
}
.hero-kicker {
    color: #72b9ff;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: 10px;
    font-weight: 800;
}
.hero h1 {
    font: 800 clamp(27px, 3vw, 42px)/1.08 "Manrope", sans-serif;
    margin: 9px 0 10px;
    letter-spacing: -.025em;
}
.hero p {
    max-width: 760px;
    color: #b8cadc;
    font-size: 14px;
    line-height: 1.65;
    margin: 0;
}
.hero-pills {
    display: flex;
    gap: 8px;
    margin-top: 19px;
    flex-wrap: wrap;
}
.hero-pill {
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    color: #dcecff;
    font-size: 10px;
    font-weight: 700;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 28px rgba(23,42,65,.045);
}
.card-title {
    font: 700 15px "Manrope", sans-serif;
    color: var(--ink);
}
.card-sub {
    color: var(--muted);
    font-size: 11px;
    margin-top: 4px;
    line-height: 1.5;
}
.section-title {
    font: 800 20px "Manrope", sans-serif;
    color: var(--ink);
    margin: 25px 0 11px;
}
.step-number {
    display: inline-grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: #eaf1ff;
    color: var(--blue);
    font-size: 11px;
    font-weight: 800;
    margin-right: 7px;
}

/* Upload */
div[data-testid="stFileUploader"] {
    background: #f9fbfe;
    border: 1.5px dashed #c9d8e8;
    border-radius: 15px;
    padding: 7px;
}
div[data-testid="stFileUploader"] section {
    border: 0;
    background: transparent;
}
div[data-testid="stFileUploader"] button {
    border-radius: 9px;
}

/* Buttons */
div.stButton > button {
    border-radius: 11px;
    min-height: 45px;
    font-weight: 800;
    border: 0;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#246bfd,#1553d6);
    box-shadow: 0 8px 20px rgba(36,107,253,.20);
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#1e61ed,#124bc4);
}

/* Result */
.result-hero {
    border-radius: 19px;
    padding: 21px;
    background: linear-gradient(135deg,#ffffff,#f7fbff);
    border: 1px solid #dfe9f5;
    box-shadow: 0 10px 30px rgba(23,42,65,.05);
}
.result-kicker {
    color: #7a8da2;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: 9px;
    font-weight: 800;
}
.result-name {
    color: #102033;
    font: 800 25px "Manrope", sans-serif;
    margin: 7px 0 4px;
}
.result-confidence {
    color: #246bfd;
    font: 800 34px "Manrope", sans-serif;
}
.result-note {
    color: #708197;
    font-size: 11px;
}
.status-good {
    display: inline-block;
    color: #087a59;
    background: #e9f8f3;
    border: 1px solid #ccefe3;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
}
.status-demo {
    display: inline-block;
    color: #946000;
    background: #fff6df;
    border: 1px solid #f5e3af;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
}
.small-stat {
    background: #f7f9fc;
    border: 1px solid #e8edf3;
    border-radius: 12px;
    padding: 11px;
}
.small-stat-label {
    color: #7d8da0;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 800;
}
.small-stat-value {
    color: #1a2c40;
    font-size: 14px;
    font-weight: 800;
    margin-top: 3px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 700;
}
div[data-baseweb="tab-list"] {
    gap: 4px;
}

/* Disclaimer */
.disclaimer {
    margin-top: 22px;
    padding: 12px 14px;
    border-radius: 12px;
    background: #fff9e9;
    border: 1px solid #f2e2b6;
    color: #795d1d;
    font-size: 10px;
    line-height: 1.55;
}
.footer {
    text-align: center;
    color: #93a0af;
    font-size: 10px;
    padding-top: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------- Sidebar --------------------------
model_exists = Path(MODEL_PATH).exists()

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-logo">◉</div>
            <div>
                <div class="side-title">RetinaX</div>
                <div class="side-subtitle">Explainable AI Screening</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section">Screening workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step active">01 &nbsp; Upload fundus image</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step">02 &nbsp; Run AI analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step">03 &nbsp; Review screening report</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step">04 &nbsp; Inspect model explanation</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-section">System status</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="side-status"><span class="dot"></span>
            {"Trained checkpoint detected" if model_exists else "Demo mode — checkpoint not found"}
        </div>
        <div class="side-status">🧠 &nbsp; Model &nbsp; <b>{MODEL_NAME}</b></div>
        <div class="side-status">⚙️ &nbsp; Device &nbsp; <b>{str(DEVICE)}</b></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section">About</div>', unsafe_allow_html=True)
    st.caption(
        "Hackathon prototype for explainable diabetic retinopathy screening. "
        "Designed as an AI-assisted decision-support interface."
    )


# ------------------------- Header ----------------------------
st.markdown(
    """
    <div class="topline">
        <div class="crumb">Workspace / Retinal screening</div>
        <div class="secure">● Local processing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">RETINAX · EXPLAINABLE AI</div>
        <h1>Diabetic Retinopathy<br>Screening Dashboard</h1>
        <p>
            Analyze a retinal fundus image with the project's existing AI pipeline,
            review the prediction and confidence distribution, and inspect a
            Grad-CAM visual explanation of the model output.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">AI-assisted screening</span>
            <span class="hero-pill">Confidence analysis</span>
            <span class="hero-pill">Grad-CAM explanation</span>
            <span class="hero-pill">Human review required</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------- Upload ----------------------------
st.markdown(
    '<div class="section-title"><span class="step-number">01</span>Upload retinal image</div>',
    unsafe_allow_html=True,
)

upload_col, info_col = st.columns([1.55, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose a fundus photograph",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
        label_visibility="collapsed",
    )

with info_col:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Before you analyze</div>
            <div class="card-sub">
                Use a clear fundus image with the retina visible. The image is
                passed directly to the project's existing preprocessing and
                inference pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if uploaded_file is None:
    st.markdown(
        """
        <div class="card" style="margin-top:13px;">
            <div class="card-title">Ready when you are</div>
            <div class="card-sub">
                Upload an image above. The analysis dashboard will appear after
                you explicitly start the screening.
            </div>
        </div>
        <div class="disclaimer">
            <b>Important:</b> RetinaX is an educational/hackathon prototype.
            It is not a medical diagnosis and must not replace assessment by a
            qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footer">RETINAX · Explainable AI Hackathon Prototype</div>',
        unsafe_allow_html=True,
    )
    st.stop()


image = Image.open(uploaded_file).convert("RGB")


# ------------------------- Preview ---------------------------
st.markdown(
    '<div class="section-title"><span class="step-number">02</span>Review & analyze</div>',
    unsafe_allow_html=True,
)

preview_col, details_col = st.columns([1.45, 1], gap="large")

with preview_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(
        image,
        caption=f"{uploaded_file.name}  ·  {image.width} × {image.height}px",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with details_col:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Image quality & metadata</div>
            <div class="card-sub">Input received by the screening workflow.</div>
            <br>
            <div class="small-stat">
                <div class="small-stat-label">File</div>
                <div class="small-stat-value">{uploaded_file.name}</div>
            </div>
            <br>
            <div class="small-stat">
                <div class="small-stat-label">Resolution</div>
                <div class="small-stat-value">{image.width} × {image.height} px</div>
            </div>
            <br>
            <div class="small-stat">
                <div class="small-stat-label">Format</div>
                <div class="small-stat-value">{uploaded_file.type or "Image"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    analyze = st.button(
        "🔍  Run AI screening",
        type="primary",
        use_container_width=True,
    )


if not analyze:
    st.info("Image loaded successfully. Press **Run AI screening** to continue.")
    st.stop()


# ------------------------- Inference ------------------------
with st.spinner("Analyzing image · preprocessing → inference → explanation"):
    model = load_model()
    result = predict(image, model=model)

    try:
        explanation = generate_gradcam(model, image, result["class_index"])
        explanation_error = None
    except Exception as exc:
        explanation = None
        explanation_error = str(exc)


# ------------------------- Report ---------------------------
st.markdown(
    '<div class="section-title"><span class="step-number">03</span>Screening report</div>',
    unsafe_allow_html=True,
)

if result["is_demo"]:
    st.warning(
        "Demo mode is active because a trained checkpoint was not loaded. "
        "The output is illustrative and must not be interpreted as a clinical prediction."
    )

label = result["label"]
confidence = float(result["confidence"])
class_name = result["class"]
class_index = int(result["class_index"])

mode_badge = (
    '<span class="status-demo">DEMO / ILLUSTRATIVE</span>'
    if result["is_demo"]
    else '<span class="status-good">TRAINED MODEL</span>'
)

r1, r2, r3 = st.columns([1.25, .85, .85], gap="medium")

with r1:
    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-kicker">Primary screening result</div>
            <div class="result-name">{label}</div>
            <div class="result-confidence">{confidence:.1%}</div>
            <div class="result-note">Model confidence for the selected class</div>
            <br>
            {mode_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

with r2:
    st.markdown(
        f"""
        <div class="card" style="height:100%;">
            <div class="card-title">Predicted class</div>
            <div class="card-sub">Internal model class</div>
            <br>
            <div style="font:800 23px Manrope;color:#102033;">{class_name}</div>
            <div class="card-sub">Class index {class_index}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r3:
    st.markdown(
        f"""
        <div class="card" style="height:100%;">
            <div class="card-title">Model</div>
            <div class="card-sub">Architecture used by project</div>
            <br>
            <div style="font:800 18px Manrope;color:#102033;">EfficientNet-B0</div>
            <div class="card-sub">Device: {DEVICE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------- Detailed tabs --------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Probability analysis", "🔥 Explainability", "⚙️ Technical details"]
)

with tab1:
    st.markdown("#### Class probability distribution")
    st.caption(
        "The probabilities below describe the model output for the two classes. "
        "They do not represent disease severity."
    )

    for key, probability in result["probabilities"].items():
        probability = float(probability)
        display_name = CLASS_DISPLAY_NAMES.get(
            key,
            "Diabetic Retinopathy" if key == "dr" else "No Diabetic Retinopathy",
        )
        a, b = st.columns([4.7, 1], gap="medium")
        with a:
            st.write(f"**{display_name}**")
            st.progress(max(0.0, min(1.0, probability)))
        with b:
            st.markdown(
                f"<div style='text-align:right;font:800 18px Manrope;color:#246bfd;padding-top:4px'>{probability:.1%}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### AI summary")
    if result["is_demo"]:
        summary = (
            "The project is currently using its deterministic demo fallback because "
            "the trained checkpoint is unavailable."
        )
    else:
        summary = (
            f"The loaded model selected <b>{label}</b> with a confidence of "
            f"<b>{confidence:.1%}</b>. This output should be reviewed by a qualified "
            "healthcare professional before any clinical decision."
        )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Interpretation</div>
            <div class="card-sub">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build a clean, reproducible report using the existing result contract.
    generated = datetime.now().strftime("%d %B %Y, %I:%M %p")
    probability_lines = "\n".join(
        f"- {CLASS_DISPLAY_NAMES.get(k, k)}: {float(v):.2%}"
        for k, v in result["probabilities"].items()
    )
    report = f"""RETINAX — SCREENING REPORT
================================

Generated: {generated}
Image: {uploaded_file.name}
Resolution: {image.width} x {image.height}px

SCREENING RESULT
----------------
Prediction: {label}
Confidence: {confidence:.2%}
Class: {class_name}
Class index: {class_index}
Mode: {"DEMO / ILLUSTRATIVE" if result["is_demo"] else "TRAINED MODEL"}

PROBABILITY DISTRIBUTION
------------------------
{probability_lines}

MODEL
-----
Architecture: EfficientNet-B0
Configured model: {MODEL_PATH}
Device: {DEVICE}

EXPLAINABILITY
--------------
Grad-CAM explanation was requested for the predicted class.

DISCLAIMER
----------
This is an educational/hackathon AI-assisted screening prototype.
It is not a medical diagnosis and must not replace evaluation by a
qualified healthcare professional.
"""

    st.download_button(
        "⬇️ Download detailed report",
        data=report,
        file_name="retinax_screening_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


with tab2:
    st.markdown("#### Why did the model predict this?")
    st.caption(
        "Grad-CAM highlights image regions that contributed more strongly to "
        "the model output. It is an explanation aid, not a clinical lesion map."
    )

    if explanation is not None:
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Original fundus image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with e2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(
                explanation,
                caption="Grad-CAM attention visualization",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.success(
            "Explanation generated successfully. Compare the original and heatmap "
            "views to understand where the model focused."
        )
    else:
        st.error("The prediction completed, but the explanation could not be generated.")
        if explanation_error:
            st.caption(f"Technical message: {explanation_error}")


with tab3:
    st.markdown("#### Inference metadata")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Architecture", "EfficientNet-B0")
    m2.metric("Device", str(DEVICE))
    m3.metric("Class index", class_index)
    m4.metric("Mode", "Demo" if result["is_demo"] else "Trained")

    st.markdown("#### Prediction payload")
    st.json(
        {
            "class": result["class"],
            "label": result["label"],
            "confidence": result["confidence"],
            "class_index": result["class_index"],
            "probabilities": result["probabilities"],
            "is_demo": result["is_demo"],
        }
    )

    st.markdown("#### Checkpoint status")
    if model_exists:
        st.success(f"Checkpoint detected at: {MODEL_PATH}")
    else:
        st.warning(
            f"No checkpoint detected at {MODEL_PATH}. "
            "The project's demo fallback is being used."
        )


# ------------------------- Disclaimer ------------------------
st.markdown(
    """
    <div class="disclaimer">
        <b>Medical safety notice:</b> RetinaX is an AI-assisted educational
        screening prototype created for a hackathon. It is not a medical
        diagnostic device. Never use this output alone to diagnose, treat,
        or rule out diabetic retinopathy. Clinical assessment by a qualified
        healthcare professional is required.
    </div>
    <div class="footer">
        RETINAX · Explainable AI for Diabetic Retinopathy Screening · Hackathon Edition
    </div>
    """,
    unsafe_allow_html=True,
)
