from datetime import datetime
import base64
import io

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image as PILImage

import config
from model import load_classifier, load_segmentation_model
from utils import (
    create_three_channel_image,
    get_preprocessing_steps,
    generate_gradcam,
    image_to_bytes,
    make_gradcam_overlay,
    predict_classification,
    predict_image,
    validate_image,
)

import logging

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=config.APP_TITLE.replace("🧠", "").strip(),
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero h1, .hero h3 { font-family: 'Poppins', sans-serif; }

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container { padding-top: 1.5rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem; max-width: 1250px; }
.stApp { background: radial-gradient(circle at 10% 0%, #EEF2FF 0%, #F5F7FA 35%, #F5F7FA 100%); }

/* ---------- Hero ---------- */
.hero {
  background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
  padding: 46px 44px; border-radius: 26px; color: white; margin-bottom: 28px;
  box-shadow: 0 20px 45px rgba(30,58,138,.28);
  position: relative; overflow: hidden;
}
.hero::before {
  content:""; position:absolute; top:-60px; right:-60px; width:220px; height:220px;
  background: radial-gradient(circle, rgba(255,255,255,.14), transparent 70%); border-radius:50%;
}
.hero .badge {
  display:inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  padding: 5px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600; letter-spacing:.04em;
  margin-bottom: 16px; text-transform: uppercase;
}
.hero h1 { font-size: 42px; font-weight: 800; margin: 0 0 6px 0; line-height:1.15; }
.hero h3 { font-size: 19px; font-weight: 500; margin: 0 0 14px 0; opacity: .92; }
.hero p { font-size: 15.5px; opacity: .8; max-width: 760px; line-height:1.5; margin:0; }

/* ---------- Cards ---------- */
.card {
  background: white; border-radius: 20px; padding: 22px 24px;
  box-shadow: 0 8px 24px rgba(15,23,42,.06); border: 1px solid rgba(15,23,42,.04);
  margin-bottom: 18px;
}
.section-title { font-family:'Poppins',sans-serif; font-weight:700; font-size: 22px; color:#0F172A; margin: 6px 0 14px 0; display:flex; align-items:center; gap:10px;}

/* ---------- Status pill ---------- */
.pill { display:inline-flex; align-items:center; gap:8px; padding: 8px 16px; border-radius: 999px; font-weight:600; font-size:14px; }
.pill-ok { background:#DCFCE7; color:#15803D; }
.pill-bad { background:#FEE2E2; color:#B91C1C; }
.dot { width:8px; height:8px; border-radius:50%; background:currentColor; }

/* ---------- Metric cards ---------- */
.metric-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:14px; margin: 8px 0 6px 0; }
.metric-card { background:white; border-radius:16px; padding:18px 18px; border:1px solid rgba(15,23,42,.05); box-shadow: 0 6px 18px rgba(15,23,42,.05); text-align:left; }
.metric-card .m-label { font-size:12.5px; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:.03em; margin-bottom:6px; }
.metric-card .m-value { font-size:26px; font-weight:800; color:#0F172A; font-family:'Poppins',sans-serif; }
.metric-card .m-icon { font-size:20px; margin-bottom:6px; }
.metric-tumor .m-value { color:#DC2626; }
.metric-clear .m-value { color:#16A34A; }

/* ---------- Buttons ---------- */
.stButton>button, .stDownloadButton>button {
  width:100%; height:52px; font-size:16px; font-weight:700; border-radius:14px; border:none;
  background: linear-gradient(135deg,#2563EB,#1E40AF); color:white; box-shadow: 0 8px 20px rgba(37,99,235,.28);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(37,99,235,.35); }
.stDownloadButton>button { background: linear-gradient(135deg,#0F172A,#1E293B); box-shadow:0 8px 20px rgba(15,23,42,.22); }

/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"] { border-radius:18px; background:#F8FAFC; border: 2px dashed #CBD5E1; }

/* ---------- Native bordered containers styled as cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 20px !important;
  border: 1px solid rgba(15,23,42,.06) !important;
  box-shadow: 0 8px 24px rgba(15,23,42,.06);
  background: white;
  padding: 6px 4px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #EEF1F6; }
[data-testid="stSidebar"] .sidebar-title { font-family:'Poppins',sans-serif; font-weight:700; font-size:18px; color:#0F172A; display:flex; gap:8px; align-items:center; }
[data-testid="stSidebar"] .sidebar-caption { color:#94A3B8; font-size:12.5px; margin-top:18px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background:#F1F5F9; padding:6px; border-radius:14px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:9px 16px; font-weight:600; color:#475569; }
.stTabs [aria-selected="true"] { background:white !important; color:#1E3A8A !important; box-shadow:0 4px 10px rgba(15,23,42,.08); }

/* ---------- Disclaimer ---------- */
.disclaimer { background:#FFFBEB; border:1px solid #FDE68A; border-radius:16px; padding:16px 20px; color:#92400E; font-size:14px; }

hr { border-top: 1px solid #E2E8F0; margin: 28px 0; }

/* ---------- Interactive pipeline ---------- */
.pipeline { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:14px 0 24px; }
.pipeline-step { background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; border-radius:12px; padding:10px 13px; font-size:12px; font-weight:700; }
.pipeline-arrow { color:#64748B; font-weight:800; }
.feature-card { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px; padding:16px; margin-bottom:12px; }
.feature-card h4 { margin:0 0 7px; font-family:'Poppins',sans-serif; color:#0F172A; }
.feature-card p { margin:0; color:#475569; font-size:13px; line-height:1.5; }
.arch-box { background:linear-gradient(135deg,#F8FAFC,#EFF6FF); border:1px solid #DBEAFE; border-radius:18px; padding:20px; text-align:center; margin:8px 0 18px; }
.arch-node { display:inline-block; background:white; border:1px solid #CBD5E1; border-radius:12px; padding:11px 16px; margin:4px; font-weight:700; color:#1E293B; box-shadow:0 4px 10px rgba(15,23,42,.05); }
.arch-arrow { font-size:18px; color:#2563EB; margin:4px; }
.score-bar { background:#E2E8F0; border-radius:999px; height:11px; overflow:hidden; margin:7px 0 12px; }
.score-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#60A5FA,#1D4ED8); }

.app-footer { text-align:center; color:#94A3B8; font-size:13px; padding-top: 10px; }
.app-footer b { color:#475569; }

/* ---------- Normalized image sizing ---------- */
/* Streamlit stretches st.image() to fill its column; cap it to a sane,
   consistent size everywhere instead so scans/masks/overlays/heatmaps
   read as figures, not full-bleed banners. */
[data-testid="stImage"] {
  display: flex !important;
  justify-content: center;
}
[data-testid="stImage"] img {
  width: 100% !important;
  max-width: 420px !important;
  height: auto !important;
  border-radius: 16px;
  border: 1px solid rgba(15,23,42,.08);
  box-shadow: 0 10px 24px rgba(15,23,42,.10);
  background: #0B1220;
}
[data-testid="stImage"] figcaption,
[data-testid="stImageCaption"] {
  text-align: center !important;
  color: #64748B !important;
  font-size: 12.5px !important;
}
/* Small multi-channel thumbnails (3-up rows) can afford to be a touch smaller */
[data-testid="column"] [data-testid="stImage"] img { max-width: 320px !important; }

/* ---------- Comparison frames (scan vs. result) ---------- */
.compare-label {
  text-align:center; font-weight:700; font-size:13px; color:#475569;
  text-transform:uppercase; letter-spacing:.03em; margin-bottom:2px;
}
.confidence-legend { display:flex; align-items:center; gap:10px; margin:14px 2px 2px; }
.confidence-legend span { font-size:12px; color:#64748B; font-weight:600; white-space:nowrap; }
.confidence-legend .bar {
  flex:1; height:10px; border-radius:999px;
  background: linear-gradient(90deg,#30123b,#4669e0,#28bceb,#3fe6b4,#a4fc3c,#e1dc37,#fa8925,#d23105,#7a0403);
}
</style>
""")


def metric_card(icon, label, value, variant=""):
    return f"""<div class="metric-card {variant}"><div class="m-icon">{icon}</div>
    <div class="m-label">{label}</div><div class="m-value">{value}</div></div>"""


def render_fixed_image(img, width, caption=None):
    """Render an image at a guaranteed pixel width.

    Uses a raw base64 <img> tag instead of st.image so the size is set
    directly in inline HTML and can't be shrunk by any other CSS rule.
    """
    pil_img = PILImage.fromarray(img) if isinstance(img, np.ndarray) else img
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    caption_html = (
        f'<div style="text-align:center;color:#64748B;font-size:12.5px;margin-top:8px;">{caption}</div>'
        if caption else ""
    )
    st.html(
        f'<div style="display:flex;justify-content:center;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:{width}px;max-width:100%;height:auto;border-radius:16px;'
        f'border:1px solid rgba(15,23,42,.08);box-shadow:0 10px 24px rgba(15,23,42,.10);'
        f'background:#0B1220;">'
        f'</div>{caption_html}'
    )


# -----------------------------------------------------------------------------
# Load models (kept out of the visible UI — only surfaced as a simple status)
# -----------------------------------------------------------------------------
try:
    model, device, seg_path = load_segmentation_model()
    segmentation_ready = True
    seg_error = None
except Exception as exc:
    segmentation_ready = False
    model = None
    device = config.DEVICE
    seg_path = None
    seg_error = "The segmentation model could not be loaded. Please check that the model files are installed correctly."
    logger.exception(exc)

try:
    classifier, _, cls_path = load_classifier()
except Exception as exc:
    classifier, cls_path = None, None
    logger.exception(exc)
cls_error = None

# -----------------------------------------------------------------------------
# Sidebar — plain-language status only, no file paths / versions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.html('<div class="sidebar-title">🧠 NeuroScan AI</div>')
    st.caption("Brain MRI analysis assistant")
    st.markdown("---")

    if segmentation_ready:
        st.html('<span class="pill pill-ok"><span class="dot"></span> System ready</span>')
    else:
        st.html('<span class="pill pill-bad"><span class="dot"></span> System unavailable</span>')

    st.html("<br>")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a brain MRI scan\n"
        "2. The model segments and classifies the tumor region\n"
        "3. Review the visual results and download a report"
    )

    st.markdown("---")
    st.markdown("**Model overview**")
    st.write("Tumor segmentation and tumor/no-tumor classification, with visual explanations for each prediction.")

    with st.expander("Technical details"):
        st.write(f"Processing device: {device.type.upper()}")
        st.write(f"Input resolution: {config.IMAGE_SIZE} × {config.IMAGE_SIZE}px")
        st.write("Model input: 3-channel tensor")
        st.write(f"Segmentation threshold: {config.THRESHOLD}")
        st.write("CLAHE: clipLimit=2.0, tileGridSize=(8, 8)")
        st.write("Gaussian blur: kernel=(5, 5), sigmaX=0")
        st.write("Gabor: 21×21, sigma=5.0, λ=10.0, γ=0.5")
        st.write("Gabor orientations: 0°, 45°, 90°, 135°")
        st.write("Sobel: x/y gradients, ksize=3")
        st.write("Final tensor: 1 × 3 × 128 × 128")

    st.html('<div class="sidebar-caption">NeuroScan AI · Research & educational use only</div>')

# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.html(f"""
<div class="hero">
<div class="badge">✦ AI-Assisted MRI Analysis</div>
<h1>🧠 {config.APP_TITLE.replace('🧠', '').strip()}</h1>
<h3>{config.APP_SUBTITLE}</h3>
<p>{config.APP_DESCRIPTION}</p>
</div>
""")

if not segmentation_ready:
    st.error(seg_error)
    st.stop()

# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------
st.html('<div class="section-title">📤 Upload an MRI scan</div>')
uploaded_file = st.file_uploader(
    "Drag and drop a brain MRI image, or browse your files",
    type=sorted(config.ALLOWED_EXTENSIONS),
    label_visibility="collapsed",
)
if uploaded_file is None:
    st.info("Supported formats: PNG, JPG, JPEG, TIFF and TIF.")
    st.stop()

try:
    image = validate_image(uploaded_file.read(), uploaded_file.name)
except Exception as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([1, 1.3])
with left:
    with st.container(border=True):
        st.markdown("**Uploaded scan**")
        st.image(image, use_container_width=True)
with right:
    with st.container(border=True):
        st.markdown("**Ready to analyze**")
        st.write(
            "Your scan will be preprocessed, segmented, and classified by the model. "
            "This usually takes just a few seconds."
        )
        run = st.button("🧠 Analyze MRI", type="primary")

if not run:
    st.stop()

with st.spinner("Analyzing scan — running preprocessing and model inference..."):
    results = predict_image(model, device, image)

input_tensor = results["input_tensor"]
mask = results["mask"]
overlay = results["overlay"]
probability = results["probability_map"]
stats = results["statistics"]
inference_time = results["time"]
coverage = stats["coverage_percent"]
has_tumor_by_seg = coverage > config.TUMOR_DETECTION_THRESHOLD

# Classification / Grad-CAM
classification_available = classifier is not None
if classification_available:
    predicted_class, probabilities = predict_classification(classifier, input_tensor, device)
    tumor_probability = float(probabilities[1])
    no_tumor_probability = float(probabilities[0])
else:
    predicted_class = None
    tumor_probability = None
    no_tumor_probability = None

st.html("<hr>")
st.html('<div class="section-title">📊 Prediction Results</div>')

status_variant = "metric-tumor" if has_tumor_by_seg else "metric-clear"
cards_html = '<div class="metric-grid">'
cards_html += metric_card("🩺", "Segmentation", "Tumor Detected" if has_tumor_by_seg else "Not Detected", status_variant)
cards_html += metric_card("📐", "Tumor Coverage", f"{coverage:.2f}%")
cards_html += metric_card("⚡", "Inference Time", f"{inference_time:.3f}s")
if classification_available:
    class_variant = "metric-tumor" if predicted_class else "metric-clear"
    cards_html += metric_card("🔬", "Classifier", "Tumor" if predicted_class else "No Tumor", class_variant)
cards_html += '</div>'
st.html(cards_html)

st.html("<br>")
if classification_available:
    if predicted_class:
        st.success(f"Model confidence — tumor probability: **{tumor_probability:.1%}**")
    else:
        st.info(f"Model confidence — no-tumor probability: **{no_tumor_probability:.1%}**")
else:
    st.warning("The classification model isn't available right now, so a confidence score and visual explanation can't be shown for this scan.")

# -----------------------------------------------------------------------------
# Visual outputs
# -----------------------------------------------------------------------------
st.html("<hr>")
st.html('<div class="section-title">🖼️ Visual Breakdown</div>')

tab_names = ["Overview", "Preprocessing Lab", "U-Net++", "Segmentation", "Overlay", "Probability Heatmap"]
if classification_available:
    tab_names.append("Grad-CAM")
tabs = st.tabs(tab_names)

with tabs[0]:
    st.markdown("### 🧠 MRI Analysis Overview")
    st.markdown(
        '<div class="pipeline">'
        '<span class="pipeline-step">MRI</span><span class="pipeline-arrow">→</span>'
        '<span class="pipeline-step">Preprocess</span><span class="pipeline-arrow">→</span>'
        '<span class="pipeline-step">3 Channels</span><span class="pipeline-arrow">→</span>'
        '<span class="pipeline-step">U-Net++</span><span class="pipeline-arrow">→</span>'
        '<span class="pipeline-step">Tumor Mask</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    with c2:
        st.markdown("#### What the AI is doing")
        st.markdown(
            '<div class="feature-card"><h4>1 · Prepare</h4>'
            '<p>Enhance contrast, reduce noise, normalize intensity, and extract texture and edge features.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="feature-card"><h4>2 · Segment</h4>'
            '<p>U-Net++ receives three complementary channels and predicts a pixel-level tumor probability map.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="feature-card"><h4>3 · Classify</h4>'
            '<p>The available ResNet18 classifier estimates tumor versus no-tumor probability from the model input.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("#### 🎯 Current prediction")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric("Segmentation", "Tumor Detected" if has_tumor_by_seg else "Not Detected")
    with o2:
        st.metric("Tumor Coverage", f"{coverage:.2f}%")
    with o3:
        st.metric("Inference", f"{inference_time:.3f}s")
    if classification_available:
        st.markdown("#### 🔬 Classification confidence")
        conf = tumor_probability if predicted_class else no_tumor_probability
        label = "Tumor" if predicted_class else "No Tumor"
        st.write(f"**{label}: {conf:.1%}**")
        st.markdown(
            f'<div class="score-bar"><div class="score-fill" style="width:{conf*100:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )

with tabs[1]:
    prep = get_preprocessing_steps(image)

    st.markdown("### 🔬 Complete MRI preprocessing pipeline")
    st.caption(
        "The stages below are the same operations used to construct the model input. "
        "No additional preprocessing is applied only for visualization."
    )

    st.markdown(
        "**Grayscale → CLAHE → Gaussian filtering → Normalization → "
        "Gabor + Sobel feature extraction → 3-channel representation → Resize → U-Net++**"
    )

    st.markdown("---")
    st.markdown("#### 1. Grayscale conversion")
    st.write(
        "The uploaded RGB image is converted to a single-channel grayscale MRI image "
        "before feature extraction."
    )
    gcol, _ = st.columns([1, 1])
    with gcol:
        st.image(
            prep["grayscale"],
            caption=f"Grayscale MRI — {prep['grayscale'].shape[1]} × {prep['grayscale'].shape[0]}",
            clamp=True,
            use_container_width=True,
        )

    st.markdown("#### 2. CLAHE contrast enhancement")
    st.write(
        "Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances local contrast "
        "while limiting excessive amplification of noise."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(prep["clahe"], caption="After CLAHE", clamp=True, use_container_width=True)
    with c2:
        st.info("Parameters\n\n• clipLimit = 2.0\n• tileGridSize = (8, 8)")

    st.markdown("#### 3. Gaussian noise removal")
    st.write(
        "A Gaussian blur is applied to reduce high-frequency noise before the texture and "
        "edge features are extracted."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(prep["gaussian"], caption="After Gaussian filtering", clamp=True, use_container_width=True)
    with c2:
        st.info("Parameters\n\n• kernel = (5, 5)\n• sigmaX = 0")

    st.markdown("#### 4. Min-max normalization")
    st.write(
        "Pixel intensities are scaled to the range [0, 1] so the feature channels have a "
        "consistent numerical range."
    )
    ncol, _ = st.columns([1, 1])
    with ncol:
        st.image(
            prep["normalized"],
            caption="Normalized MRI — intensity range [0, 1]",
            clamp=True,
            use_container_width=True,
        )

    st.markdown("#### 5. Gabor texture extraction")
    st.write(
        "Gabor filters extract directional texture information from the normalized MRI. "
        "Responses from four orientations are combined using the maximum response."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(prep["gabor"], caption="Gabor texture feature map", clamp=True, use_container_width=True)
    with c2:
        st.info(
            "Parameters\n\n"
            "• Kernel = 21 × 21\n"
            "• Sigma = 5.0\n"
            "• Lambda = 10.0\n"
            "• Gamma = 0.5\n"
            "• Orientations = 0°, 45°, 90°, 135°"
        )

    st.markdown("#### 6. Sobel edge extraction")
    st.write(
        "Sobel filters compute horizontal and vertical image gradients. Their magnitude "
        "provides an edge/gradient feature map."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(prep["sobel"], caption="Sobel gradient-magnitude feature map", clamp=True, use_container_width=True)
    with c2:
        st.info("Parameters\n\n• X gradient + Y gradient\n• ksize = 3\n• Magnitude normalized to [0, 1]")

    st.markdown("---")
    st.markdown("#### 7. Three-channel model representation")
    st.write(
        "The three processed representations are stacked into the exact channel order "
        "used by the segmentation model."
    )
    channels = prep["three_channel"]
    c1, c2, c3 = st.columns(3)
    c1.image(channels[0], caption="Channel 1 — Normalized MRI", clamp=True, use_container_width=True)
    c2.image(channels[1], caption="Channel 2 — Gabor texture", clamp=True, use_container_width=True)
    c3.image(channels[2], caption="Channel 3 — Sobel edges", clamp=True, use_container_width=True)

    st.markdown("#### 8. Resize to model input resolution")
    resized = prep["resized_channels"]
    c1, c2, c3 = st.columns(3)
    c1.image(resized[0], caption=f"Channel 1 — {config.IMAGE_SIZE} × {config.IMAGE_SIZE}", clamp=True, use_container_width=True)
    c2.image(resized[1], caption=f"Channel 2 — {config.IMAGE_SIZE} × {config.IMAGE_SIZE}", clamp=True, use_container_width=True)
    c3.image(resized[2], caption=f"Channel 3 — {config.IMAGE_SIZE} × {config.IMAGE_SIZE}", clamp=True, use_container_width=True)

    st.success(
        f"Final segmentation input: **1 × 3 × {config.IMAGE_SIZE} × {config.IMAGE_SIZE}** "
        "(batch × channels × height × width)"
    )

    with st.expander("📘 Why each preprocessing step is used"):
        st.markdown(
            """
            | Step | Purpose |
            |---|---|
            | Grayscale | Converts the scan to a single intensity channel for consistent processing |
            | CLAHE | Enhances local MRI contrast |
            | Gaussian filtering | Reduces high-frequency noise |
            | Normalization | Scales intensity values to [0, 1] |
            | Gabor filtering | Extracts directional texture information |
            | Sobel filtering | Extracts gradient/edge information |
            | Channel stacking | Combines intensity, texture, and edge information |
            | Resize | Converts every channel to the model's required spatial resolution |
            """
        )

    # Download the intermediate preprocessing outputs.
    prep_download = {
        "grayscale": prep["grayscale"],
        "clahe": prep["clahe"],
        "gaussian": prep["gaussian"],
        "normalized": (prep["normalized"] * 255).clip(0, 255).astype(np.uint8),
        "gabor": (prep["gabor"] * 255).clip(0, 255).astype(np.uint8),
        "sobel": (prep["sobel"] * 255).clip(0, 255).astype(np.uint8),
    }

    with st.expander("📥 Download preprocessing stages"):
        cols = st.columns(3)
        for idx, (name, arr) in enumerate(prep_download.items()):
            with cols[idx % 3]:
                st.download_button(
                    f"⬇️ {name.title()}",
                    image_to_bytes(PILImage.fromarray(arr)),
                    f"preprocessing_{name}.png",
                    "image/png",
                    use_container_width=True,
                    key=f"prep_download_{name}",
                )

with tabs[2]:
    st.markdown("### 🏗️ U-Net++ Architecture Explorer")
    st.write(
        "A simplified conceptual view of the segmentation flow used by NeuroScan AI."
    )
    st.markdown(
        '<div class="arch-box">'
        '<div class="arch-node">3 × 128 × 128 Input</div>'
        '<div class="arch-arrow">↓</div>'
        '<div class="arch-node">ResNet34 Encoder</div>'
        '<div class="arch-arrow">↓</div>'
        '<div class="arch-node">Nested U-Net++ Decoder</div>'
        '<div class="arch-arrow">↓</div>'
        '<div class="arch-node">Segmentation Head</div>'
        '<div class="arch-arrow">↓</div>'
        '<div class="arch-node">Tumor Probability Map</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            '<div class="feature-card"><h4>🔹 ResNet34 Encoder</h4>'
            '<p>Extracts progressively richer visual features from the three-channel MRI representation.</p></div>',
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            '<div class="feature-card"><h4>🔹 U-Net++ Decoder</h4>'
            '<p>Uses nested skip connections to combine encoder features at different spatial scales.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("#### Why three channels?")
    st.markdown(
        "- **Channel 1:** normalized MRI intensity\n"
        "- **Channel 2:** Gabor-derived texture information\n"
        "- **Channel 3:** Sobel gradient/edge information"
    )
    st.info(
        "This is a conceptual architecture view, not a layer-by-layer rendering of every internal tensor."
    )

with tabs[3]:
    st.markdown("### 🩻 Predicted Tumor Segmentation")
    st.caption(
        "Binary mask after thresholding, morphological cleanup, and keeping only the "
        "largest connected region."
    )
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="compare-label">Original scan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.image(image, use_container_width=True)
    with s2:
        st.markdown('<div class="compare-label">Predicted mask</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.image(mask, use_container_width=True)
    seg_pill = "pill-bad" if has_tumor_by_seg else "pill-ok"
    seg_text = "Tumor region detected" if has_tumor_by_seg else "No tumor region detected"
    st.html(
        f'<div style="text-align:center;margin-top:14px;">'
        f'<span class="pill {seg_pill}"><span class="dot"></span> {seg_text} · {coverage:.2f}% of the scan</span>'
        f'</div>'
    )

with tabs[4]:
    st.markdown("### 🎯 Tumor Overlay")
    st.caption("Predicted tumor region highlighted and outlined directly on the original scan.")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="compare-label">Original scan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.image(image, use_container_width=True)
    with s2:
        st.markdown('<div class="compare-label">Overlay</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.image(overlay, use_container_width=True)
    st.caption("🟥 Highlighted fill = predicted tumor area  ·  🟩 Outline = region boundary")

with tabs[5]:
    st.markdown("### 🌡️ Model Confidence Heatmap")
    st.caption("Per-pixel tumor probability from the segmentation model — warmer colors mean higher confidence.")
    heat = (np.clip(probability, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    s1, s2 = st.columns([1, 1.6])
    with s1:
        st.markdown('<div class="compare-label">Original scan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_fixed_image(np.array(image.convert("RGB")), width=260)
    with s2:
        st.markdown('<div class="compare-label">Confidence heatmap</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_fixed_image(heat_color, width=250)
    st.html(
        '<div class="confidence-legend">'
        '<span>Low confidence</span><div class="bar"></div><span>High confidence</span>'
        '</div>'
    )

if classification_available:
    with tabs[6]:
        cam, cam_class, cam_probability = generate_gradcam(classifier, input_tensor, device)
        mri = input_tensor[0, 0].detach().cpu().numpy()
        cam_overlay = make_gradcam_overlay(mri, cam)
        g1, g2, g3 = st.columns(3)
        g1.image(mri, caption="MRI scan", clamp=True, use_container_width=True)
        g2.image(cam, caption="ResNet18 Grad-CAM", clamp=True, use_container_width=True)
        g3.image(cam_overlay, caption="Grad-CAM overlay", use_container_width=True)
        st.caption(f"The highlighted regions show what the model focused on when predicting **{'tumor' if cam_class else 'no tumor'}** ({cam_probability:.1%} confidence).")
        st.info("Note: this explanation reflects the model's own reasoning and is not compared against a ground-truth label, since uploaded scans don't come with one.")

st.html("<hr>")
st.html('<div class="section-title">📏 Segmentation Statistics</div>')
stat_html = '<div class="metric-grid">'
stat_html += metric_card("🧩", "Tumor Pixels", f"{stats['tumor_pixels']:,}")
stat_html += metric_card("🗺️", "Total Pixels", f"{stats['total_pixels']:,}")
stat_html += metric_card("📈", "Coverage", f"{coverage:.2f}%")
stat_html += '</div>'
st.html(stat_html)

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.html("<hr>")
st.html('<div class="section-title">🧪 Research View</div>')
with st.expander("Explore the numerical model input"):
    arr = input_tensor[0].detach().cpu().numpy()
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Channel 1 min", f"{arr[0].min():.3f}")
        st.metric("Channel 1 max", f"{arr[0].max():.3f}")
    with r2:
        st.metric("Channel 2 min", f"{arr[1].min():.3f}")
        st.metric("Channel 2 max", f"{arr[1].max():.3f}")
    with r3:
        st.metric("Channel 3 min", f"{arr[2].min():.3f}")
        st.metric("Channel 3 max", f"{arr[2].max():.3f}")
    st.caption(
        "These values describe the tensor passed to the segmentation model for this scan."
    )

st.html("<hr>")
st.html('<div class="section-title">📥 Download Results</div>')
heatmap_img = PILImage.fromarray(heat_color)
mask_bytes = image_to_bytes(mask)
overlay_bytes = image_to_bytes(overlay)
heatmap_bytes = image_to_bytes(heatmap_img)
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("⬇️ Tumor Mask", mask_bytes, "tumor_mask.png", "image/png", use_container_width=True)
with d2:
    st.download_button("⬇️ Overlay Image", overlay_bytes, "tumor_overlay.png", "image/png", use_container_width=True)
with d3:
    st.download_button("⬇️ Confidence Heatmap", heatmap_bytes, "probability_heatmap.png", "image/png", use_container_width=True)

classifier_text = (
    "Not available (classifier model not installed)"
    if tumor_probability is None
    else f"Tumor probability: {tumor_probability:.4f}\nPredicted class: {'Tumor' if predicted_class else 'No Tumor'}"
)
report = f"""
===============================
      NEUROSCAN AI REPORT
===============================

Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
Filename: {uploaded_file.name}

Segmentation result: {'Tumor detected' if has_tumor_by_seg else 'No tumor detected'}
Tumor coverage: {coverage:.2f}%
Tumor pixels: {stats['tumor_pixels']}
Analysis time: {inference_time:.3f} sec

PREPROCESSING PIPELINE
----------------------
1. RGB image -> grayscale
2. CLAHE: clipLimit=2.0, tileGridSize=(8, 8)
3. Gaussian blur: kernel=(5, 5), sigmaX=0
4. Min-max normalization to [0, 1]
5. Gabor texture extraction: 21x21 kernel, sigma=5.0, lambda=10.0, gamma=0.5,
   orientations = 0°, 45°, 90°, 135°
6. Sobel edge extraction: X/Y gradients, ksize=3
7. Channel stack: [normalized MRI, Gabor texture, Sobel magnitude]
8. Resize to {config.IMAGE_SIZE}x{config.IMAGE_SIZE}
9. Final model tensor: 1 x 3 x {config.IMAGE_SIZE} x {config.IMAGE_SIZE}

POSTPROCESSING
--------------
Binary threshold: {config.THRESHOLD}
Morphological opening + closing: 3x3 kernel
Largest connected component retained

{classifier_text}

==========================================================
This report is generated by an AI system for research and
educational purposes only and must not be used as a
substitute for professional medical diagnosis.
==========================================================
"""
st.download_button("📄 Download Full Report", report, "BrainTumorReport.txt", "text/plain", use_container_width=True)

st.html("<hr>")
st.html(
    '<div class="disclaimer">⚠️ <b>Medical Disclaimer:</b> This application is intended for educational and '
    'research purposes only. AI predictions must not be considered a medical diagnosis. Always consult a '
    'qualified healthcare professional for clinical decisions.</div>'
)

with st.expander("📊 Model validation results"):
    st.markdown("""
| Task | Metric | Score |
|---|---|---:|
| Segmentation | Accuracy | 0.9971 |
| Segmentation | Precision | 0.8564 |
| Segmentation | Recall | 0.8355 |
| Segmentation | Dice | 0.8458 |
| Classification | Accuracy | 0.9211 |
| Classification | Precision | 0.8446 |
| Classification | Recall | 0.9398 |
| Classification | F1 | 0.8897 |
| Classification | ROC-AUC | 0.9792 |
""")

with st.expander("ℹ️ About NeuroScan AI"):
    st.markdown("""
NeuroScan AI analyzes brain MRI scans in a few steps:

- **Preprocessing** — the scan is enhanced and cleaned up for consistent analysis
- **Segmentation** — a model outlines the tumor region, if one is present
- **Classification** — a second model estimates the likelihood of a tumor
- **Explanation** — a visual attention map shows what the model focused on

Trained and validated on the BraTS 2020 brain MRI dataset.
""")

st.html(
    '<div class="app-footer"><b>🧠 NeuroScan AI</b> — AI-assisted brain MRI analysis, for research & education</div>'
)
