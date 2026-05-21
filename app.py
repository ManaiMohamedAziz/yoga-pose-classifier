import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="🧘",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #FAFAF7; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1A1A1A;
    margin: 0;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1rem;
    color: #666;
    margin-top: 8px;
    font-weight: 300;
}

.result-box {
    background: #FFFFFF;
    border: 1px solid #E8E8E0;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-pose {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1A1A1A;
    margin: 0.5rem 0;
    text-transform: capitalize;
}

.result-conf {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
}

.conf-bar-wrap {
    background: #F0F0EA;
    border-radius: 99px;
    height: 8px;
    margin: 1rem auto;
    width: 80%;
}

.conf-bar-fill {
    height: 8px;
    border-radius: 99px;
    background: #2D6A4F;
    transition: width 0.5s ease;
}

.pose-desc {
    font-size: 0.9rem;
    color: #555;
    line-height: 1.6;
    margin-top: 1rem;
    padding: 1rem;
    background: #F7F7F2;
    border-radius: 10px;
    text-align: left;
}

.upload-hint {
    font-size: 0.85rem;
    color: #999;
    margin-top: 0.5rem;
    font-weight: 300;
}

.divider {
    border: none;
    border-top: 1px solid #E8E8E0;
    margin: 2rem 0;
}

.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #BBB;
    margin-top: 3rem;
    font-weight: 300;
}

.stButton > button {
    background: #1A1A1A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Pose descriptions ──────────────────────────────────────────────────────
POSE_INFO = {
    "downdog": {
        "emoji": "🐕",
        "full_name": "Downward Dog",
        "description": "Adho Mukha Svanasana — an inverted V-shape pose where hands and feet are on the ground. "
                       "Strengthens the arms and legs, stretches the hamstrings, calves, and spine. "
                       "One of the most recognized poses in yoga practice.",
        "color": "#2D6A4F"
    },
    "goddess": {
        "emoji": "✨",
        "full_name": "Goddess Pose",
        "description": "Utkata Konasana — a wide-legged squat with arms extended or raised. "
                       "Strengthens the thighs, glutes, and core. Opens the hips and chest. "
                       "A powerful standing pose that builds heat and strength.",
        "color": "#6B4F9E"
    }
}

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "yoga_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ── Predict function ───────────────────────────────────────────────────────
def predict(image, model):
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    classes = ["downdog", "goddess"]
    idx = np.argmax(preds)
    return classes[idx], float(preds[idx]), {classes[i]: float(preds[i]) for i in range(len(classes))}

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧘 Yoga Pose<br>Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a yoga pose photo and let the AI identify it instantly.</p>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

model = load_model()

if model is None:
    st.error("⚠️ Model file not found. Please make sure `yoga_model.keras` is in the same folder as `app.py`.")
    st.stop()

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Supported formats: JPG, JPEG, PNG</p>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Your uploaded image", use_container_width=True)

    st.markdown("")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        predict_btn = st.button("🔍 Identify Pose")

    if predict_btn:
        with st.spinner("Analyzing pose..."):
            pose, confidence, all_probs = predict(image, model)
            info = POSE_INFO[pose]

        conf_pct = round(confidence * 100, 1)
        bar_color = info["color"]

        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:3rem">{info['emoji']}</div>
            <p class="result-pose">{info['full_name']}</p>
            <p class="result-conf">Confidence: {conf_pct}%</p>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{conf_pct}%; background:{bar_color};"></div>
            </div>
            <div class="pose-desc">{info['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Prediction breakdown:**")
        for cls, prob in all_probs.items():
            label = POSE_INFO[cls]["full_name"]
            st.progress(float(prob), text=f"{label}: {round(prob*100, 1)}%")

st.markdown('<p class="footer">Yoga Pose Classifier · Built with TensorFlow & Streamlit · Deep Learning Project</p>', unsafe_allow_html=True)
