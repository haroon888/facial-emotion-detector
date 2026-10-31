import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="FACIAL EMOTIONS DETECTOR // M. HAROON",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# MONOCHROME CYBERPUNK HUD CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700;800&family=Orbitron:wght@500;700;900&display=swap');

:root {
    --bg-void: #030303;
    --bg-panel: #080808;
    --fg-pure: #ffffff;
    --fg-muted: #888888;
    --fg-dim: #333333;
    --border-dim: #222222;
}

/* Base App & Subtle Cyber Grid */
.stApp {
    background-color: var(--bg-void);
    background-image: 
        linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    color: var(--fg-pure);
    font-family: 'JetBrains Mono', monospace;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Top Telemetry Bar */
.hud-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--fg-dim);
    padding-bottom: 10px;
    margin-bottom: 24px;
    font-size: 11px;
    letter-spacing: 2.5px;
    color: var(--fg-muted);
    text-transform: uppercase;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background-color: var(--fg-pure);
    margin-right: 8px;
    box-shadow: 0 0 8px #ffffff;
}

/* Main Hero Banner with Chamfered Corners */
.cyber-hero {
    position: relative;
    background: var(--bg-panel);
    border: 1px solid var(--fg-dim);
    border-left: 5px solid var(--fg-pure);
    padding: 36px 40px;
    margin-bottom: 28px;
    clip-path: polygon(0 0, 100% 0, 100% calc(100% - 22px), calc(100% - 22px) 100%, 0 100%);
}

.cyber-hero::after {
    content: "SYS.ID // MH-2026-CV";
    position: absolute;
    top: 18px;
    right: 24px;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--fg-muted);
    border: 1px solid var(--border-dim);
    padding: 4px 10px;
    background: var(--bg-void);
}

.hero-tag {
    font-size: 11px;
    letter-spacing: 4px;
    color: var(--fg-muted);
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(28px, 4.2vw, 50px);
    font-weight: 900;
    letter-spacing: 2px;
    line-height: 1.08;
    color: var(--fg-pure);
    text-transform: uppercase;
    margin: 0 0 14px 0;
}

.hero-sub {
    font-size: 13px;
    letter-spacing: 2px;
    color: #aaaaaa;
    margin-bottom: 26px;
}

.creator-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    border-top: 1px dashed var(--fg-dim);
    padding-top: 20px;
}

.creator-badge {
    background: var(--fg-pure);
    color: var(--bg-void);
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 2px;
    padding: 6px 14px;
    text-transform: uppercase;
}

.creator-meta {
    border: 1px solid var(--fg-dim);
    color: var(--fg-pure);
    font-size: 11px;
    letter-spacing: 2px;
    padding: 5px 12px;
    text-transform: uppercase;
}

/* Telemetry KPI Strip */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}

.kpi-box {
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    border-top: 2px solid var(--fg-pure);
    padding: 16px 20px;
}

.kpi-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    color: var(--fg-muted);
    margin-bottom: 6px;
}

.kpi-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--fg-pure);
    letter-spacing: 1px;
}

/* Tactical Section Panels */
.cyber-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-dim);
    padding: 22px;
    margin-bottom: 20px;
    position: relative;
}

.cyber-panel::before {
    content: "";
    position: absolute;
    top: -1px;
    left: -1px;
    width: 10px;
    height: 10px;
    border-top: 2px solid var(--fg-pure);
    border-left: 2px solid var(--fg-pure);
}

.cyber-panel::after {
    content: "";
    position: absolute;
    bottom: -1px;
    right: -1px;
    width: 10px;
    height: 10px;
    border-bottom: 2px solid var(--fg-pure);
    border-right: 2px solid var(--fg-pure);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 12px;
    margin-bottom: 16px;
}

.panel-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: var(--fg-pure);
}

.panel-code {
    font-size: 10px;
    color: var(--fg-muted);
    letter-spacing: 2px;
}

/* Spec Table */
.spec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.spec-table tr {
    border-bottom: 1px solid #151515;
}

.spec-table td {
    padding: 10px 4px;
}

.spec-key {
    color: var(--fg-muted);
    letter-spacing: 1.5px;
}

.spec-val {
    color: var(--fg-pure);
    font-weight: 600;
    text-align: right;
    letter-spacing: 1px;
}

/* Emotion Class Pills */
.class-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}

.class-chip {
    border: 1px solid var(--border-dim);
    background: var(--bg-void);
    padding: 8px 12px;
    font-size: 11px;
    letter-spacing: 1.5px;
    display: flex;
    justify-content: space-between;
    color: #cccccc;
}

.class-chip span {
    color: var(--fg-muted);
}

/* Streamlit Buttons & WebRTC Styling */
.stButton button {
    background: var(--fg-pure) !important;
    color: var(--bg-void) !important;
    border: 1px solid var(--fg-pure) !important;
    border-radius: 0px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.15s ease !important;
}

.stButton button:hover {
    background: var(--bg-void) !important;
    color: var(--fg-pure) !important;
}

[data-testid="stWebRtcPlayer"] {
    border: 1px solid var(--fg-pure) !important;
    border-radius: 0 !important;
    background: #000000 !important;
}

.stAlert {
    background-color: #090909 !important;
    border: 1px solid var(--fg-dim) !important;
    border-left: 3px solid var(--fg-pure) !important;
    border-radius: 0 !important;
    color: #cccccc !important;
}

/* Footer */
.cyber-footer {
    margin-top: 45px;
    border-top: 1px solid var(--border-dim);
    padding: 24px 0;
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--fg-muted);
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL & DETECTOR
# =========================
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("emotion_detector.keras")

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

model = load_emotion_model()
face_detector = load_face_detector()

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# =========================
# CYBERPUNK HUD FRAME CALLBACK
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h_img, w_img, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Top-left tactical camera overlay
    cv2.rectangle(img, (12, 12), (265, 40), (0, 0, 0), -1)
    cv2.rectangle(img, (12, 12), (265, 40), (255, 255, 255), 1)
    cv2.putText(
        img,
        "OPTIC.STREAM // ACTIVE",
        (22, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        # Fast tensor inference for real-time video
        predictions = model(roi, training=False).numpy()[0]
        index = int(np.argmax(predictions))
        emotion = EMOTIONS[index].upper()
        confidence = float(predictions[index]) * 100

        # Subtle outer guide frame
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)

        # Cyberpunk Corner Targeting Brackets
        corner = max(14, int(min(w, h) * 0.22))
        t = 2
        white = (255, 255, 255)

        # Top-Left
        cv2.line(img, (x, y), (x + corner, y), white, t)
        cv2.line(img, (x, y), (x, y + corner), white, t)
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - corner, y), white, t)
        cv2.line(img, (x + w, y), (x + w, y + corner), white, t)
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + corner, y + h), white, t)
        cv2.line(img, (x, y + h), (x, y + h - corner), white, t)
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - corner, y + h), white, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner), white, t)

        # Top Label Banner
        label = f"{emotion} // {confidence:.1f}%"
        label_y = max(32, y - 10)
        cv2.rectangle(img, (x, label_y - 24), (x + w, label_y + 4), (0, 0, 0), -1)
        cv2.rectangle(img, (x, label_y - 24), (x + w, label_y + 4), white, 1)

        cv2.putText(
            img,
            label,
            (x + 6, label_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            white,
            1,
            cv2.LINE_AA
        )

        # Bottom Monochrome Confidence Progress Bar
        bar_y = min(h_img - 15, y + h + 10)
        cv2.rectangle(img, (x, bar_y), (x + w, bar_y + 6), (40, 40, 40), -1)
        fill_w = int(w * (confidence / 100.0))
        cv2.rectangle(img, (x, bar_y), (x + fill_w, bar_y + 6), white, -1)

        # Target Coordinates Readout
        coord_text = f"LOC [{x},{y}]"
        cv2.putText(
            img,
            coord_text,
            (x, min(h_img - 2, bar_y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# TOP TELEMETRY BAR & HERO
# =========================
st.markdown("""
<div class="hud-topbar">
    <div><span class="status-dot"></span>NEURAL.VISION.CORE // ONLINE</div>
    <div>MODE // MONOCHROME TACTICAL HUD</div>
    <div>LATENCY // REAL-TIME WEBRTC</div>
</div>

<div class="cyber-hero">
    <div class="hero-tag">// DEEP LEARNING &amp; COMPUTER VISION SYSTEM</div>
    <div class="hero-title">FACIAL EMOTIONS DETECTOR</div>
    <div class="hero-sub">REAL-TIME NEURAL FACIAL EXPRESSION RECOGNITION &amp; TELEMETRY</div>
    <div class="creator-bar">
        <div class="creator-badge">PROJECT CREATOR // MUHAMMAD HAROON</div>
        <div class="creator-meta">ARCH // CNN-FER2013</div>
        <div class="creator-meta">INPUT // 48x48 GRAYSCALE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# SYSTEM METRICS STRIP
# =========================
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-box">
        <div class="kpi-label">01 // ARCHITECTURE</div>
        <div class="kpi-val">DEEP CNN</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">02 // TRAINING CORPUS</div>
        <div class="kpi-val">FER-2013</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">03 // TARGET CLASSES</div>
        <div class="kpi-val">07 STATES</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">04 // DETECTOR ENGINE</div>
        <div class="kpi-val">HAAR CASCADE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# MAIN WORKSPACE
# =========================
col_stream, col_specs = st.columns([1.65, 1], gap="large")

with col_stream:
    st.markdown("""
    <div class="cyber-panel">
        <div class="panel-header">
            <div class="panel-title">01 // OPTICAL SENSOR FEED</div>
            <div class="panel-code">STREAM.WEBRTC // CH-01</div>
        </div>
        <div style="font-size:12px; color:#999999; letter-spacing:1px;">
            INITIALIZE WEBCAM BELOW TO ENGAGE REAL-TIME FACE LOCK &amp; EMOTION CLASSIFICATION.
        </div>
    </div>
    """, unsafe_allow_html=True)

    webrtc_streamer(
        key="facial-emotion-detector",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_specs:
    st.markdown("""
    <div class="cyber-panel">
        <div class="panel-header">
            <div class="panel-title">02 // SYSTEM SPECIFICATIONS</div>
            <div class="panel-code">MANIFEST // V1.0</div>
        </div>
        <table class="spec-table">
            <tr>
                <td class="spec-key">PROJECT TITLE</td>
                <td class="spec-val">Facial Emotions Detector</td>
            </tr>
            <tr>
                <td class="spec-key">CREATOR</td>
                <td class="spec-val">Muhammad Haroon</td>
            </tr>
            <tr>
                <td class="spec-key">DOMAIN</td>
                <td class="spec-val">Computer Vision / DL</td>
            </tr>
            <tr>
                <td class="spec-key">MODEL FORMAT</td>
                <td class="spec-val">emotion_detector.keras</td>
            </tr>
            <tr>
                <td class="spec-key">TENSOR SHAPE</td>
                <td class="spec-val">(1, 48, 48, 1)</td>
            </tr>
            <tr>
                <td class="spec-key">NORMALIZATION</td>
                <td class="spec-val">Float32 [0.0 – 1.0]</td>
            </tr>
            <tr>
                <td class="spec-key">STACK</td>
                <td class="spec-val">TF · OpenCV · Streamlit</td>
            </tr>
        </table>
    </div>

    <div class="cyber-panel">
        <div class="panel-header">
            <div class="panel-title">03 // CLASSIFICATION MATRIX</div>
            <div class="panel-code">07 OUTPUT NODES</div>
        </div>
        <div class="class-grid">
            <div class="class-chip"><span>[00]</span> ANGRY</div>
            <div class="class-chip"><span>[01]</span> DISGUST</div>
            <div class="class-chip"><span>[02]</span> FEAR</div>
            <div class="class-chip"><span>[03]</span> HAPPY</div>
            <div class="class-chip"><span>[04]</span> NEUTRAL</div>
            <div class="class-chip"><span>[05]</span> SAD</div>
            <div class="class-chip"><span>[06]</span> SURPRISE</div>
            <div class="class-chip"><span>[SYS]</span> READY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="cyber-footer">
    <div>FACIAL EMOTIONS DETECTOR // ENGINEERED BY MUHAMMAD HAROON</div>
    <div>TENSORFLOW · OPENCV · STREAMLIT WEBRTC</div>
</div>
""", unsafe_allow_html=True)
