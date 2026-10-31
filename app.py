
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av
import time

from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="NEURO//SCAN",
    page_icon="🔮",
    layout="wide"
)

# ==============================
# CYBERPUNK UI
# ==============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');

.stApp {
    background: #080810;
    color: #e9e9ff;
    font-family: 'Rajdhani', sans-serif;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 2px;
}

h1 {
    color: #ff2bd6 !important;
    text-shadow: 0 0 12px #ff2bd688;
    font-size: 42px !important;
}

h2, h3 {
    color: #00f5ff !important;
}

p, label {
    color: #c4c4dd !important;
    font-size: 17px;
}

/* Main header */
.cyber-header {
    background: linear-gradient(120deg, #121225, #1b102b);
    border: 1px solid #ff2bd6;
    border-left: 5px solid #ff2bd6;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 0 20px #ff2bd622;
}

.cyber-subtitle {
    color: #00f5ff;
    letter-spacing: 4px;
    font-size: 13px;
}

/* Panels */
.cyber-panel {
    background: #11111e;
    border: 1px solid #292944;
    border-top: 2px solid #00f5ff;
    padding: 20px;
    margin: 18px 0;
    box-shadow: 0 0 15px #00f5ff12;
}

.cyber-panel-title {
    font-family: 'Orbitron', sans-serif;
    color: #00f5ff;
    font-size: 15px;
    letter-spacing: 2px;
    border-bottom: 1px solid #30304a;
    padding-bottom: 12px;
    margin-bottom: 15px;
}

/* Streamlit buttons */
.stButton button {
    background: #ff2bd6 !important;
    color: #080810 !important;
    border: 1px solid #ff2bd6 !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: bold !important;
    box-shadow: 0 0 15px #ff2bd655;
}

.stButton button:hover {
    background: #00f5ff !important;
    border-color: #00f5ff !important;
    box-shadow: 0 0 20px #00f5ff77;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #151526;
    border: 1px solid #30304a;
    padding: 16px;
    border-radius: 3px;
}

[data-testid="stMetricLabel"] {
    color: #00f5ff !important;
    font-family: 'Orbitron', sans-serif;
}

[data-testid="stMetricValue"] {
    color: #ff2bd6 !important;
}

/* Alerts */
.stAlert {
    background: #151526;
    border: 1px solid #00f5ff;
}

/* Webcam controls */
[data-testid="stWebRtcPlayer"] {
    border: 1px solid #ff2bd6;
    box-shadow: 0 0 20px #ff2bd622;
}

/* Status text */
.status {
    color: #00ff9c;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 2px;
}

/* Footer */
.cyber-footer {
    text-align: center;
    border-top: 1px solid #30304a;
    padding: 20px;
    color: #666680;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    letter-spacing: 2px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_detector.keras")

model = load_model()

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==============================
# HEADER
# ==============================

st.markdown("""
<div class="cyber-header">
    <div class="cyber-subtitle">
        NEO-TOKYO // ARTIFICIAL INTELLIGENCE DIVISION
    </div>
    <h1>NEURO//SCAN</h1>
    <p>REAL-TIME FACIAL EXPRESSION RECOGNITION SYSTEM</p>
    <div class="status">● SYSTEM INITIALIZED // CNN MODEL ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ==============================
# SYSTEM INFORMATION
# ==============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI MODEL", "CNN")

with col2:
    st.metric("DATASET", "FER-2013")

with col3:
    st.metric("INPUT", "48 × 48 PX")


# ==============================
# FACE DETECTION
# ==============================

def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        predictions = model.predict(face, verbose=0)[0]

        index = int(np.argmax(predictions))
        emotion = EMOTIONS[index]
        confidence = float(predictions[index]) * 100

        # Neon cyan face box
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (255, 245, 0),
            2
        )

        label = f"{emotion.upper()} // {confidence:.1f}%"

        cv2.rectangle(
            img,
            (x, max(0, y-35)),
            (x+w, y),
            (180, 0, 180),
            -1
        )

        cv2.putText(
            img,
            label,
            (x+5, max(20, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# ==============================
# LIVE CAMERA
# ==============================

st.markdown("""
<div class="cyber-panel">
    <div class="cyber-panel-title">
        ◈ LIVE SURVEILLANCE FEED
    </div>
    <div class="status">
        CAMERA INPUT // AWAITING CONNECTION
    </div>
</div>
""", unsafe_allow_html=True)

webrtc_streamer(
    key="neuro-scan",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# ==============================
# MODEL INFORMATION
# ==============================

st.markdown("""
<div class="cyber-panel">
    <div class="cyber-panel-title">
        ◈ NEURAL NETWORK SPECIFICATIONS
    </div>
    <p>ARCHITECTURE: CONVOLUTIONAL NEURAL NETWORK</p>
    <p>TRAINING DATA: FER-2013</p>
    <p>FACE DETECTION: OPENCV HAAR CASCADE</p>
    <p>IMAGE PROCESSING: GRAYSCALE / 48 × 48</p>
    <p>OUTPUT CLASSES: 07 FACIAL EXPRESSIONS</p>
    <div class="status">● NEURAL ENGINE READY</div>
</div>
""", unsafe_allow_html=True)


# ==============================
# FOOTER
# ==============================

st.markdown("""
<div class="cyber-footer">
    NEURO//SCAN © 2026
    <br><br>
    BUILT WITH TENSORFLOW // OPENCV // PYTHON
</div>
""", unsafe_allow_html=True)
