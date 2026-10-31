
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="Facial Emotions Detector",
    page_icon="◈",
    layout="wide"
)

# =========================
# BLACK & WHITE UI
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@500;600;700&display=swap');

.stApp {
    background: #050505;
    color: #ffffff;
    font-family: 'Rajdhani', sans-serif;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Typography */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

h1 {
    font-size: 42px !important;
    line-height: 1.2;
}

p, label {
    color: #c9c9c9 !important;
    font-size: 17px;
}

/* Main header */
.main-header {
    background: #0b0b0b;
    border: 1px solid #ffffff;
    border-left: 5px solid #ffffff;
    padding: 30px;
    margin-bottom: 25px;
}

.project-label {
    font-family: 'Space Mono', monospace;
    color: #888888;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.main-title {
    font-family: 'Space Mono', monospace;
    color: #ffffff;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -2px;
    margin-top: 18px;
    margin-bottom: 10px;
}

.subtitle {
    color: #aaaaaa;
    font-size: 16px;
    letter-spacing: 1px;
}

.owner {
    color: #ffffff;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: bold;
    margin-top: 25px;
}

/* Panels */
.panel {
    background: #0c0c0c;
    border: 1px solid #333333;
    padding: 22px;
    margin: 18px 0;
}

.panel-heading {
    color: #ffffff;
    font-family: 'Space Mono', monospace;
    font-weight: bold;
    font-size: 14px;
    letter-spacing: 2px;
    border-bottom: 1px solid #444444;
    padding-bottom: 14px;
    margin-bottom: 18px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0c0c0c;
    border: 1px solid #444444;
    padding: 17px;
    border-radius: 0;
}

[data-testid="stMetricLabel"] {
    color: #999999 !important;
    font-family: 'Space Mono', monospace;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Space Mono', monospace;
    font-weight: bold;
}

/* Buttons */
.stButton button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: bold !important;
    transition: 0.2s ease;
}

.stButton button:hover {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}

/* Webcam */
[data-testid="stWebRtcPlayer"] {
    border: 1px solid #ffffff;
    border-radius: 0;
}

/* Alerts */
.stAlert {
    background: #111111;
    border: 1px solid #666666;
    border-radius: 0;
}

/* Status */
.status {
    color: #ffffff;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
}

/* Footer */
.footer {
    border-top: 1px solid #444444;
    margin-top: 35px;
    padding: 22px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #777777;
    letter-spacing: 2px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD TRAINED MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_detector.keras")

model = load_model()

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================
# HEADER
# =========================

st.markdown("""
<div class="main-header">
    <div class="project-label">
        DEEP LEARNING // COMPUTER VISION
    </div>

    <div class="main-title">
        FACIAL EMOTIONS<br>DETECTOR
    </div>

    <div class="subtitle">
        REAL-TIME FACIAL EXPRESSION RECOGNITION SYSTEM
    </div>

    <div class="owner">
        PROJECT BY — MUHAMMAD HAROON
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# MODEL OVERVIEW
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MODEL", "CNN")

with col2:
    st.metric("DATASET", "FER-2013")

with col3:
    st.metric("CLASSES", "07")


# =========================
# REAL-TIME DETECTION
# =========================

def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

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

        predictions = model.predict(
            face,
            verbose=0
        )[0]

        index = int(np.argmax(predictions))

        emotion = EMOTIONS[index]
        confidence = float(predictions[index]) * 100

        # White detection rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (255, 255, 255),
            2
        )

        label = f"{emotion.upper()}  |  {confidence:.1f}%"

        # Black label background
        cv2.rectangle(
            img,
            (x, max(0, y-35)),
            (x+w, y),
            (0, 0, 0),
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


# =========================
# WEBCAM INTERFACE
# =========================

st.markdown("""
<div class="panel">
    <div class="panel-heading">
        01 / LIVE CAMERA FEED
    </div>

    <div class="status">
        CAMERA INPUT // REAL-TIME ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)

st.info("Select START to activate your webcam and begin emotion detection.")

webrtc_streamer(
    key="facial-emotion-detector",
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


# =========================
# TECHNICAL INFORMATION
# =========================

st.markdown("""
<div class="panel">
    <div class="panel-heading">
        02 / PROJECT SPECIFICATIONS
    </div>

    <p><b>PROJECT:</b> Facial Emotions Detector</p>
    <p><b>DEVELOPER:</b> Muhammad Haroon</p>
    <p><b>FIELD:</b> Deep Learning & Computer Vision</p>
    <p><b>MODEL:</b> Convolutional Neural Network (CNN)</p>
    <p><b>DATASET:</b> FER-2013</p>
    <p><b>FACE DETECTION:</b> OpenCV Haar Cascade</p>
    <p><b>INPUT:</b> 48 × 48 Grayscale Images</p>
    <p><b>OUTPUT:</b> 7 Facial Expression Classes</p>
</div>
""", unsafe_allow_html=True)


# =========================
# FOOTER
# =========================

st.markdown("""
<div class="footer">
    MUHAMMAD HAROON // DEEP LEARNING PROJECT
    <br><br>
    FACIAL EMOTIONS DETECTOR
    <br><br>
    TENSORFLOW · PYTHON · OPENCV
</div>
""", unsafe_allow_html=True)
