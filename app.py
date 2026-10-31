
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="Emotion.exe",
    page_icon="💿",
    layout="wide"
)

# WINDOWS XP / Y2K STYLING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

.stApp {
    background: linear-gradient(145deg, #1675c9, #75b9ed);
    font-family: Tahoma, sans-serif;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

/* Main XP window */
.xp-window {
    background: #ece9d8;
    border: 2px solid #ffffff;
    border-radius: 9px 9px 3px 3px;
    box-shadow: 5px 7px 18px #123b6570;
    padding: 4px;
    margin-bottom: 18px;
}

/* Blue title bar */
.xp-titlebar {
    background: linear-gradient(
        to bottom,
        #3b8eea 0%,
        #1266c5 45%,
        #0054b8 50%,
        #267ee5 100%
    );
    color: white;
    padding: 10px 14px;
    border-radius: 6px 6px 0 0;
    font-weight: bold;
    font-size: 17px;
    text-shadow: 1px 1px #174477;
    display: flex;
    justify-content: space-between;
}

.xp-content {
    background: #ece9d8;
    padding: 20px;
    color: #222;
}

/* Retro heading */
h1 {
    color: #073c80 !important;
    font-family: 'VT323', monospace !important;
    font-size: 48px !important;
    text-align: center;
}

h2, h3 {
    color: #164b8a !important;
}

/* XP buttons */
.stButton button {
    background: linear-gradient(#69b8ff, #1976d2) !important;
    color: white !important;
    border: 2px solid white !important;
    border-radius: 5px !important;
    font-weight: bold !important;
    box-shadow: 1px 2px 4px #777;
}

.stButton button:hover {
    background: linear-gradient(#8dccff, #2586e8) !important;
}

/* Panels */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #a4a4a4;
    padding: 15px;
    border-radius: 5px;
    box-shadow: inset 1px 1px 3px #ddd;
}

/* Info boxes */
.stAlert {
    border-radius: 4px;
}

/* Text */
p, label {
    color: #252525;
}

/* Bottom taskbar */
.xp-taskbar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 38px;
    background: linear-gradient(
        to bottom,
        #3c91ed,
        #0759b8
    );
    border-top: 2px solid #8bc4ff;
    color: white;
    display: flex;
    align-items: center;
    padding-left: 15px;
    z-index: 999;
    font-size: 13px;
    font-weight: bold;
}

.xp-start {
    background: linear-gradient(#73d66b, #168b24);
    border: 1px solid #b5f2a4;
    border-radius: 5px 12px 12px 5px;
    padding: 5px 18px;
    margin-right: 12px;
    color: white;
    font-weight: bold;
}

.xp-footer {
    text-align: center;
    color: #164b8a;
    font-size: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# LOAD CNN MODEL
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


# WINDOWS XP HEADER
st.markdown("""
<div class="xp-window">
    <div class="xp-titlebar">
        <span>💿 Emotion.exe - Facial Recognition System</span>
        <span>_　□　✕</span>
    </div>
    <div class="xp-content">
        <h1>EMOTION DETECTOR</h1>
        <p style="text-align:center;">
            Welcome to Emotion.exe
            <br>
            Artificial Intelligence Facial Expression Recognition
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# WEBCAM PROCESSING
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

        # XP green face detection box
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )

        label = f"{emotion} | {confidence:.1f}%"

        cv2.rectangle(
            img,
            (x, max(0, y-32)),
            (x+w, y),
            (0, 160, 0),
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
        img, format="bgr24"
    )


# WEBCAM WINDOW
st.markdown("""
<div class="xp-titlebar">
    📷 Live Camera Preview
    <span>● LIVE SYSTEM</span>
</div>
""", unsafe_allow_html=True)

st.info("📀 Camera ready. Press START to begin facial analysis.")

webrtc_streamer(
    key="emotion-detector",
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


# INFORMATION PANEL
st.markdown("""
<div class="xp-window">
    <div class="xp-titlebar">
        🖥️ System Information
    </div>
    <div class="xp-content">
        <b>Application:</b> Emotion.exe<br>
        <b>AI Architecture:</b> Convolutional Neural Network<br>
        <b>Dataset:</b> FER-2013<br>
        <b>Input Resolution:</b> 48 × 48 Grayscale<br>
        <b>Face Detection:</b> OpenCV Haar Cascade<br>
        <b>Processing:</b> Real-Time
    </div>
</div>
""", unsafe_allow_html=True)


# BOTTOM TASKBAR
st.markdown("""
<div class="xp-taskbar">
    <span class="xp-start">⊞ START</span>
    💿 Emotion.exe
    <span style="margin-left:auto;margin-right:20px;">
        SYSTEM ONLINE
    </span>
</div>
""", unsafe_allow_html=True)
