
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="Real-Time Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_detector.keras")

model = load_model()

# FER-2013 classes in alphabetical order
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😊 Real-Time Facial Emotion Detector")
st.write(
    "Allow webcam access to detect facial expressions "
    "continuously using our trained CNN model."
)

st.info("Click START below and allow your browser to access your camera.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Convert webcam frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:

        # Extract and prepare face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        # Predict emotion
        predictions = model.predict(face, verbose=0)[0]

        emotion_index = int(np.argmax(predictions))
        emotion = EMOTIONS[emotion_index]
        confidence = float(predictions[emotion_index]) * 100

        # Draw face rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Display emotion and confidence
        label = f"{emotion}: {confidence:.1f}%"

        cv2.rectangle(
            img,
            (x, y-35),
            (x+w, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            img,
            label,
            (x+5, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start live webcam
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

st.caption(
    "Powered by TensorFlow, OpenCV and a CNN trained on FER-2013."
)
