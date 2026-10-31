
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Facial Emotion Detector",
    page_icon="😊",
    layout="centered"
)

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

st.title("Facial Emotion Detector")
st.write("Upload a facial image and let the AI predict the expression.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Emotion"):
        with st.spinner("Analyzing facial expression..."):
            image = image.convert("L")
            image = image.resize((48, 48))

            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=(0, -1))

            predictions = model.predict(image_array, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))

            emotion = EMOTIONS[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

        st.success(f"Predicted Emotion: {emotion}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        for i, emotion_name in enumerate(EMOTIONS):
            st.write(f"{emotion_name}: {predictions[0][i] * 100:.2f}%")
            st.progress(float(predictions[0][i]))
