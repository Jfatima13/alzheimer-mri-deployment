import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Alzheimer MRI Classification",
    layout="centered"
)

st.title("🧠 Alzheimer MRI Classification System")
st.write("Upload a brain MRI image to predict Alzheimer disease stage.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzheimer_model.h5")

model = load_model()

# Class labels
class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# Accuracy info (hardcoded from your results)
st.markdown("### 📊 Model Performance")
st.write("**Validation Accuracy:** ~92%")
st.write("**Balanced Accuracy:** ~25% (class-imbalanced dataset)")

# Image upload
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.markdown("### 🧪 Prediction Result")
    st.success(
        f"**Prediction:** {class_names[predicted_class]}\n\n"
        f"**Confidence:** {confidence:.2f}%"
    )

    # Show probabilities
    st.markdown("### 🔍 Prediction Probabilities")
    for label, prob in zip(class_names, predictions[0]):
        st.write(f"{label}: {prob*100:.2f}%")
