import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# Title
st.title("🔢 Handwritten Digit Classification")

st.write(
    "Upload an image of a handwritten digit and the CNN model "
    "will predict the digit."
)


# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.keras")


model = load_model()


# Upload image
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    predictions = model.predict(image_array)

    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # Display result
    st.success(f"Predicted Digit: {predicted_digit}")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


# Model information
st.divider()

st.subheader("About the Model")

st.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "trained on the MNIST handwritten digit dataset."
)

st.write("Test Accuracy: **98.66%**")
