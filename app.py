import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
model = load_model('modelpickle.h5')  

# Define the labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set Streamlit page configuration
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="wide")

# Basic CSS for a clean, modern design
st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            margin-top: 20px;
            font-size: 2.5rem;
            color: #333;
        }
        .upload-section {
            text-align: center;
            margin-top: 20px;
        }
        .prediction-result {
            text-align: center;
            font-size: 1.5rem;
            color: #333;
            margin-top: 10px;
        }
        .prediction {
            font-size: 1.25rem;
            font-weight: bold;
            color: #007BFF;
        }
        .image-preview {
            max-width: 70%;
            height: auto;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header">Image Classifier</div>', unsafe_allow_html=True)
st.write("Upload an image to predict its class")

# Create Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)

    # Preprocess the image
    image = image.convert("RGB")  # Ensure the image is RGB
    image = image.resize((32, 32))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Display the result in a simple layout
    st.image(image, caption="Uploaded Image", use_column_width='auto', width=600, channels="RGB")

    # Add a slight delay for suspense
    st.markdown('<div class="prediction-result">Classifying your image...</div>', unsafe_allow_html=True)
    time.sleep(1)

    # Predict the class using the trained model
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    # Display the predicted class with the prediction probability
    predicted_prob = np.max(prediction) * 100
    st.markdown(f'<div class="prediction">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Prediction Probability: {predicted_prob:.2f}%</div>', unsafe_allow_html=True)
