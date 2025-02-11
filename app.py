import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_prediction_model.h5')

model = load_model()

# Class names for predictions
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Mapping from 38 classes to 3 classes
CLASS_MAPPING = {
    0: "Early Blight", 1: "Late Blight", 2: "Healthy", 3: "Early Blight", 4: "Late Blight", 5: "Healthy",
    6: "Early Blight", 7: "Late Blight", 8: "Healthy", 9: "Early Blight", 10: "Late Blight", 11: "Healthy",
    12: "Early Blight", 13: "Late Blight", 14: "Healthy", 15: "Early Blight", 16: "Late Blight", 17: "Healthy",
    18: "Early Blight", 19: "Late Blight", 20: "Healthy", 21: "Early Blight", 22: "Late Blight", 23: "Healthy",
    24: "Early Blight", 25: "Late Blight", 26: "Healthy", 27: "Early Blight", 28: "Late Blight", 29: "Healthy",
    30: "Early Blight", 31: "Late Blight", 32: "Healthy", 33: "Early Blight", 34: "Late Blight", 35: "Healthy",
    36: "Early Blight", 37: "Late Blight"
}

def map_prediction_to_class(prediction):
    return CLASS_MAPPING[np.argmax(prediction)]

def verify_model_output_shape(model, class_names):
    # Get the model's output shape
    output_shape = model.output_shape
    if len(output_shape) > 1 and output_shape[1] != 38:
        st.error(f"Model output shape {output_shape[1]} does not match the expected 38 classes.")
        return False
    return True

# Verify the model's output shape
if not verify_model_output_shape(model, CLASS_NAMES):
    st.stop()

def preprocess_image(img):
    # Resize the image to the expected input shape of the model
    img = img.resize((256, 256))  # Adjust the size to match the model's expected input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant leaf to detect diseases")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Make prediction
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            
            # Debugging information
            st.write(f"Predictions shape: {predictions.shape}")
            st.write(f"Predictions: {predictions}")

            # Map the prediction to the desired class
            predicted_class = map_prediction_to_class(predictions[0])
            confidence = float(np.max(predictions[0])) * 100

            # Display results
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%") # Display confidence as a percentage

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()
