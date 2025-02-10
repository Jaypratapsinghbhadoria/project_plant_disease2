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
    return tf.keras.models.load_model('plant_disease_model.h5')

model = load_model()

# Class names for predictions
CLASS_NAMES = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 
               'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus', 
               'Yellow Leaf Curl Virus']

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

            # Ensure the predictions match the number of classes
            if predictions.shape[1] != len(CLASS_NAMES):
                st.error("Model output does not match the number of class names.")
                return

            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0])) * 100

            # Display results
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()
