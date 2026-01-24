import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Class names for dementia classification
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

@st.cache_resource
def load_model(model_path="resnet50_dementia_model.h5"):
    """
    Load the pre-trained ResNet50 dementia classification model from .h5 file
    
    Args:
        model_path: Path to the .h5 model file
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {os.path.abspath(model_path)}")
        st.stop()

    try:
        model = keras.models.load_model(model_path, compile=False)
        st.success("✅ ResNet50 model loaded successfully")
        return model
    except Exception as e:
        st.error("❌ Failed to load model")
        st.exception(e)
        st.stop()

def predict_dementia(model, preprocessed_image):
    """
    Predict dementia class from preprocessed image
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array (shape: (1, 224, 224, 3))
        
    Returns:
        tuple: (predicted_class_name, confidence_percentage)
    """
    try:
        # Get model prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted class index
        predicted_index = np.argmax(predictions[0])
        
        # Get predicted class name
        predicted_class = CLASS_NAMES[predicted_index]
        
        # Get confidence (probability) as percentage
        confidence = predictions[0][predicted_index] * 100
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return "Unknown", 0.0

def get_prediction_probabilities(model, preprocessed_image):
    """
    Get probability distribution across all classes
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array (shape: (1, 224, 224, 3))
        
    Returns:
        dict: Dictionary mapping class names to probabilities (as percentages)
    """
    try:
        # Get model prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Convert to dictionary with class names and probabilities
        probabilities = {
            class_name: float(prob * 100) 
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
        
        return probabilities
        
    except Exception as e:
        st.error(f"❌ Failed to get probabilities: {str(e)}")
        return {class_name: 0.0 for class_name in CLASS_NAMES}
