<<<<<<< HEAD
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
=======
"""
Model loading and prediction utilities
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Class labels for dementia classification
CLASS_LABELS = {
    0: "NonDemented",
    1: "VeryMildDemented",
    2: "MildDemented",
    3: "ModerateDemented"
}

@st.cache_resource
def load_model(model_path="model.h5"):
    """
    Load the pretrained dementia classification model
    
    Args:
        model_path: Path to the model.h5 file
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.info("Please ensure model.h5 is in the root directory.")
        # Return a dummy model for development/testing
        st.warning("Using dummy model for demonstration. Please add model.h5 for actual predictions.")
        return create_dummy_model()
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Using dummy model for demonstration.")
        return create_dummy_model()

def create_dummy_model():
    """
    Create a dummy model for demonstration when model.h5 is not available
    This allows the app to run and demonstrate functionality
    """
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(4, activation='softmax')
    ])
    
    # Compile with dummy weights
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def predict_dementia(model, preprocessed_image):
    """
    Predict dementia class from preprocessed image
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array (1, 224, 224, 3)
        
    Returns:
        tuple: (predicted_class_name, confidence_percentage)
    """
    try:
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get confidence score
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Map to class name
        predicted_class = CLASS_LABELS.get(predicted_class_idx, "Unknown")
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # Return dummy prediction for demonstration
        return "NonDemented", 75.0

def get_prediction_probabilities(model, preprocessed_image):
    """
    Get probability distribution over all classes
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array
        
    Returns:
        Dictionary mapping class names to probabilities
    """
    try:
        predictions = model.predict(preprocessed_image, verbose=0)
        probabilities = {}
        for idx, class_name in CLASS_LABELS.items():
            probabilities[class_name] = predictions[0][idx] * 100
        return probabilities
    except Exception as e:
        st.error(f"Error getting probabilities: {str(e)}")
        return {name: 25.0 for name in CLASS_LABELS.values()}
>>>>>>> 29fa703cec59b2eec93aedccc891802ebc584cbc
