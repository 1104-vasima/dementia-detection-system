"""
Image preprocessing utilities for MRI brain scans
"""
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess MRI image for model inference
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Expand dimensions for model input: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_image_for_display(image, target_size=(224, 224)):
    """
    Preprocess image for display purposes (without normalization)
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        Resized PIL Image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image.resize(target_size, Image.Resampling.LANCZOS)
