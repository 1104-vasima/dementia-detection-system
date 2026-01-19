"""
Grad-CAM visualization for explainable AI
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image
    
    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class
        
    Returns:
        Heatmap array
    """
    # Find the last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Convolutional layer
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        # Fallback: use the first conv layer
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        # If no conv layer found, return a dummy heatmap
        return np.zeros((224, 224))
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

def generate_gradcam(model, preprocessed_image, original_image, alpha=0.4):
    """
    Generate Grad-CAM visualization overlaid on the original image
    
    Args:
        model: Keras model
        preprocessed_image: Preprocessed image array (1, 224, 224, 3)
        original_image: Original PIL Image
        alpha: Transparency factor for overlay (0-1)
        
    Returns:
        PIL Image with Grad-CAM overlay
    """
    try:
        # Generate heatmap
        heatmap = make_gradcam_heatmap(preprocessed_image, model)
        
        # Resize heatmap to match original image size
        original_size = original_image.size
        heatmap_resized = cv2.resize(heatmap, original_size)
        
        # Convert heatmap to RGB
        heatmap_rgb = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy array
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        original_array = np.array(original_image)
        
        # Resize original to match if needed
        if original_array.shape[:2] != heatmap_colored.shape[:2]:
            original_array = cv2.resize(original_array, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        
        # Overlay heatmap on original image
        superimposed_img = heatmap_colored * alpha + original_array * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(superimposed_img)
        
        return result_image
        
    except Exception as e:
        # If Grad-CAM fails, return original image with a warning
        import streamlit as st
        st.warning(f"Grad-CAM generation failed: {str(e)}. Returning original image.")
        return original_image
