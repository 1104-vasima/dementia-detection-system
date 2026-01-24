import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam(model, preprocessed_image, class_index=None):
    """
    Generate Grad-CAM heatmap for ResNet50
    """
    # Try to find the last convolutional layer
    # Common ResNet50 layer names
    possible_layer_names = [
        "conv5_block3_out",
        "conv5_block3_3_conv",
        "conv5_block3_add",
        "res5c_branch2c"
    ]
    
    last_conv_layer = None
    for layer_name in possible_layer_names:
        try:
            layer = model.get_layer(layer_name)
            last_conv_layer = layer_name
            break
        except:
            continue
    
    # If none found, try to find the last conv layer automatically
    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break
    
    if last_conv_layer is None:
        raise ValueError("Could not find a suitable convolutional layer for Grad-CAM")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Get outputs from gradient model
        try:
            outputs = grad_model(preprocessed_image, training=False)
        except Exception as e:
            # Fallback: get outputs separately
            predictions = model(preprocessed_image, training=False)
            conv_outputs = model.get_layer(last_conv_layer)(preprocessed_image, training=False)
        else:
            # Handle both tuple and list returns
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                conv_outputs = outputs[0]
                predictions = outputs[1]
            else:
                # If single output, try to get both separately
                predictions = model(preprocessed_image, training=False)
                conv_outputs = model.get_layer(last_conv_layer)(preprocessed_image, training=False)

        # Ensure predictions is a tensor
        if not isinstance(predictions, tf.Tensor):
            predictions = tf.convert_to_tensor(predictions)
        
        # Ensure conv_outputs is a tensor
        if not isinstance(conv_outputs, tf.Tensor):
            conv_outputs = tf.convert_to_tensor(conv_outputs)
        
        # Get class index - compute outside tape if needed, or use tensor ops
        if class_index is None:
            # Get the index of the highest prediction using tensor operations
            # We'll use the tensor directly for indexing
            class_index_tensor = tf.argmax(predictions[0])
        else:
            # Convert to tensor if needed
            if isinstance(class_index, tf.Tensor):
                class_index_tensor = class_index
            else:
                class_index_tensor = tf.constant(int(class_index))
        
        # Get the loss for the predicted class using tensor operations
        # Handle different prediction shapes
        pred_shape = predictions.shape
        if len(pred_shape) == 2:
            # Shape: (batch, classes) - standard case
            # Use tf.gather to safely index
            loss = tf.gather(predictions[0], class_index_tensor)
        elif len(pred_shape) == 1:
            # Shape: (classes,) - single sample
            loss = tf.gather(predictions, class_index_tensor)
        else:
            # Fallback: flatten and take
            pred_flat = tf.reshape(predictions, [-1])
            loss = tf.gather(pred_flat, class_index_tensor)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Handle case where grads might be None
    if grads is None:
        raise ValueError("Gradients are None. Check model architecture compatibility.")

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the conv outputs for the first (and only) image in batch
    conv_outputs = conv_outputs[0]
    
    # Weight the feature maps by the pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap - ReLU to remove negative values
    heatmap = tf.maximum(heatmap, 0)
    
    # Apply smoothing to reduce noise
    # Convert to numpy for smoothing operations
    heatmap_np = heatmap.numpy()
    
    # Apply Gaussian blur to smooth the heatmap (reduces noise)
    heatmap_np = cv2.GaussianBlur(heatmap_np, (11, 11), 0)
    
    # Normalize after smoothing
    heatmap_max = np.max(heatmap_np)
    if heatmap_max > 0:
        heatmap_np = heatmap_np / heatmap_max
    else:
        heatmap_np = heatmap_np
    
    # Resize to match input image size
    heatmap_np = cv2.resize(heatmap_np, (224, 224))
    
    # Ensure values are in [0, 1] range
    heatmap_np = np.clip(heatmap_np, 0, 1)

    return heatmap_np

def overlay_gradcam(original_image, heatmap, alpha=0.5, colormap_type='jet'):
    """
    Overlay Grad-CAM heatmap on original image with improved visualization
    
    Args:
        original_image: Original image (BGR format, numpy array)
        heatmap: Grad-CAM heatmap (numpy array, shape: (224, 224))
        alpha: Transparency factor for heatmap overlay (0.0 to 1.0)
        colormap_type: Type of colormap ('jet', 'hot', 'viridis', 'plasma')
        
    Returns:
        Overlaid image (BGR format, numpy array)
    """
    # Ensure original image is the right size and format
    if original_image.shape[:2] != (224, 224):
        original_image = cv2.resize(original_image, (224, 224))
    
    # Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap)
    
    # Normalize heatmap to 0-1 range (if not already)
    if heatmap.max() > 1.0:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    
    # Apply threshold to remove very low activations (improve contrast)
    threshold = 0.3
    heatmap = np.maximum(heatmap - threshold, 0) / (1 - threshold + 1e-8)
    
    # Normalize again after thresholding
    if heatmap.max() > 0:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    
    # Convert to 0-255 range
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap based on type
    colormap_options = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA
    }
    
    colormap = colormap_options.get(colormap_type.lower(), cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert original image to float for better blending
    original_float = original_image.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    
    # Create a mask from the heatmap for better blending
    # Use the heatmap intensity to weight the overlay
    heatmap_mask = heatmap[:, :, np.newaxis]
    
    # Blend images with adaptive alpha based on heatmap intensity
    overlaid = (1 - alpha * heatmap_mask) * original_float + (alpha * heatmap_mask) * heatmap_float
    
    # Convert back to uint8
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
    
    return overlaid