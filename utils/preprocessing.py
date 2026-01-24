import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess image exactly like ResNet50 training
    """
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    return img_array
