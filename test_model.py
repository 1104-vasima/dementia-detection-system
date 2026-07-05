import numpy as np
import tensorflow as tf

def test():
    print("Loading model...")
    model = tf.keras.models.load_model("c:/Users/I Raafiha/Downloads/dementia/dementia/resnet50_dementia_model.h5", compile=False)
    
    # Dummy image (grey image, normalized)
    img_array_norm = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5
    
    # Dummy image (grey image, resnet preprocessing)
    img_array_resnet = np.ones((1, 224, 224, 3), dtype=np.float32) * 127.5
    img_array_resnet = tf.keras.applications.resnet50.preprocess_input(img_array_resnet)
    
    print("Prediction with [0,1] normalization:")
    pred_1 = model.predict(img_array_norm)
    print(pred_1)
    
    print("Prediction with ResNet50 preprocess_input:")
    pred_2 = model.predict(img_array_resnet)
    print(pred_2)

if __name__ == "__main__":
    test()
