from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from app.config import MODEL_PATH

def load_model_from_file():
    return load_model(MODEL_PATH)

def predict_image(model, img_path):
    class_names = ['Incorrect', 'Mask', 'No Mask']  
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    # Optional: print the raw prediction for debugging
    print("Prediction probabilities:", prediction[0])

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    return predicted_class

def retrain_model(new_data_dir):
    # Your retraining logic here
    pass
