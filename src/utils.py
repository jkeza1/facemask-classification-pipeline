import json
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_image(image_path, model_path='../models/mask_detector.h5'):
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    pred = model.predict(img_array)
    classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    predicted_class = classes[np.argmax(pred)]
    
    return {
        'prediction': predicted_class,
        'confidence': float(np.max(pred)),
        'timestamp': datetime.datetime.now().isoformat()
    }

def save_metadata(history, filepath='../models/metadata.json'):
    metadata = {
        'training_date': datetime.datetime.now().isoformat(),
        'val_accuracy': max(history.history['val_accuracy']),
        'val_loss': min(history.history['val_loss']),
        'classes': ['mask_weared_incorrect', 'with_mask', 'without_mask']
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)