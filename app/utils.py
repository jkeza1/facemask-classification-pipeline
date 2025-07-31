import os
from PIL import Image
import numpy as np

def preprocess_image(file):
    """
    Load and preprocess an image file for model input.

    Args:
        file: A file-like object or file path.

    Returns:
        Numpy array: Preprocessed image array normalized between 0-1,
                     resized to (128, 128), and ready for model prediction.
    """
    img = Image.open(file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0  # normalize pixel values
    return img_array
