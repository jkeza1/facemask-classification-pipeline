import os

# Get the project root folder (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to the model in the root models folder
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_detection_model.h5')

# Example data directory (adjust if needed)
DATA_DIR = os.path.join(BASE_DIR, 'data')
