import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mask_detection_model.h5')
DATA_DIR = os.path.join(BASE_DIR, 'data')