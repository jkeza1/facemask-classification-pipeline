"""
Data preprocessing module for the ML pipeline
Handles image preprocessing, feature extraction, and data transformation
"""

import numpy as np
import cv2
import os
import pickle
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional, Union
import tensorflow as tf
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing and feature extraction"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoder = LabelEncoder()
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def preprocess_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image from bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Preprocessed image array
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Resize image
            image_array = cv2.resize(image_array, self.target_size)
            
            # Normalize pixel values
            image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image from bytes: {str(e)}")
            raise
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Feature vector
        """
        try:
            # Flatten image for traditional ML models
            features = image.flatten()
            
            # Add some statistical features
            mean_rgb = np.mean(image, axis=(0, 1))
            std_rgb = np.std(image, axis=(0, 1))
            
            # Combine features
            combined_features = np.concatenate([features, mean_rgb, std_rgb])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def process_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Process a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (feature_matrix, successful_paths)
        """
        features_list = []
        successful_paths = []
        
        for path in image_paths:
            try:
                image = self.preprocess_image(path)
                features = self.extract_features(image)
                features_list.append(features)
                successful_paths.append(path)
            except Exception as e:
                logger.warning(f"Skipping {path}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No images could be processed successfully")
        
        return np.array(features_list), successful_paths
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit the scaler on training data"""
        self.scaler.fit(X)
        logger.info("Scaler fitted successfully")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        return self.scaler.transform(X)
    
    def fit_pca(self, X: np.ndarray, n_components: Union[int, float] = 0.95) -> None:
        """
        Fit PCA on training data
        
        Args:
            X: Training features
            n_components: Number of components or variance to retain
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        logger.info(f"PCA fitted with {self.pca.n_components_} components")
    
    def apply_pca(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA transformation"""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.transform(X)
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers"""
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode integer labels to strings"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the preprocessor state"""
        preprocessor_state = {
            'target_size': self.target_size,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoder': self.label_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load the preprocessor state"""
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.target_size = preprocessor_state['target_size']
        self.scaler = preprocessor_state['scaler']
        self.pca = preprocessor_state['pca']
        self.label_encoder = preprocessor_state['label_encoder']
        
        logger.info(f"Preprocessor loaded from {filepath}")

class DataAugmentation:
    """Handles data augmentation for training"""
    
    def __init__(self):
        self.augmentation_pipeline = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single image"""
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image_tensor)
        
        # Convert back to numpy
        return augmented.numpy()[0]
    
    def augment_batch(self, images: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Augment a batch of images
        
        Args:
            images: Batch of images
            factor: Augmentation factor (how many augmented versions per image)
            
        Returns:
            Augmented image batch
        """
        augmented_images = []
        
        for image in images:
            # Add original image
            augmented_images.append(image)
            
            # Add augmented versions
            for _ in range(factor - 1):
                aug_image = self.augment_image(image)
                augmented_images.append(aug_image)
        
        return np.array(augmented_images)

def create_data_splits(X: np.ndarray, y: np.ndarray, 
                      train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, 
                      test_ratio: float = 0.15,
                      random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Create train/validation/test splits
    
    Args:
        X: Features
        y: Labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data splits created:")
    logger.info(f"  Training: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_dataset_from_directory(data_dir: str, 
                               preprocessor: ImagePreprocessor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset from directory structure
    
    Expected structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
            image4.jpg
    
    Args:
        data_dir: Root directory containing class subdirectories
        preprocessor: ImagePreprocessor instance
        
    Returns:
        Tuple of (features, labels, class_names)
    """
    features_list = []
    labels_list = []
    class_names = []
    
    # Get class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    class_names = sorted(class_dirs)
    
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        logger.info(f"Processing {len(image_files)} images for class '{class_name}'")
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            
            try:
                image = preprocessor.preprocess_image(image_path)
                features = preprocessor.extract_features(image)
                
                features_list.append(features)
                labels_list.append(class_idx)
                
            except Exception as e:
                logger.warning(f"Skipping {image_path}: {str(e)}")
                continue
    
    if not features_list:
        raise ValueError("No images could be processed successfully")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
    
    return X, y, class_names

if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Create some dummy data for testing
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = preprocessor.extract_features(dummy_image.astype(np.float32) / 255.0)
    
    print(f"Extracted features shape: {features.shape}")
    print("Preprocessing module loaded successfully!")
