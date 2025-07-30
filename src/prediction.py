"""
Prediction module for making inferences with trained models
Handles single predictions, batch predictions, and model serving
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
from PIL import Image
import io
import base64
from src.preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.model_name = None
        self.class_names = []
        self.feature_names = []
        self.prediction_history = []
        
    def load_model_artifacts(self, model_name: Optional[str] = None) -> bool:
        """
        Load model and associated artifacts
        
        Args:
            model_name: Specific model name to load (optional)
            
        Returns:
            Success status
        """
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Use best model if no specific model requested
                if model_name is None:
                    model_name = self.metadata.get('best_model', 'best_model')
            
            self.model_name = model_name
            
            # Load model
            model_path_h5 = os.path.join(self.model_dir, f"{model_name}.h5")
            model_path_pkl = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            if os.path.exists(model_path_h5):
                self.model = tf.keras.models.load_model(model_path_h5)
                logger.info(f"Loaded Keras model from {model_path_h5}")
            elif os.path.exists(model_path_pkl):
                self.model = joblib.load(model_path_pkl)
                logger.info(f"Loaded sklearn model from {model_path_pkl}")
            else:
                # Try loading best_model files
                best_model_h5 = os.path.join(self.model_dir, "best_model.h5")
                best_model_pkl = os.path.join(self.model_dir, "best_model.pkl")
                
                if os.path.exists(best_model_h5):
                    self.model = tf.keras.models.load_model(best_model_h5)
                    logger.info(f"Loaded best Keras model from {best_model_h5}")
                elif os.path.exists(best_model_pkl):
                    self.model = joblib.load(best_model_pkl)
                    logger.info(f"Loaded best sklearn model from {best_model_pkl}")
                else:
                    raise FileNotFoundError(f"No model found for {model_name}")
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    preprocessor_state = pickle.load(f)
                
                self.preprocessor = ImagePreprocessor()
                self.preprocessor.target_size = preprocessor_state['target_size']
                self.preprocessor.scaler = preprocessor_state['scaler']
                self.preprocessor.pca = preprocessor_state['pca']
                self.preprocessor.label_encoder = preprocessor_state['label_encoder']
                
                logger.info("Loaded preprocessor successfully")
            else:
                # Create default preprocessor
                self.preprocessor = ImagePreprocessor()
                logger.warning("No preprocessor found, using default")
            
            # Load additional artifacts
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                self.preprocessor.scaler = scaler
            
            pca_path = os.path.join(self.model_dir, 'pca.pkl')
            if os.path.exists(pca_path):
                pca = joblib.load(pca_path)
                self.preprocessor.pca = pca
            
            # Set class names and feature names
            if self.metadata:
                self.class_names = self.metadata.get('class_names', ['Class_0', 'Class_1'])
                self.feature_names = self.metadata.get('feature_names', [])
            
            logger.info("Model artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            return False
    
    def predict_single(self, features: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            features: Input features (array, image bytes, or base64 string)
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_artifacts first.")
        
        try:
            # Preprocess input
            processed_features = self._preprocess_input(features)
            
            # Make prediction
            if isinstance(self.model, tf.keras.Model):
                prediction_proba = self.model.predict(processed_features.reshape(1, -1))[0][0]
                prediction = int(prediction_proba > 0.5)
            else:
                prediction = self.model.predict(processed_features.reshape(1, -1))[0]
                prediction_proba = self.model.predict_proba(processed_features.reshape(1, -1))[0][1]
            
            # Calculate confidence
            confidence = max(prediction_proba, 1 - prediction_proba)
            
            # Get class name
            class_name = self.class_names[prediction] if prediction < len(self.class_names) else f"Class_{prediction}"
            
            # Create result
            result = {
                'prediction': int(prediction),
                'probability': float(prediction_proba),
                'confidence': float(confidence),
                'class_name': class_name,
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'all_probabilities': {
                    self.class_names[0]: float(1 - prediction_proba),
                    self.class_names[1]: float(prediction_proba)
                } if len(self.class_names) == 2 else {}
            }
            
            # Store prediction in history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, features_list: List[Union[np.ndarray, bytes, str]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples
        
        Args:
            features_list: List of input features
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, features in enumerate(features_list):
            try:
                result = self.predict_single(features)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting sample {i}: {str(e)}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _preprocess_input(self, features: Union[np.ndarray, bytes, str]) -> np.ndarray:
        """
        Preprocess input features based on type
        
        Args:
            features: Input features in various formats
            
        Returns:
            Preprocessed feature array
        """
        if isinstance(features, np.ndarray):
            # Already a feature array
            processed = features
        elif isinstance(features, bytes):
            # Image bytes
            if self.preprocessor is None:
                raise ValueError("Preprocessor not available for image processing")
            
            image = self.preprocessor.preprocess_image_from_bytes(features)
            processed = self.preprocessor.extract_features(image)
        elif isinstance(features, str):
            # Base64 encoded image
            try:
                image_bytes = base64.b64decode(features)
                image = self.preprocessor.preprocess_image_from_bytes(image_bytes)
                processed = self.preprocessor.extract_features(image)
            except Exception as e:
                raise ValueError(f"Error decoding base64 image: {str(e)}")
        else:
            raise ValueError(f"Unsupported input type: {type(features)}")
        
        # Apply scaling if available
        if self.preprocessor and hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler:
            processed = self.preprocessor.scaler.transform(processed.reshape(1, -1)).flatten()
        
        # Apply PCA if available and model requires it
        if (self.preprocessor and hasattr(self.preprocessor, 'pca') and 
            self.preprocessor.pca and self.model_name and 'svm' in self.model_name.lower()):
            processed = self.preprocessor.pca.transform(processed.reshape(1, -1)).flatten()
        
        return processed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {'error': 'No model loaded'}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'class_names': self.class_names,
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown',
            'loaded_at': datetime.now().isoformat()
        }
        
        # Add model-specific information
        if isinstance(self.model, tf.keras.Model):
            info.update({
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_params': self.model.count_params(),
                'layers': len(self.model.layers)
            })
        
        # Add metadata if available
        if self.metadata:
            info.update({
                'training_date': self.metadata.get('training_timestamp'),
                'model_version': self.metadata.get('model_version', '1.0.0'),
                'performance_metrics': self.metadata.get('model_performances', {}).get(self.model_name, {})
            })
        
        return info
    
    def get_prediction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get prediction history
        
        Args:
            limit: Maximum number of recent predictions to return
            
        Returns:
            List of prediction records
        """
        if limit is None:
            return self.prediction_history
        else:
            return self.prediction_history[-limit:]
    
    def clear_prediction_history(self) -> None:
        """Clear prediction history"""
        self.prediction_history = []
        logger.info("Prediction history cleared")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about predictions made"""
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        # Filter out error predictions
        valid_predictions = [p for p in self.prediction_history if 'error' not in p]
        
        if not valid_predictions:
            return {'total_predictions': len(self.prediction_history), 'valid_predictions': 0}
        
        predictions = [p['prediction'] for p in valid_predictions]
        confidences = [p['confidence'] for p in valid_predictions]
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'valid_predictions': len(valid_predictions),
            'error_predictions': len(self.prediction_history) - len(valid_predictions),
            'class_distribution': {
                str(class_idx): predictions.count(class_idx) 
                for class_idx in set(predictions)
            },
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_std': np.std(confidences)
        }
        
        return stats
    
    def export_predictions(self, filepath: str, format: str = 'json') -> str:
        """
        Export prediction history to file
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file
        """
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(self.prediction_history, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(self.prediction_history)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Predictions exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {str(e)}")
            raise

class BatchPredictor:
    """Handles batch predictions efficiently"""
    
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
    
    def predict_from_directory(self, directory_path: str, 
                             file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']) -> Dict[str, Any]:
        """
        Make predictions for all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            file_extensions: List of valid file extensions
            
        Returns:
            Dictionary with results and statistics
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            return {
                'total_files': 0,
                'predictions': [],
                'error': 'No valid image files found'
            }
        
        logger.info(f"Found {len(image_files)} image files for batch prediction")
        
        # Process images in batches
        results = []
        errors = []
        
        for i, image_path in enumerate(image_files):
            try:
                # Load and preprocess image
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Make prediction
                result = self.predictor.predict_single(image_bytes)
                result['file_path'] = image_path
                result['file_name'] = os.path.basename(image_path)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                error_record = {
                    'file_path': image_path,
                    'file_name': os.path.basename(image_path),
                    'error': str(e)
                }
                errors.append(error_record)
                logger.error(f"Error processing {image_path}: {str(e)}")
        
        # Calculate statistics
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            predictions = [r['prediction'] for r in valid_results]
            confidences = [r['confidence'] for r in valid_results]
            
            stats = {
                'class_distribution': {
                    str(class_idx): predictions.count(class_idx) 
                    for class_idx in set(predictions)
                },
                'average_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            }
        else:
            stats = {}
        
        return {
            'total_files': len(image_files),
            'successful_predictions': len(valid_results),
            'failed_predictions': len(errors),
            'predictions': results,
            'errors': errors,
            'statistics': stats
        }
    
    def predict_from_csv(self, csv_path: str, feature_columns: List[str]) -> pd.DataFrame:
        """
        Make predictions for data in CSV file
        
        Args:
            csv_path: Path to CSV file
            feature_columns: List of column names containing features
            
        Returns:
            DataFrame with original data and predictions
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Extract features
            if feature_columns:
                features = df[feature_columns].values
            else:
                # Use all numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                features = df[numeric_columns].values
            
            # Make predictions
            predictions = []
            for i, feature_row in enumerate(features):
                try:
                    result = self.predictor.predict_single(feature_row)
                    predictions.append(result)
                except Exception as e:
                    predictions.append({'error': str(e)})
            
            # Add predictions to DataFrame
            df['prediction'] = [p.get('prediction', None) for p in predictions]
            df['probability'] = [p.get('probability', None) for p in predictions]
            df['confidence'] = [p.get('confidence', None) for p in predictions]
            df['class_name'] = [p.get('class_name', None) for p in predictions]
            df['prediction_error'] = [p.get('error', None) for p in predictions]
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise

def create_prediction_api_response(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format prediction result for API response
    
    Args:
        prediction_result: Raw prediction result
        
    Returns:
        Formatted API response
    """
    if 'error' in prediction_result:
        return {
            'success': False,
            'error': prediction_result['error'],
            'timestamp': prediction_result.get('timestamp', datetime.now().isoformat())
        }
    
    return {
        'success': True,
        'data': {
            'prediction': prediction_result['prediction'],
            'class_name': prediction_result['class_name'],
            'confidence': round(prediction_result['confidence'], 4),
            'probability': round(prediction_result['probability'], 4),
            'all_probabilities': {
                k: round(v, 4) for k, v in prediction_result.get('all_probabilities', {}).items()
            }
        },
        'metadata': {
            'model_name': prediction_result.get('model_name'),
            'timestamp': prediction_result['timestamp']
        }
    }

if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Load model artifacts
    if predictor.load_model_artifacts():
        print("Model loaded successfully!")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test prediction with dummy data
        dummy_features = np.random.randn(50)  # Adjust size based on your model
        result = predictor.predict_single(dummy_features)
        print(f"Test prediction: {result}")
    else:
        print("Failed to load model artifacts")
