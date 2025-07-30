"""
Model creation, training, and management module
Supports multiple model types and provides unified interface
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Unified model training and management class"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_performances = {}
        self.best_model_name = None
        self.training_history = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest model with default or custom parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        return RandomForestClassifier(**default_params)
    
    def create_svm(self, **kwargs) -> SVC:
        """Create SVM model with default or custom parameters"""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        return SVC(**default_params)
    
    def create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Create Logistic Regression model with default or custom parameters"""
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        return LogisticRegression(**default_params)
    
    def create_neural_network(self, input_shape: int, num_classes: int = 2, **kwargs) -> tf.keras.Model:
        """Create Neural Network model with default or custom architecture"""
        
        # Default architecture parameters
        default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid' if num_classes == 2 else 'softmax'
        }
        default_params.update(kwargs)
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            default_params['hidden_layers'][0], 
            activation=default_params['activation'],
            input_shape=(input_shape,)
        ))
        model.add(layers.Dropout(default_params['dropout_rate']))
        
        # Hidden layers
        for units in default_params['hidden_layers'][1:]:
            model.add(layers.Dense(units, activation=default_params['activation']))
            model.add(layers.Dropout(default_params['dropout_rate']))
        
        # Output layer
        output_units = 1 if num_classes == 2 else num_classes
        model.add(layers.Dense(output_units, activation=default_params['output_activation']))
        
        # Compile model
        loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   **training_kwargs) -> Dict[str, Any]:
        """
        Train a model and store it
        
        Args:
            model_name: Name identifier for the model
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for neural networks)
            y_val: Validation labels (for neural networks)
            **training_kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_name}...")
        start_time = datetime.now()
        
        try:
            if isinstance(model, tf.keras.Model):
                # Neural network training
                training_params = {
                    'epochs': 100,
                    'batch_size': 32,
                    'verbose': 1,
                    'validation_data': (X_val, y_val) if X_val is not None else None
                }
                training_params.update(training_kwargs)
                
                # Add callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                if 'callbacks' in training_params:
                    callbacks_list.extend(training_params['callbacks'])
                training_params['callbacks'] = callbacks_list
                
                # Train model
                history = model.fit(X_train, y_train, **training_params)
                self.training_history[model_name] = history.history
                
            else:
                # Scikit-learn model training
                model.fit(X_train, y_train)
            
            # Store model
            self.models[model_name] = model
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'training_completed': True
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'training_time': 0,
                'training_completed': False,
                'error': str(e)
            }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.models[model_name]
        
        try:
            # Make predictions
            if isinstance(model, tf.keras.Model):
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Store performance
            self.model_performances[model_name] = metrics
            
            logger.info(f"{model_name} evaluation completed:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {}
    
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                            models_to_train: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Train multiple models for comparison
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of training results
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'svm', 'logistic_regression', 'neural_network']
        
        results = {}
        
        for model_name in models_to_train:
            try:
                if model_name == 'random_forest':
                    model = self.create_random_forest()
                elif model_name == 'svm':
                    model = self.create_svm()
                elif model_name == 'logistic_regression':
                    model = self.create_logistic_regression()
                elif model_name == 'neural_network':
                    model = self.create_neural_network(X_train.shape[1])
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Train model
                training_result = self.train_model(
                    model_name, model, X_train, y_train, X_val, y_val
                )
                results[model_name] = training_result
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e), 'training_completed': False}
        
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with model comparison
        """
        results = []
        
        for model_name in self.models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            if metrics:
                metrics['model_name'] = model_name
                results.append(metrics)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.set_index('model_name')
        
        # Find best model based on F1 score
        if 'f1_score' in df.columns:
            self.best_model_name = df['f1_score'].idxmax()
            logger.info(f"Best model: {self.best_model_name} (F1: {df.loc[self.best_model_name, 'f1_score']:.4f})")
        
        return df
    
    def hyperparameter_tuning(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                            param_grid: Dict, cv: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'logistic_regression')
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and score
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        # Create base model
        if model_type == 'random_forest':
            base_model = self.create_random_forest()
        elif model_type == 'svm':
            base_model = self.create_svm()
        elif model_type == 'logistic_regression':
            base_model = self.create_logistic_regression()
        else:
            raise ValueError(f"Hyperparameter tuning not supported for {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        best_model_name = f"{model_type}_tuned"
        self.models[best_model_name] = grid_search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed for {model_type}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'model_name': best_model_name
        }
    
    def save_model(self, model_name: str, filepath: Optional[str] = None) -> str:
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            filepath: Custom filepath (optional)
            
        Returns:
            Path where model was saved
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if filepath is None:
            if isinstance(model, tf.keras.Model):
                filepath = os.path.join(self.model_dir, f"{model_name}.h5")
            else:
                filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        try:
            if isinstance(model, tf.keras.Model):
                model.save(filepath)
            else:
                joblib.dump(model, filepath)
            
            logger.info(f"Model {model_name} saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a saved model
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
        """
        try:
            if filepath.endswith('.h5'):
                model = tf.keras.models.load_model(filepath)
            else:
                model = joblib.load(filepath)
            
            self.models[model_name] = model
            logger.info(f"Model loaded as {model_name} from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise
    
    def save_best_model(self) -> str:
        """Save the best performing model"""
        if self.best_model_name is None:
            raise ValueError("No best model identified. Run evaluate_all_models first.")
        
        return self.save_model(self.best_model_name)
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models"""
        summary = {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'best_model': self.best_model_name,
            'performances': self.model_performances,
            'training_history': self.training_history
        }
        
        return summary
    
    def save_training_metadata(self, additional_info: Optional[Dict] = None) -> str:
        """
        Save training metadata and model information
        
        Args:
            additional_info: Additional information to include
            
        Returns:
            Path to saved metadata file
        """
        metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'model_performances': self.model_performances,
            'training_history': {k: {key: [float(val) for val in values] if isinstance(values, list) else values 
                               for key, values in v.items()} for k, v in self.training_history.items()},
            'model_summary': self.get_model_summary()
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        filepath = os.path.join(self.model_dir, 'training_metadata.json')
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {filepath}")
        return filepath

class ModelRetrainer:
    """Handles model retraining with new data"""
    
    def __init__(self, model_trainer: ModelTrainer):
        self.model_trainer = model_trainer
        self.retraining_history = []
    
    def should_retrain(self, new_data_size: int, performance_threshold: float = 0.05,
                      data_threshold: int = 100) -> bool:
        """
        Determine if model should be retrained based on criteria
        
        Args:
            new_data_size: Size of new data available
            performance_threshold: Performance degradation threshold
            data_threshold: Minimum new data size to trigger retraining
            
        Returns:
            Boolean indicating if retraining is recommended
        """
        reasons = []
        
        # Check data threshold
        if new_data_size >= data_threshold:
            reasons.append(f"New data size ({new_data_size}) exceeds threshold ({data_threshold})")
        
        # Check performance degradation (would need monitoring data)
        # This is a placeholder for actual performance monitoring
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            logger.info(f"Retraining recommended. Reasons: {'; '.join(reasons)}")
        
        return should_retrain
    
    def retrain_model(self, model_name: str, X_new: np.ndarray, y_new: np.ndarray,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                     incremental: bool = False) -> Dict:
        """
        Retrain a model with new data
        
        Args:
            model_name: Name of model to retrain
            X_new: New training features
            y_new: New training labels
            X_val: Validation features
            y_val: Validation labels
            incremental: Whether to use incremental learning (if supported)
            
        Returns:
            Retraining results
        """
        if model_name not in self.model_trainer.models:
            raise ValueError(f"Model {model_name} not found")
        
        logger.info(f"Starting retraining for {model_name}")
        start_time = datetime.now()
        
        try:
            model = self.model_trainer.models[model_name]
            
            if isinstance(model, tf.keras.Model):
                # For neural networks, we can continue training
                if incremental:
                    # Continue training with lower learning rate
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                
                # Train with new data
                history = model.fit(
                    X_new, y_new,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=50,
                    batch_size=32,
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                    ]
                )
                
                # Update training history
                if model_name in self.model_trainer.training_history:
                    # Append new history
                    for key, values in history.history.items():
                        self.model_trainer.training_history[model_name][key].extend(values)
                else:
                    self.model_trainer.training_history[model_name] = history.history
            
            else:
                # For sklearn models, retrain from scratch with new data
                model.fit(X_new, y_new)
            
            # Record retraining event
            retraining_time = (datetime.now() - start_time).total_seconds()
            retraining_record = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'new_data_size': len(X_new),
                'retraining_time': retraining_time,
                'incremental': incremental
            }
            
            self.retraining_history.append(retraining_record)
            
            logger.info(f"Retraining completed for {model_name} in {retraining_time:.2f} seconds")
            
            return {
                'success': True,
                'retraining_time': retraining_time,
                'new_data_size': len(X_new),
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def get_retraining_history(self) -> List[Dict]:
        """Get history of all retraining events"""
        return self.retraining_history

def create_ensemble_model(models: Dict[str, Any], weights: Optional[List[float]] = None) -> 'EnsembleModel':
    """
    Create an ensemble model from multiple trained models
    
    Args:
        models: Dictionary of trained models
        weights: Optional weights for each model
        
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(models, weights)

class EnsembleModel:
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[List[float]] = None):
        self.models = models
        self.model_names = list(models.keys())
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for model_name, model in self.models.items():
            if isinstance(model, tf.keras.Model):
                pred = model.predict(X).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Convert to binary predictions
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        predictions = []
        
        for model_name, model in self.models.items():
            if isinstance(model, tf.keras.Model):
                pred = model.predict(X).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Return as probability matrix
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

if __name__ == "__main__":
    # Example usage
    print("Model training module loaded successfully!")
    
    # Create dummy data for testing
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.randn(200, 50)
    y_test = np.random.randint(0, 2, 200)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train multiple models
    training_results = trainer.train_multiple_models(X_train, y_train)
    print(f"Training results: {training_results}")
    
    # Evaluate models
    evaluation_df = trainer.evaluate_all_models(X_test, y_test)
    print(f"Model evaluation:\n{evaluation_df}")
    
    # Save best model
    if trainer.best_model_name:
        trainer.save_best_model()
        print(f"Best model ({trainer.best_model_name}) saved successfully!")
