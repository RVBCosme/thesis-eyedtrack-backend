#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Predictor module for EyeDTrack driver monitoring system.
Handles loading and using the trained model for real-time predictions.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path: str):
        """
        Initialize the model predictor with the trained model.
        
        Args:
            model_path: Path to the trained model file (best_model.h5)
        """
        # Set TensorFlow compatibility settings
        tf.keras.backend.set_floatx('float32')
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Try to use GPU if available
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                self.model = load_model(model_path, compile=False)
            logger.info("Model loaded successfully on GPU")
        else:
            self.model = load_model(model_path, compile=False)
            logger.info("Model loaded successfully on CPU")
        
        # Enable mixed precision for better performance
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled for faster inference")
        except Exception as e:
            logger.warning(f"Mixed precision not available: {e}")
        
        # Compile model with appropriate optimizer and loss
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Image preprocessing settings
        self.img_size = (224, 224)  # Standard size for MobileNetV2
        
        # Enable XLA compilation for better performance
        tf.config.optimizer.set_jit(True)
        
        # Create a prediction function for better performance
        self.predict_fn = tf.function(
            self.model,
            input_signature=[tf.TensorSpec([None, *self.img_size, 3], tf.float32)]
        )
        
        # Define behavior categories for mapping predictions
        self.behavior_categories = [
            "NO RISKY BEHAVIOR DETECTED",
            "DROWSY",
            "DISTRACTED",
            "YAWNING"
        ]
        
        # Initialize prediction history for temporal smoothing
        self.prediction_history = []
        self.history_size = 5  # Number of predictions to consider for smoothing
        
        # Confidence thresholds - Adjusted for better accuracy
        self.CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.7 for better detection
        self.SMOOTHING_THRESHOLD = 0.55  # Lowered from 0.6 for better detection
        self.YAWNING_THRESHOLD = 0.60  # Specific threshold for yawning
        
        # Initialize mouth coverage detection
        self.mouth_coverage_history = []
        self.mouth_coverage_threshold = 0.3  # Threshold for detecting covered mouth
        
        logger.info(f"Model loaded successfully from {model_path}")
    
    def detect_mouth_coverage(self, frame: np.ndarray) -> float:
        """
        Detect if mouth is covered using color analysis.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            float: Coverage confidence (0.0 to 1.0)
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate coverage ratio
            total_pixels = frame.shape[0] * frame.shape[1]
            skin_pixels = np.sum(skin_mask > 0)
            coverage_ratio = skin_pixels / total_pixels
            
            return min(1.0, max(0.0, coverage_ratio))
        except Exception as e:
            logger.error(f"Error detecting mouth coverage: {e}")
            return 0.0
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess frame for model prediction with enhanced preprocessing.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame or None if preprocessing fails
        """
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Apply CLAHE for better contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, self.img_size)
            
            # Normalize and convert to float32
            frame = frame.astype(np.float32) / 255.0
            
            # Add batch dimension
            frame = np.expand_dims(frame, axis=0)
            
            return frame
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None
    
    def apply_temporal_smoothing(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """
        Apply temporal smoothing to predictions for more stable results.
        
        Args:
            prediction: Current behavior prediction
            confidence: Current prediction confidence
            
        Returns:
            Tuple of (smoothed prediction, confidence)
        """
        self.prediction_history.append((prediction, confidence))
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Count occurrences of each prediction
        pred_counts = {}
        for pred, conf in self.prediction_history:
            if conf >= self.SMOOTHING_THRESHOLD:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        if not pred_counts:
            return prediction, confidence
        
        # Get the most frequent prediction
        most_frequent = max(pred_counts.items(), key=lambda x: x[1])
        return most_frequent[0], confidence
    
    def predict(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Make prediction on the frame with enhanced accuracy.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (predicted behavior, confidence)
        """
        try:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return None, 0.0
            
            # Get prediction using the optimized function
            prediction = self.predict_fn(processed_frame)
            pred_class = np.argmax(prediction[0])
            confidence = float(prediction[0][pred_class])
            
            # Check for mouth coverage if prediction is yawning
            if pred_class == 3:  # Yawning class
                mouth_coverage = self.detect_mouth_coverage(frame)
                self.mouth_coverage_history.append(mouth_coverage)
                if len(self.mouth_coverage_history) > self.history_size:
                    self.mouth_coverage_history.pop(0)
                
                # If mouth is covered, increase confidence for yawning
                avg_coverage = sum(self.mouth_coverage_history) / len(self.mouth_coverage_history)
                if avg_coverage > self.mouth_coverage_threshold:
                    confidence = min(1.0, confidence * 1.2)  # Boost confidence by 20%
            
            # If confidence is below threshold, consider it as NO RISKY BEHAVIOR DETECTED
            if confidence < self.CONFIDENCE_THRESHOLD:
                return "NO RISKY BEHAVIOR DETECTED", 1.0 - confidence
            
            # Map prediction to behavior category
            if 0 <= pred_class < len(self.behavior_categories):
                behavior = self.behavior_categories[pred_class]
                
                # Apply temporal smoothing
                behavior, confidence = self.apply_temporal_smoothing(behavior, confidence)
                
                return behavior, confidence
            else:
                logger.error(f"Invalid prediction class: {pred_class}")
                return "NO RISKY BEHAVIOR DETECTED", 1.0
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, 0.0 
