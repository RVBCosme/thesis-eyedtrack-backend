#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EyeDTrack: Combined Model and Face Metrics Testing Script

This script evaluates the accuracy of the combined behavior detection system,
using both the trained model and face metrics for predictions.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
import json
from datetime import datetime
from face_analysis import FaceDetector, HeadPoseEstimator
from face_analysis.ear_mar import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    get_facial_state
)
from model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define thresholds
EAR_THRESHOLD = 0.25  # For drowsiness detection
MAR_THRESHOLD = 0.55  # For yawning detection
HEAD_POSE_THRESHOLD = 25.0  # For distraction detection

def load_and_preprocess_test_data(test_dir, img_size=(224, 224), batch_size=32):
    """Load and preprocess test data"""
    test_data = []
    test_labels = []
    
    # Define behavior categories and their directories
    behavior_dirs = {
        "DISTRACTED": "dataset/test/is_distracted",
        "DROWSY": "dataset/test/is_drowsy",
        "YAWNING": "dataset/test/is_yawning"
    }
    
    # Load one image from each category
    for behavior, dir_path in behavior_dirs.items():
        if os.path.exists(dir_path):
            # Get the first image from each directory
            for img_file in os.listdir(dir_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dir_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize image to match model input size
                            img = cv2.resize(img, img_size)
                            test_data.append(img)
                            test_labels.append(behavior)
                            logger.info(f"Loaded {behavior} image: {img_path}")
                            break  # Only take one image per category
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
    
    return test_data, test_labels

def plot_confusion_matrix(cm, classes, title='Combined System Confusion Matrix'):
    """Plot confusion matrix with better visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    save_dir = 'test_results/combined'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(test_data, test_labels, predictions, model_scores, face_metrics_list, num_samples=2):
    """Plot sample predictions for each behavior category with improved visualization"""
    save_dir = 'test_results/combined'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get unique behavior categories
    behavior_categories = list(set(test_labels))
    
    for behavior in behavior_categories:
        # Find indices of current behavior
        indices = [i for i, label in enumerate(test_labels) if label == behavior]
        
        # Select random samples
        sample_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        
        for idx in sample_indices:
            img = test_data[idx]
            true_label = test_labels[idx]
            pred_label = predictions[idx]
            model_score = model_scores[idx]
            face_metrics = face_metrics_list[idx]
            
            # Create figure with larger size and better spacing
            plt.figure(figsize=(15, 10))
            
            # Plot image
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            title = f'True: {true_label}\nPredicted: {pred_label}'
            if true_label == pred_label:
                title += '\n(Correct)'
            else:
                title += '\n(Incorrect)'
            plt.title(title)
            
            # Plot model confidence scores
            plt.subplot(2, 2, 2)
            behaviors = list(model_score.keys())
            scores = list(model_score.values())
            bars = plt.bar(behaviors, scores)
            plt.title('Model Confidence Scores')
            plt.ylim(0, 1.2)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            # Plot face metrics
            plt.subplot(2, 2, 3)
            metrics = {
                'EAR': face_metrics['avg_ear'],
                'MAR': face_metrics['mar'],
                'Head Pose Yaw': face_metrics['head_pose']['yaw'] if face_metrics['head_pose'] else 0
            }
            
            bars = plt.bar(metrics.keys(), metrics.values())
            plt.title('Face Metrics')
            plt.ylim(0, 1.2)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            # Plot confidence scores
            plt.subplot(2, 2, 4)
            confidences = {
                'Model': max(model_score.values()),
                'EAR': face_metrics['ear_confidence'],
                'MAR': face_metrics['mar_confidence']
            }
            
            bars = plt.bar(confidences.keys(), confidences.values())
            plt.title('Confidence Scores')
            plt.ylim(0, 1.2)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save with specific behavior name and sample number
            save_path = os.path.join(save_dir, f'sample_prediction_{behavior}_{idx}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Log prediction details
            logger.info(f"Saved prediction for {behavior} (Sample {idx})")
            logger.info(f"True label: {true_label}")
            logger.info(f"Predicted behavior: {pred_label}")
            logger.info(f"Model scores: {model_score}")
            logger.info(f"Face metrics: {face_metrics}")

def evaluate_combined_system():
    """Evaluate the combined behavior detection system"""
    try:
        # Initialize components
        face_detector = FaceDetector()
        head_pose_estimator = HeadPoseEstimator(frame_size=(640, 480))
        model_predictor = ModelPredictor()
        
        # Load test data
        test_dir = Path('dataset/test')
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found at {test_dir}")
        
        test_data, test_labels = load_and_preprocess_test_data(test_dir)
        if not test_data:
            raise ValueError("No test data loaded")
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Process each test image
        predictions = []
        model_scores = []
        face_metrics_list = []
        
        for img in test_data:
            # Get model prediction
            model_prediction = model_predictor.predict(img)
            model_scores.append(model_prediction['scores'])
            
            # Detect face
            face_boxes = face_detector.detect_faces(img)
            
            if face_boxes:
                x, y, w, h = face_boxes[0]
                face_roi = img[y:y+h, x:x+w]
                
                # Get landmarks
                landmarks = face_detector.get_landmarks(img, face_boxes[0])
                
                if landmarks is not None:
                    # Extract eye and mouth landmarks
                    left_eye = []
                    right_eye = []
                    mouth = []
                    
                    # Left eye (points 36-41)
                    for i in range(36, 42):
                        pt = landmarks.part(i)
                        left_eye.append((pt.x, pt.y))
                    
                    # Right eye (points 42-47)
                    for i in range(42, 48):
                        pt = landmarks.part(i)
                        right_eye.append((pt.x, pt.y))
                    
                    # Mouth (points 48-68)
                    for i in range(48, 68):
                        pt = landmarks.part(i)
                        mouth.append((pt.x, pt.y))
                    
                    # Calculate EAR and MAR
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    mar = mouth_aspect_ratio(mouth)
                    
                    # Get facial state with temporal smoothing
                    facial_state = get_facial_state(right_ear, left_ear, mar, use_smoothing=True)
                    
                    # Get head pose
                    head_pose = head_pose_estimator.estimate(landmarks)
                    
                    face_metrics = {
                        "right_ear": facial_state["right_ear"],
                        "left_ear": facial_state["left_ear"],
                        "avg_ear": facial_state["avg_ear"],
                        "mar": facial_state["mar"],
                        "normalized_ear": facial_state["normalized_ear"],
                        "normalized_mar": facial_state["normalized_mar"],
                        "ear_confidence": facial_state["ear_confidence"],
                        "mar_confidence": facial_state["mar_confidence"],
                        "head_pose": head_pose.euler_angles if head_pose else None
                    }
                    
                    # Determine behavior based on face metrics with confidence thresholds
                    if facial_state["normalized_ear"] > 0.7 and facial_state["ear_confidence"] > 0.6:
                        face_metrics_behavior = "DROWSY"
                    elif facial_state["normalized_mar"] > 0.7 and facial_state["mar_confidence"] > 0.6:
                        face_metrics_behavior = "YAWNING"
                    elif head_pose and abs(head_pose.euler_angles[1]) > HEAD_POSE_THRESHOLD:
                        face_metrics_behavior = "DISTRACTED"
                    else:
                        face_metrics_behavior = "NO RISKY BEHAVIOR DETECTED"
                else:
                    logger.warning("No landmarks detected in image")
                    face_metrics = {
                        "right_ear": 0.25,
                        "left_ear": 0.25,
                        "avg_ear": 0.25,
                        "mar": 0.5,
                        "normalized_ear": 0.5,
                        "normalized_mar": 0.5,
                        "ear_confidence": 0.5,
                        "mar_confidence": 0.5,
                        "head_pose": None
                    }
                    face_metrics_behavior = "NO FACE DETECTED"
            else:
                logger.warning("No face detected in image")
                face_metrics = {
                    "right_ear": 0.25,
                    "left_ear": 0.25,
                    "avg_ear": 0.25,
                    "mar": 0.5,
                    "normalized_ear": 0.5,
                    "normalized_mar": 0.5,
                    "ear_confidence": 0.5,
                    "mar_confidence": 0.5,
                    "head_pose": None
                }
                face_metrics_behavior = "NO FACE DETECTED"
            
            # Combine predictions based on confidence
            model_behavior = model_prediction['behavior']
            model_confidence = model_prediction['scores'][model_behavior]
            
            # Get face metrics confidence
            face_metrics_confidence = 0.0
            if face_metrics_behavior == "DROWSY":
                face_metrics_confidence = face_metrics['ear_confidence']
            elif face_metrics_behavior == "YAWNING":
                face_metrics_confidence = face_metrics['mar_confidence']
            elif face_metrics_behavior == "DISTRACTED":
                face_metrics_confidence = 0.8  # High confidence for head pose detection
            
            # Combine predictions
            if model_confidence > 0.7 and face_metrics_confidence > 0.6:
                # Both systems agree with high confidence
                behavior = model_behavior
            elif model_confidence > 0.8:
                # Model has very high confidence
                behavior = model_behavior
            elif face_metrics_confidence > 0.7:
                # Face metrics has high confidence
                behavior = face_metrics_behavior
            else:
                # Default to model prediction
                behavior = model_behavior
            
            predictions.append(behavior)
            face_metrics_list.append(face_metrics)
            
            # Log metrics for debugging
            logger.info(f"Model prediction: {model_behavior} (confidence: {model_confidence:.2f})")
            logger.info(f"Face metrics prediction: {face_metrics_behavior} (confidence: {face_metrics_confidence:.2f})")
            logger.info(f"Combined prediction: {behavior}")
        
        # Generate classification report
        report = classification_report(test_labels, predictions)
        logger.info("\nClassification Report:\n" + report)
        
        # Save classification report
        save_dir = 'test_results/combined'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Generate and plot confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plot_confusion_matrix(cm, list(set(test_labels)))
        
        # Plot sample predictions
        plot_sample_predictions(test_data, test_labels, predictions, model_scores, face_metrics_list)
        
        logger.info(f"Test results saved in {save_dir} directory")
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        logger.error(f"Error during combined system evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = evaluate_combined_system()
        logger.info("Combined system evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise 
