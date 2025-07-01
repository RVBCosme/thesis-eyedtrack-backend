#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EyeDTrack: Face Metrics Testing Script

This script evaluates the accuracy of face metrics-based behavior detection,
showing detailed metrics and visualizations for each behavior class.
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
    get_facial_state,
    get_eye_state,
    get_mouth_state,
    normalize_value,
    configure_thresholds
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define thresholds from face_analysis
EAR_LOWER = 0.15
EAR_UPPER = 0.3
MAR_LOWER = 0.4
MAR_UPPER = 0.9
HEAD_POSE_THRESHOLD = 30.0  # For distraction detection

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

def plot_confusion_matrix(cm, classes, title='Face Metrics Confusion Matrix'):
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
    save_dir = 'test_results/face_metrics'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(test_data, test_labels, predictions, face_metrics_list, num_samples=2):
    """Plot sample predictions for each behavior category with improved visualization"""
    save_dir = 'test_results/face_metrics'
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
            face_metrics = face_metrics_list[idx]
            
            # Create figure with reasonable size
            plt.figure(figsize=(15, 5))
            
            # Plot image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            title = f'True: {true_label}\nPredicted: {pred_label}'
            if true_label == pred_label:
                title += '\n(Correct)'
            else:
                title += '\n(Incorrect)'
            plt.title(title, fontsize=10, pad=10)
            
            # Plot face metrics
            plt.subplot(1, 3, 2)
            # Get head pose yaw (second element of the tuple)
            head_pose_yaw = face_metrics['head_pose'][1] if face_metrics['head_pose'] else 0
            # For plotting, scale yaw to 0-1 range (assuming max |yaw| is 60 degrees for visualization)
            scaled_yaw = min(abs(head_pose_yaw) / 60.0, 1.0)
            metrics = {
                'EAR': face_metrics['avg_ear'],
                'MAR': face_metrics['mar'],
                'Head Pose Yaw': scaled_yaw
            }
            bars = plt.bar(metrics.keys(), metrics.values())
            plt.title('Face Metrics', fontsize=10, pad=10)
            plt.ylim(0, 1.2)
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i == 2:  # Head Pose Yaw
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{head_pose_yaw:.1f}°', ha='center', va='bottom', fontsize=8)
                else:
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot confidence scores
            plt.subplot(1, 3, 3)
            confidences = {
                'EAR Confidence': face_metrics['ear_confidence'],
                'MAR Confidence': face_metrics['mar_confidence'],
                'Eye States': f"{face_metrics['left_eye_state']}/{face_metrics['right_eye_state']}",
                'Mouth State': face_metrics['mouth_state']
            }
            
            # Create a text box with confidence scores
            plt.text(0.1, 0.9, 'Confidence Scores:', fontsize=10, fontweight='bold')
            y_pos = 0.8
            for metric, value in confidences.items():
                if isinstance(value, (int, float)):
                    plt.text(0.1, y_pos, f'{metric}: {value:.2f}', fontsize=8)
                else:
                    plt.text(0.1, y_pos, f'{metric}: {value}', fontsize=8)
                y_pos -= 0.15
            
            plt.axis('off')
            plt.title('Detection Details', fontsize=10, pad=10)
            
            # Adjust layout with smaller margins
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
            
            # Save with specific behavior name and sample number
            save_path = os.path.join(save_dir, f'sample_prediction_{behavior}_{idx}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=150)  # Reduced DPI
            plt.close()
            
            # Log prediction details
            logger.info(f"Saved prediction for {behavior} (Sample {idx})")
            logger.info(f"True label: {true_label}")
            logger.info(f"Predicted behavior: {pred_label}")
            logger.info(f"Face metrics: {face_metrics}")
            logger.info(f"Eye states: {face_metrics['left_eye_state']}/{face_metrics['right_eye_state']}")
            logger.info(f"Mouth state: {face_metrics['mouth_state']}")
            logger.info(f"Confidence scores: EAR={face_metrics['ear_confidence']:.2f}, MAR={face_metrics['mar_confidence']:.2f}")

def evaluate_face_metrics():
    """Evaluate face metrics-based behavior detection"""
    try:
        # Set the correct path to the model files
        shape_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'shape_predictor_68_face_landmarks.dat')
        cnn_detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mmod_human_face_detector.dat')
        
        # Initialize face detector and head pose estimator with correct model paths
        face_detector = FaceDetector(
            use_cnn=True,  # Enable CNN detector
            use_media_pipe=False,  # Disable MediaPipe for now
            shape_predictor_path=shape_predictor_path,
            cnn_detector_path=cnn_detector_path
        )
        head_pose_estimator = HeadPoseEstimator(frame_size=(640, 480))
        
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
        face_metrics_list = []
        
        for img in test_data:
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
                    
                    # Get eye and mouth states
                    left_eye_state, left_eye_conf = get_eye_state(left_ear)
                    right_eye_state, right_eye_conf = get_eye_state(right_ear)
                    mouth_state, mouth_conf = get_mouth_state(mar)
                    
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
                        "left_eye_state": left_eye_state,
                        "right_eye_state": right_eye_state,
                        "mouth_state": mouth_state,
                        "head_pose": head_pose.euler_angles if head_pose else None
                    }
                    
                    # Determine behavior from face metrics independently (main.py logic order)
                    is_drowsy = ((left_eye_state in ["CLOSED", "PARTIALLY_CLOSED"] and left_eye_conf > 0.7) or
                                 (right_eye_state in ["CLOSED", "PARTIALLY_CLOSED"] and right_eye_conf > 0.7))
                    is_yawning = mouth_state == "WIDE OPEN" and mouth_conf > 0.15
                    is_distracted = head_pose and abs(head_pose.euler_angles[1]) > HEAD_POSE_THRESHOLD
                    
                    if is_yawning:
                        behavior = "YAWNING"
                    elif is_drowsy:
                        behavior = "DROWSY"
                    elif is_distracted:
                        behavior = "DISTRACTED"
                    else:
                        behavior = "NO RISKY BEHAVIOR DETECTED"
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
                        "left_eye_state": "UNKNOWN",
                        "right_eye_state": "UNKNOWN",
                        "mouth_state": "UNKNOWN",
                        "head_pose": None
                    }
                    behavior = "NO FACE DETECTED"
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
                    "left_eye_state": "UNKNOWN",
                    "right_eye_state": "UNKNOWN",
                    "mouth_state": "UNKNOWN",
                    "head_pose": None
                }
                behavior = "NO FACE DETECTED"
            
            predictions.append(behavior)
            face_metrics_list.append(face_metrics)
            
            # Log metrics for debugging
            logger.info(f"Face metrics for prediction: {face_metrics}")
            logger.info(f"Predicted behavior: {behavior}")
        
        # Generate classification report
        report = classification_report(test_labels, predictions)
        logger.info("\nClassification Report:\n" + report)
        
        # Save classification report
        save_dir = 'test_results/face_metrics'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Generate and plot confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plot_confusion_matrix(cm, list(set(test_labels)))
        
        # Plot sample predictions
        plot_sample_predictions(test_data, test_labels, predictions, face_metrics_list)
        
        logger.info(f"Test results saved in {save_dir} directory")
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        logger.error(f"Error during face metrics evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = evaluate_face_metrics()
        logger.info("Face metrics evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

"""Test face metrics calculations"""

import unittest
import numpy as np
from face_analysis.ear_mar import (
    eye_aspect_ratio, 
    mouth_aspect_ratio,
    normalize_value,
    get_facial_state,
    configure_thresholds
)

class TestFaceMetrics(unittest.TestCase):
    def setUp(self):
        # Reset thresholds to default values
        configure_thresholds(
            ear_lower=0.15,
            ear_upper=0.35,
            mar_lower=0.20,
            mar_upper=0.65
        )

    def test_eye_aspect_ratio_normal(self):
        """Test EAR calculation with normal values"""
        # Simulate eye points for open eye (EAR ≈ 0.3)
        eye_points = np.array([
            [0, 0],    # P1
            [0, 2],    # P2
            [0, 4],    # P3
            [10, 0],   # P4
            [10, 2],   # P5
            [10, 4]    # P6
        ])
        ear = eye_aspect_ratio(eye_points)
        self.assertIsNotNone(ear)
        self.assertTrue(0.15 <= ear <= 0.35)

    def test_eye_aspect_ratio_closed(self):
        """Test EAR calculation with nearly closed eye"""
        # Simulate eye points for nearly closed eye (EAR ≈ 0.15)
        eye_points = np.array([
            [0, 0],    # P1
            [0, 1],    # P2
            [0, 2],    # P3
            [10, 0],   # P4
            [10, 1],   # P5
            [10, 2]    # P6
        ])
        ear = eye_aspect_ratio(eye_points)
        self.assertIsNotNone(ear)
        self.assertGreaterEqual(ear, 0.1)
        self.assertLessEqual(ear, 0.2)

    def test_eye_aspect_ratio_invalid(self):
        """Test EAR calculation with invalid inputs"""
        # Test None input
        self.assertIsNone(eye_aspect_ratio(None))

        # Test invalid points
        invalid_points = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.assertIsNone(eye_aspect_ratio(invalid_points))

        # Test points with zero distance
        zero_distance = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ])
        self.assertIsNone(eye_aspect_ratio(zero_distance))

    def test_mouth_aspect_ratio_normal(self):
        """Test MAR calculation with normal values"""
        # Simulate mouth points for slightly open mouth (MAR ≈ 0.4)
        mouth_points = np.array([
            [0, 0],    # P1
            [0, 4],    # P2
            [0, 8],    # P3
            [20, 0],   # P4
            [20, 4],   # P5
            [20, 8]    # P6
        ])
        mar = mouth_aspect_ratio(mouth_points)
        self.assertIsNotNone(mar)
        self.assertTrue(0.2 <= mar <= 0.65)

    def test_mouth_aspect_ratio_wide(self):
        """Test MAR calculation with wide open mouth"""
        # Simulate mouth points for wide open mouth (MAR ≈ 0.6)
        mouth_points = np.array([
            [0, 0],    # P1
            [0, 6],    # P2
            [0, 12],   # P3
            [20, 0],   # P4
            [20, 6],   # P5
            [20, 12]   # P6
        ])
        mar = mouth_aspect_ratio(mouth_points)
        self.assertIsNotNone(mar)
        self.assertGreaterEqual(mar, 0.5)
        self.assertLessEqual(mar, 0.65)

    def test_mouth_aspect_ratio_invalid(self):
        """Test MAR calculation with invalid inputs"""
        # Test None input
        self.assertIsNone(mouth_aspect_ratio(None))

        # Test invalid points
        invalid_points = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.assertIsNone(mouth_aspect_ratio(invalid_points))

    def test_normalization(self):
        """Test value normalization"""
        # Test normal values
        self.assertAlmostEqual(normalize_value(0.25, 0.15, 0.35), 0.5)
        self.assertEqual(normalize_value(0.15, 0.15, 0.35), 0.0)
        self.assertEqual(normalize_value(0.35, 0.15, 0.35), 1.0)

        # Test out of bounds values
        self.assertEqual(normalize_value(0.1, 0.15, 0.35), 0.0)
        self.assertEqual(normalize_value(0.4, 0.15, 0.35), 1.0)

        # Test None value
        self.assertIsNone(normalize_value(None, 0.15, 0.35))

    def test_facial_state(self):
        """Test facial state calculation"""
        # Test normal values
        state = get_facial_state(0.25, 0.25, 0.4, use_smoothing=False)
        self.assertIsNotNone(state["avg_ear"])
        self.assertIsNotNone(state["mar"])
        self.assertTrue(0 <= state["ear_confidence"] <= 1)
        self.assertTrue(0 <= state["mar_confidence"] <= 1)

        # Test None values
        state = get_facial_state(None, None, 0.4, use_smoothing=False)
        self.assertIsNone(state["avg_ear"])
        self.assertIsNotNone(state["mar"])
        self.assertEqual(state["ear_confidence"], 0.0)

        # Test invalid values
        state = get_facial_state(0.5, 0.5, 1.0, use_smoothing=False)
        self.assertIsNotNone(state["avg_ear"])
        self.assertIsNotNone(state["mar"])

if __name__ == '__main__':
    unittest.main() 
