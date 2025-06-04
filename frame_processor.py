"""
Frame processor module for driver monitoring system.
"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
import os
from typing import Dict, Any, Optional, Tuple, List, Deque
from collections import deque
import traceback

from face_analysis import FaceDetector, HeadPoseEstimator
from face_analysis import eye_aspect_ratio, mouth_aspect_ratio, get_eye_state, get_mouth_state
from face_analysis import weighted_temporal_smoothing
from behavior_categories import BEHAVIOR_CATEGORIES
from video_recorder import VideoRecorder

logger = logging.getLogger(__name__)

# Define landmark indices for eye and mouth measurements
# These indices are based on dlib's 68-point facial landmark detector
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
MOUTH_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

class OptimizedFrameProcessor:
    def __init__(self, config):
        """Initialize the frame processor with configuration"""
        self.config = config
        
        # Initialize face detector with proper configuration
        self.face_detector = FaceDetector(config)
        
        # Initialize head pose estimator
        self.head_pose_estimator = HeadPoseEstimator(config)
        
        # Initialize state tracking with configurable thresholds
        self._drowsy_frames = 0
        self._yawn_frames = 0
        self._distraction_frames = 0
        self._last_behavior = "NO BEHAVIOR DETECTED"
        self._behavior_confidence = 0.0
        
        # Get thresholds from config
        self.ear_threshold = config["thresholds"]["ear_lower"]
        self.mar_threshold = config["thresholds"]["mar_upper"]
        self.drowsy_frames_threshold = config["thresholds"]["drowsy_frames"]
        self.yawn_frames_threshold = config["thresholds"]["yawn_frames"]
        self.distraction_frames_threshold = config["thresholds"]["distraction_frames"]
        
        # Head pose thresholds
        self.yaw_threshold = config["thresholds"]["yaw_threshold"]
        self.pitch_threshold = config["thresholds"]["pitch_threshold"]
        self.roll_threshold = config["thresholds"]["roll_threshold"]
        
        # Head pose smoothing
        self.last_head_pose = None
        self.max_angle_change = config["head_pose"].get("max_angle_change", 15.0)
        self.use_angle_smoothing = config["head_pose"].get("angle_smoothing", True)
        self.head_pose_history = deque(maxlen=5)
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps = 0.0
        self.frame_skip_counter = 0
        self.frame_skip_threshold = config["performance"]["skip_frames"]
        self.min_frame_interval = 1.0 / config["camera"]["fps"]
        self.last_processed_time = time.time()
        
        # Face tracking with improved robustness
        self.face_tracking_failed = 0
        self.max_tracking_failures = 10
        self.last_face_box = None
        
        # Temporal smoothing for stability
        self.use_temporal_smoothing = config["behavior"]["temporal_smoothing"]
        self.smoothing_window = config["behavior"]["smoothing_window"]
        self.behavior_confidence_history = deque(maxlen=self.smoothing_window)
        
        # Initialize video recorder for risky behavior monitoring
        video_output_dir = os.path.join(config["logging"]["log_dir"], "videos")
        self.video_recorder = VideoRecorder(video_output_dir)
        logger.info(f"Video recorder initialized with output directory: {video_output_dir}")
            
        logger.info("Frame processor initialized with configuration")
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame with comprehensive analysis"""
        try:
            # Validate input frame
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.error("Invalid input frame")
                return {
                    "error": "Invalid input frame",
                    "capture_fps": self.calculate_fps()
                }

            # Initialize result dictionary
            result = {
                "timestamp": time.time(),
                "face_box": None,
                "landmarks": None,
                "head_pose": None,
                "ear": 0.0,
                "mar": 0.0,
                "is_drowsy": False,
                "is_yawning": False,
                "is_distracted": False,
                "behavior_confidence": 0.0,
                "behavior": self._last_behavior,
                "fps": self.calculate_fps()
            }
            
            # Apply CLAHE if enabled
            if self.config["clahe"]["enabled"]:
                frame = self.apply_clahe(frame)
            
            # Detect face with fallback modes
            face_box = self.face_detector.detect(frame)
            result["face_box"] = face_box
            
            if face_box is not None:
                # Reset tracking failures
                self.face_tracking_failed = 0
                self.last_face_box = face_box
                
                # Get face landmarks
                landmarks = self.face_detector.get_landmarks(frame, face_box)
                result["landmarks"] = landmarks
                
                if landmarks is not None:
                    # Calculate eye aspect ratio (EAR)
                    left_eye, right_eye = self.extract_eye_landmarks(landmarks)
                    if left_eye is not None and right_eye is not None:
                        ear_left = eye_aspect_ratio(left_eye)
                        ear_right = eye_aspect_ratio(right_eye)
                        ear = (ear_left + ear_right) / 2.0
                        result["ear"] = ear
                        
                        # Check for drowsiness - more sensitive detection
                        if ear < self.ear_threshold:
                            self._drowsy_frames += 2  # Increment by 2 to detect drowsiness faster
                            if self._drowsy_frames >= self.drowsy_frames_threshold:
                                result["is_drowsy"] = True
                                self._drowsy_frames = self.drowsy_frames_threshold  # Cap at threshold
                        else:
                            self._drowsy_frames = max(0, self._drowsy_frames - 1)  # Gradual decrease
                    
                    # Calculate mouth aspect ratio (MAR)
                    mouth = self.extract_mouth_landmarks(landmarks)
                    if mouth is not None:
                        mar = mouth_aspect_ratio(mouth)
                        result["mar"] = mar
                        
                        # Check for yawning - more sensitive detection
                        if mar > self.mar_threshold:
                            self._yawn_frames += 2  # Increment by 2 to detect yawning faster
                            if self._yawn_frames >= self.yawn_frames_threshold:
                                result["is_yawning"] = True
                                self._yawn_frames = self.yawn_frames_threshold  # Cap at threshold
                        else:
                            self._yawn_frames = max(0, self._yawn_frames - 1)  # Gradual decrease
                    
                    # Calculate head pose
                    head_pose = self.head_pose_estimator.estimate(frame, landmarks)
                    if head_pose is not None:
                        result["head_pose"] = head_pose
                        
                        # Check for distraction
                        if self.is_distracted(head_pose):
                            self._distraction_frames += 1
                            if self._distraction_frames >= self.distraction_frames_threshold:
                                result["is_distracted"] = True
                                self._distraction_frames = self.distraction_frames_threshold  # Cap at threshold
                        else:
                            self._distraction_frames = max(0, self._distraction_frames - 1)  # Gradual decrease
            else:
                # Reset all counters when no face is detected
                self.face_tracking_failed += 1
                if self.face_tracking_failed > self.max_tracking_failures:
                    self.last_face_box = None
                    self._drowsy_frames = 0
                    self._yawn_frames = 0
                    self._distraction_frames = 0
            
            # Update behavior confidence with temporal smoothing
            behavior_confidence = 0.0
            if result["is_drowsy"]:
                behavior_confidence = max(behavior_confidence, self._drowsy_frames / self.drowsy_frames_threshold)
            if result["is_yawning"]:
                behavior_confidence = max(behavior_confidence, self._yawn_frames / self.yawn_frames_threshold)
            if result["is_distracted"]:
                behavior_confidence = max(behavior_confidence, self._distraction_frames / self.distraction_frames_threshold)
            
            if self.use_temporal_smoothing:
                behavior_confidence = self.smooth_confidence(behavior_confidence)
            
            result["behavior_confidence"] = behavior_confidence
            
            # Update behavior status
            if face_box is None:
                self._last_behavior = "NO FACE DETECTED"
            elif behavior_confidence > self.config["behavior"]["confidence_threshold"]:
                if result["is_drowsy"]:
                    self._last_behavior = "DROWSY"
                elif result["is_yawning"]:
                    self._last_behavior = "YAWNING"
                elif result["is_distracted"]:
                    self._last_behavior = "DISTRACTED"
            else:
                self._last_behavior = "NO BEHAVIOR DETECTED"
            
            result["behavior"] = self._last_behavior
            
            # Record video if enabled and risky behavior is detected
            if self.video_recorder:
                try:
                    is_risky = any([
                        result["is_drowsy"],
                        result["is_yawning"],
                        result["is_distracted"]
                    ])
                    
                    if is_risky:
                        logger.info("Risky behavior detected in frame processor")
                        behavior_type = None
                        if result["is_drowsy"]:
                            behavior_type = "drowsy"
                        elif result["is_yawning"]:
                            behavior_type = "yawning"
                        elif result["is_distracted"]:
                            behavior_type = "distracted"
                        logger.info(f"Behavior type: {behavior_type}")
                    
                    success = self.video_recorder.write_frame(frame, is_risky)
                    if not success and is_risky:
                        logger.error("Failed to write frame during risky behavior")
                except Exception as e:
                    logger.error(f"Error in video recording: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Update performance metrics
            self.frame_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "capture_fps": self.calculate_fps()
            }
    
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for improved image quality"""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=self.config["clahe"]["clip_limit"],
                tileGridSize=tuple(self.config["clahe"]["base_tile_grid_size"])
            )
            
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"CLAHE application failed: {e}")
            return frame
            
    def smooth_head_pose(self, head_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply smoothing to head pose angles"""
        if not self.use_angle_smoothing or not head_pose:
            return head_pose
            
        yaw, pitch, roll = head_pose
        
        # Add to history
        self.head_pose_history.append(head_pose)
        
        if len(self.head_pose_history) < 2:
            return head_pose
            
        # Calculate mean of recent poses
        poses = np.array(self.head_pose_history)
        mean_pose = np.mean(poses, axis=0)
        
        # Limit maximum change from previous pose
        if self.last_head_pose is not None:
            last_yaw, last_pitch, last_roll = self.last_head_pose
            
            # Limit each angle change
            yaw = np.clip(mean_pose[0], last_yaw - self.max_angle_change, last_yaw + self.max_angle_change)
            pitch = np.clip(mean_pose[1], last_pitch - self.max_angle_change, last_pitch + self.max_angle_change)
            roll = np.clip(mean_pose[2], last_roll - self.max_angle_change, last_roll + self.max_angle_change)
            
            smoothed_pose = (yaw, pitch, roll)
        else:
            smoothed_pose = tuple(mean_pose)
            
        self.last_head_pose = smoothed_pose
        return smoothed_pose

    def is_distracted(self, head_pose: Tuple[float, float, float]) -> bool:
        """Check if the driver is distracted based on head pose angles"""
        try:
            if head_pose is None:
                return False
                
            # Apply smoothing to head pose
            smoothed_pose = self.smooth_head_pose(head_pose)
            if smoothed_pose is None:
                return False
                
            yaw, pitch, roll = smoothed_pose
            
            # Calculate the percentage of threshold exceeded for each angle
            yaw_exceeded = max(0, (abs(yaw) - self.yaw_threshold * 0.5)) / (self.yaw_threshold * 0.5)
            pitch_exceeded = max(0, (abs(pitch) - self.pitch_threshold * 0.5)) / (self.pitch_threshold * 0.5)
            roll_exceeded = max(0, (abs(roll) - self.roll_threshold * 0.5)) / (self.roll_threshold * 0.5)
            
            # Consider distracted only if angles significantly exceed thresholds
            angle_violation = max(yaw_exceeded, pitch_exceeded, roll_exceeded)
            
            # Must exceed at least 50% of threshold to be considered distracted
            return angle_violation > 0.5
            
        except Exception as e:
            logger.warning(f"Head pose analysis failed: {e}")
            return False
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
        
        return self.fps
    
    def extract_eye_landmarks(self, landmarks):
        """Extract eye landmarks from the full set of landmarks"""
        try:
            # For dlib landmarks
            if hasattr(landmarks, 'part'):
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES]
                return left_eye, right_eye
            # For numpy array landmarks
            elif isinstance(landmarks, np.ndarray):
                left_eye = landmarks[LEFT_EYE_INDICES]
                right_eye = landmarks[RIGHT_EYE_INDICES]
                return left_eye, right_eye
            # For other types
            else:
                logger.warning(f"Unsupported landmark type: {type(landmarks)}")
                return None, None
        except Exception as e:
            logger.error(f"Error extracting eye landmarks: {e}")
            return None, None
    
    def extract_mouth_landmarks(self, landmarks):
        """Extract mouth landmarks from the full set of landmarks"""
        try:
            # For dlib landmarks
            if hasattr(landmarks, 'part'):
                mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH_INDICES]
                return mouth
            # For numpy array landmarks
            elif isinstance(landmarks, np.ndarray):
                mouth = landmarks[MOUTH_INDICES]
                return mouth
            # For other types
            else:
                logger.warning(f"Unsupported landmark type: {type(landmarks)}")
                return None
        except Exception as e:
            logger.error(f"Error extracting mouth landmarks: {e}")
            return None
    
    def smooth_confidence(self, confidence: float) -> float:
        """Apply temporal smoothing to behavior confidence value"""
        if not self.use_temporal_smoothing:
            return confidence
            
        # Add current confidence to history
        self.behavior_confidence_history.append(confidence)
        
        # Apply weighted smoothing
        return weighted_temporal_smoothing(
            confidence, 
            self.behavior_confidence_history, 
            alpha=0.3
        )
    
    def run(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                result = self.process_frame(frame)
                
                # Put result in queue
                self.result_queue.put(result)
                
                # Update frame count for FPS calculation
                self.frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error in frame processing loop: {str(e)}")
                continue
                
        # Clean up video recorder when stopping
        try:
            if self.video_recorder.is_recording:
                self.video_recorder.stop_recording()
        except Exception as e:
            logger.error(f"Error stopping video recorder: {str(e)}") 