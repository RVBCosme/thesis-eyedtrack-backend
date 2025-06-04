"""
Batch processor for analyzing driver behavior in video files or image sequences.
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import project modules
from llava import BehaviorAnalyzer
from face_analysis import FaceDetector, HeadPoseEstimator
from clahe import apply_clahe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def process_frame(
    frame: np.ndarray,
    face_detector: FaceDetector,
    head_pose_estimator: HeadPoseEstimator,
    behavior_analyzer: BehaviorAnalyzer,
    behavior_categories: List[str],
    enhance_frame: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Process a single frame for driver behavior analysis.
    Modified to first classify behavior before determining risk,
    and to skip logging normal behaviors.
    """
    start_time = time.time()
    results = {
        "timestamp": datetime.now().isoformat(),
        "faces_detected": 0,
        "face_box": None,
        "head_pose": None,
        "behavior": None,
        "behavior_confidence": 0.0,
        "is_risky": False,
        "processing_time": 0.0,
        "avg_ear": None,
        "mar": None
    }
    
    try:
        # Enhance frame if requested
        if enhance_frame:
            enhanced = apply_clahe(frame, preserve_color=True)
        else:
            enhanced = frame.copy()
        
        # Detect faces
        faces = face_detector.detect_faces(enhanced)
        results["faces_detected"] = len(faces)
        
        if faces:
            # Use largest face for analysis
            face_box = max(faces, key=lambda box: box[2] * box[3])
            x, y, w, h = face_box
            results["face_box"] = face_box
            
            # Extract face region for behavior analysis
            margin = int(max(w, h) * 0.2)  # Add margin around face for better context
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            face_region = enhanced[y1:y2, x1:x2]
            
            # FIRST: Analyze behavior using BehaviorAnalyzer
            behavior_result = behavior_analyzer.analyze(
                frame=face_region,
                behavior_categories=behavior_categories
            )
            
            if behavior_result and behavior_result.top_behaviors:
                top_behavior = behavior_result.top_behaviors[0]
                behavior_name = top_behavior.name.lower()
                
                # Skip behaviors that are not in our defined risky categories
                # or that contain "alert" or "attentive"
                if "alert" in behavior_name or "attentive" in behavior_name:
                    return None
                
                # Update results with behavior info
                results["behavior"] = top_behavior.name
                results["behavior_confidence"] = top_behavior.confidence
                
                # Explicitly determine if behavior is risky based on keywords
                results["is_risky"] = any(
                    keyword in behavior_name 
                    for keyword in ["drowsy", "distracted", "fatigue", "phone", "yawning"]
                )
                
                # Only if behavior is risky, proceed with detailed analysis
                if results["is_risky"]:
                    # Get landmarks and analyze facial features
                    landmarks = face_detector.get_landmarks(enhanced, face_box)
                    if landmarks:
                        # Analyze EAR (Eye Aspect Ratio)
                        if hasattr(landmarks, 'part'):  # dlib landmarks
                            # Extract left and right eye landmarks
                            left_eye_points = []
                            right_eye_points = []
                            
                            # Left eye (points 36-41)
                            for i in range(36, 42):
                                pt = landmarks.part(i)
                                left_eye_points.append((pt.x, pt.y))
                                
                            # Right eye (points 42-47)
                            for i in range(42, 48):
                                pt = landmarks.part(i)
                                right_eye_points.append((pt.x, pt.y))
                                
                            # Calculate EAR
                            left_ear = eye_aspect_ratio(left_eye_points)
                            right_ear = eye_aspect_ratio(right_eye_points)
                            avg_ear = (left_ear + right_ear) / 2.0
                            results["avg_ear"] = avg_ear
                            
                            # Extract mouth landmarks for MAR (points 48-68)
                            mouth_points = []
                            for i in range(48, 68):
                                pt = landmarks.part(i)
                                mouth_points.append((pt.x, pt.y))
                                
                            # Calculate MAR
                            mar = mouth_aspect_ratio(mouth_points)
                            results["mar"] = mar
                        
                        # Get head pose
                        head_pose = head_pose_estimator.estimate(landmarks)
                        if head_pose:
                            results["head_pose"] = head_pose.euler_angles
                else:
                    # Not risky behavior, don't log
                    return None
            else:
                # No behaviors detected, don't log this frame
                return None
    except Exception as e:
        logger.exception(f"Error processing frame: {str(e)}")
    
    # Calculate total processing time
    results["processing_time"] = time.time() - start_time
    
    # Only return results if a risky behavior is detected
    if results["is_risky"]:
        return results
    else:
        return None

def process_video(
    video_path: str,
    output_dir: str,
    behavior_categories: List[str],
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
    num_workers: int = 1,
    enhance_frames: bool = True,
    visualize: bool = False
) -> str:
    """
    Process a video file for driver behavior analysis.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results
        behavior_categories: List of behavior categories to analyze
        sample_rate: Process every Nth frame
        max_frames: Maximum number of frames to process (None for all)
        num_workers: Number of parallel workers
        enhance_frames: Whether to apply CLAHE enhancement
        visualize: Whether to generate visualization
        
    Returns:
        Path to the results CSV file
    """
    logger.info(f"Processing video: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    # Set the correct path to model files
    shape_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'shape_predictor_68_face_landmarks.dat')
    cnn_detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mmod_human_face_detector.dat')
    
    face_detector = FaceDetector(shape_predictor_path=shape_predictor_path, cnn_detector_path=cnn_detector_path)
    head_pose_estimator = HeadPoseEstimator()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")
    
    # Determine frames to process
    if max_frames and max_frames < frame_count:
        frames_to_process = min(max_frames, frame_count)
    else:
        frames_to_process = frame_count
    
    # Adjust for sample rate
    frames_to_process = frames_to_process // sample_rate
    
    logger.info(f"Processing {frames_to_process} frames with sample rate {sample_rate}")
    
    # Process frames
    results = []
    frames_processed = 0
    
    if num_workers > 1:
        # Parallel processing
        frame_indices = list(range(0, frame_count, sample_rate))
        if max_frames:
            frame_indices = frame_indices[:max_frames]
        
        # Pre-load frames to avoid seeking issues in parallel processing
        frames = []
        for idx in tqdm(frame_indices, desc="Loading frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        
        logger.info(f"Loaded {len(frames)} frames for parallel processing")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for idx, frame in frames:
                future = executor.submit(
                    process_frame,
                    frame,
                    face_detector,
                    head_pose_estimator,
                    behavior_analyzer,
                    behavior_categories,
                    enhance_frames
                )
                futures.append((idx, future))
            
            for idx, future in tqdm(futures, desc="Processing frames"):
                try:
                    result = future.result()
                    result["frame_number"] = idx
                    result["video_time"] = idx / fps if fps > 0 else 0
                    results.append(result)
                    frames_processed += 1
                except Exception as e:
                    logger.error(f"Error processing frame {idx}: {str(e)}")
    else:
        # Sequential processing
        with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frames_processed >= max_frames):
                    break
                
                if frame_idx % sample_rate == 0:
                    result = process_frame(
                        frame,
                        face_detector,
                        head_pose_estimator,
                        behavior_analyzer,
                        behavior_categories,
                        enhance_frames
                    )
                    
                    result["frame_number"] = frame_idx
                    result["video_time"] = frame_idx / fps if fps > 0 else 0
                    results.append(result)
                    frames_processed += 1
                    pbar.update(1)
                
                frame_idx += 1
    
    # Release video
    cap.release()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{video_name}_analysis_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Processed {frames_processed} frames, results saved to {output_file}")
    
    # Generate visualization if requested
    if visualize and not df.empty:
        visualization_file = os.path.join(output_dir, f"{video_name}_visualization_{timestamp}.png")
        generate_visualization(df, visualization_file)
        logger.info(f"Visualization saved to {visualization_file}")
    
    return output_file

def process_image_sequence(
    image_dir: str,
    output_dir: str,
    behavior_categories: List[str],
    image_pattern: str = "*.jpg",
    sample_rate: int = 1,
    max_images: Optional[int] = None,
    num_workers: int = 1,
    enhance_frames: bool = True,
    visualize: bool = False
) -> str:
    """
    Process a sequence of images for driver behavior analysis.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save results
        behavior_categories: List of behavior categories to analyze
        image_pattern: Glob pattern to match image files
        sample_rate: Process every Nth image
        max_images: Maximum number of images to process (None for all)
        num_workers: Number of parallel workers
        enhance_frames: Whether to apply CLAHE enhancement
        visualize: Whether to generate visualization
        
    Returns:
        Path to the results CSV file
    """
    import glob
    
    logger.info(f"Processing image sequence in: {image_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_pattern_path = os.path.join(image_dir, image_pattern)
    image_files = sorted(glob.glob(image_pattern_path))
    
    if not image_files:
        raise ValueError(f"No images found matching pattern: {image_pattern_path}")
    
    logger.info(f"Found {len(image_files)} images")
    
    # Apply sample rate and max_images
    image_files = image_files[::sample_rate]
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Processing {len(image_files)} images with sample rate {sample_rate}")
    
    # Initialize components
    # Set the correct path to model files
    shape_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'shape_predictor_68_face_landmarks.dat')
    cnn_detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mmod_human_face_detector.dat')
    
    face_detector = FaceDetector(shape_predictor_path=shape_predictor_path, cnn_detector_path=cnn_detector_path)
    head_pose_estimator = HeadPoseEstimator()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Process images
    results = []
    
    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for idx, image_path in enumerate(image_files):
                try:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        logger.warning(f"Could not read image: {image_path}")
                        continue
                    
                    future = executor.submit(
                        process_frame,
                        frame,
                        face_detector,
                        head_pose_estimator,
                        behavior_analyzer,
                        behavior_categories,
                        enhance_frames
                    )
                    futures.append((idx, image_path, future))
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {str(e)}")
            
            for idx, image_path, future in tqdm(futures, desc="Processing images"):
                try:
                    result = future.result()
                    result["frame_number"] = idx
                    result["image_path"] = image_path
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {str(e)}")
    else:
        # Sequential processing
        for idx, image_path in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                
                result = process_frame(
                    frame,
                    face_detector,
                    head_pose_estimator,
                    behavior_analyzer,
                    behavior_categories,
                    enhance_frames
                )
                
                result["frame_number"] = idx
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    dir_name = os.path.basename(os.path.normpath(image_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{dir_name}_analysis_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Processed {len(results)} images, results saved to {output_file}")
    
    # Generate visualization if requested
    if visualize and not df.empty:
        visualization_file = os.path.join(output_dir, f"{dir_name}_visualization_{timestamp}.png")
        generate_visualization(df, visualization_file)
        logger.info(f"Visualization saved to {visualization_file}")
    
    return output_file

def generate_visualization(df: pd.DataFrame, output_file: str) -> None:
    """
    Generate visualization of analysis results.
    
    Args:
        df: DataFrame containing analysis results
        output_file: Path to save visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Behavior distribution
    if "behavior" in df.columns:
        behavior_counts = df["behavior"].value_counts()
        sns.barplot(x=behavior_counts.index, y=behavior_counts.values, ax=axes[0, 0])
        axes[0, 0].set_title("Behavior Distribution")
        axes[0, 0].set_xlabel("Behavior")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Risk assessment
    if "is_risky" in df.columns:
        risk_counts = df["is_risky"].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[0, 1])
        axes[0, 1].set_title("Risk Assessment")
        axes[0, 1].set_xlabel("Is Risky")
        axes[0, 1].set_ylabel("Count")
    
    # 3. Confidence distribution
    if "behavior_confidence" in df.columns:
        sns.histplot(df["behavior_confidence"], bins=20, ax=axes[1, 0])
        axes[1, 0].set_title("Confidence Distribution")
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Count")
    
    # 4. Processing time
    if "processing_time" in df.columns and "frame_number" in df.columns:
        sns.lineplot(x="frame_number", y="processing_time", data=df, ax=axes[1, 1])
        axes[1, 1].set_title("Processing Time")
        axes[1, 1].set_xlabel("Frame Number")
        axes[1, 1].set_ylabel("Processing Time (s)")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def process_camera_feed(
    output_dir: str,
    behavior_categories: List[str],
    camera_id: int = 0,
    sample_rate: int = 1,
    enhance_frames: bool = True,
    visualize: bool = False,
    max_duration: Optional[int] = None,
    display_feed: bool = True
) -> str:
    """
    Process live video feed from a camera for driver behavior analysis.
    
    Args:
        output_dir: Directory to save results
        behavior_categories: List of behavior categories to analyze
        camera_id: Camera device ID (default: 0 for default camera)
        sample_rate: Process every Nth frame
        enhance_frames: Whether to apply CLAHE enhancement
        visualize: Whether to generate visualization
        max_duration: Maximum duration to record in seconds (None for unlimited)
        display_feed: Whether to display the camera feed in a window
        
    Returns:
        Path to the results CSV file
    """
    logger.info(f"Processing camera feed from camera ID: {camera_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    # Set the correct path to model files
    shape_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'shape_predictor_68_face_landmarks.dat')
    cnn_detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mmod_human_face_detector.dat')
    
    face_detector = FaceDetector(shape_predictor_path=shape_predictor_path, cnn_detector_path=cnn_detector_path)
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera with ID: {camera_id}")
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Assume 30 FPS if not available
        logger.info(f"Camera FPS not available, assuming {fps} FPS")
    else:
        logger.info(f"Camera FPS: {fps}")
    
    # Get frame size for head pose estimator
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from camera")
    
    frame_height, frame_width = first_frame.shape[:2]
    frame_size = (frame_width, frame_height)
    
    # Now initialize head pose estimator with frame size
    head_pose_estimator = HeadPoseEstimator(frame_size)
    behavior_analyzer = BehaviorAnalyzer()
    
    # Process frames
    results = []
    frames_processed = 0
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            # Check if max duration reached
            if max_duration and (time.time() - start_time) > max_duration:
                logger.info(f"Maximum duration of {max_duration} seconds reached")
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            # Process every Nth frame
            if frame_idx % sample_rate == 0:
                # Process frame
                result = process_frame(
                    frame,
                    face_detector,
                    head_pose_estimator,
                    behavior_analyzer,
                    behavior_categories,
                    enhance_frames
                )
                
                result["frame_number"] = frame_idx
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)
                frames_processed += 1
                
                # Display frame with annotations if requested
                if display_feed:
                    # Add annotations
                    annotated_frame = frame.copy()
                    
                    # Draw face box if detected
                    if result["face_box"]:
                        x, y, w, h = result["face_box"]
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add behavior text
                    if result["behavior"]:
                        behavior_text = f"{result['behavior']} ({result['behavior_confidence']:.2f})"
                        cv2.putText(
                            annotated_frame, 
                            behavior_text, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255) if result["is_risky"] else (0, 255, 0), 
                            2
                        )
                    
                    # Show frame
                    cv2.imshow("Driver Behavior Analysis", annotated_frame)
                    
                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User terminated camera feed")
                        break
            
            frame_idx += 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        # Release camera and close windows
        cap.release()
        if display_feed:
            cv2.destroyAllWindows()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = "camera_feed"  # Fixed the variable name issue
    output_file = os.path.join(output_dir, f"{dir_name}_analysis_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Processed {frames_processed} frames, results saved to {output_file}")
    
    # Generate visualization if requested
    if visualize and not df.empty:
        visualization_file = os.path.join(output_dir, f"{dir_name}_visualization_{timestamp}.png")
        generate_visualization(df, visualization_file)
        logger.info(f"Visualization saved to {visualization_file}")
    
    return output_file

def main():
    """Main entry point for the batch processor."""
    parser = argparse.ArgumentParser(description="Batch processor for driver behavior analysis")
    
    # Input type
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to video file")
    input_group.add_argument("--image-dir", type=str, help="Path to directory containing images")
    input_group.add_argument("--camera", type=int, default=None, help="Camera device ID (default: 0)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    # Processing options
    parser.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no-enhance", action="store_true", help="Disable CLAHE enhancement")
    
    # Behavior categories
    parser.add_argument("--behaviors", type=str, nargs="+", 
                        default=["distracted", "drowsy", "attentive", "looking_away"],
                        help="Behavior categories to analyze")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    
    # Image sequence options
    parser.add_argument("--image-pattern", type=str, default="*.jpg", 
                        help="Glob pattern for image files (only with --image-dir)")
    
    # Camera options
    parser.add_argument("--max-duration", type=int, help="Maximum duration to record in seconds (only with --camera)")
    parser.add_argument("--no-display", action="store_true", help="Disable camera feed display (only with --camera)")
    
    args = parser.parse_args()
    
    try:
        if args.video:
            output_file = process_video(
                video_path=args.video,
                output_dir=args.output_dir,
                behavior_categories=args.behaviors,
                sample_rate=args.sample_rate,
                max_frames=args.max_frames,
                num_workers=args.num_workers,
                enhance_frames=not args.no_enhance,
                visualize=args.visualize
            )
        elif args.image_dir:
            output_file = process_image_sequence(
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                behavior_categories=args.behaviors,
                image_pattern=args.image_pattern,
                sample_rate=args.sample_rate,
                max_images=args.max_frames,
                num_workers=args.num_workers,
                enhance_frames=not args.no_enhance,
                visualize=args.visualize
            )
        elif args.camera is not None:
            output_file = process_camera_feed(
                output_dir=args.output_dir,
                behavior_categories=args.behaviors,
                camera_id=args.camera,
                sample_rate=args.sample_rate,
                enhance_frames=not args.no_enhance,
                visualize=args.visualize,
                max_duration=args.max_duration,
                display_feed=not args.no_display
            )
        
        logger.info(f"Analysis complete. Results saved to: {output_file}")
        
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())