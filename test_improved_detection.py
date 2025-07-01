#!/usr/bin/env python3
"""
Test script for the improved dlib-based behavior detection system.
This script validates that the improvements are working correctly.
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime

from config_loader import ConfigLoader
from frame_processor import OptimizedFrameProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_improved_detection():
    """Test the improved detection system"""
    print("🔍 TESTING IMPROVED DLIB-BASED DETECTION")
    print("="*60)
    
    # Load configuration
    config_loader = ConfigLoader("config.yaml")
    config = config_loader.get_config()
    
    # Initialize frame processor
    frame_processor = OptimizedFrameProcessor(config)
    
    # Check if improved detection is enabled
    if frame_processor.use_improved_detection:
        print("✅ Improved dlib detection is ENABLED")
        print(f"📊 Detection thresholds:")
        print(f"   EAR threshold: {frame_processor.ear_threshold}")
        print(f"   MAR threshold: {frame_processor.mar_threshold}")
        print(f"   Yaw threshold: ±{frame_processor.yaw_threshold}°")
        print(f"   Pitch threshold: ±{frame_processor.pitch_threshold}°")
    else:
        print("❌ Improved dlib detection is DISABLED")
        print("   Falling back to MediaPipe detection")
    
    print("\n🎥 Starting camera test...")
    print("Instructions:")
    print("- Look normally at the camera")
    print("- Close your eyes to test drowsiness detection")
    print("- Open your mouth wide to test yawning detection")
    print("- Turn your head left/right/up/down to test distraction detection")
    print("- Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            frame_count += 1
            
            # Process frame
            result = frame_processor.process_frame(frame)
            
            # Display results
            if result['success']:
                metrics = result['metrics']
                behaviors = result['behavior_category']
                
                # Print metrics every 30 frames (about once per second)
                if frame_count % 30 == 0:
                    print(f"\n📊 Frame {frame_count} Analysis:")
                    print(f"   EAR: {metrics['ear']:.4f} (threshold: {frame_processor.ear_threshold})")
                    print(f"   MAR: {metrics['mar']:.4f} (threshold: {frame_processor.mar_threshold})")
                    
                    if metrics['head_pose'] is not None and len(metrics['head_pose']) >= 2:
                        yaw = metrics['head_pose'][1]
                        pitch = metrics['head_pose'][0]
                        print(f"   Yaw: {yaw:+6.1f}° (threshold: ±{frame_processor.yaw_threshold}°)")
                        print(f"   Pitch: {pitch:+6.1f}° (threshold: ±{frame_processor.pitch_threshold}°)")
                    
                    # Show detected behaviors
                    detected_behaviors = []
                    if behaviors['is_drowsy']:
                        detected_behaviors.append("😴 DROWSY")
                    if behaviors['is_yawning']:
                        detected_behaviors.append("🥱 YAWNING")
                    if behaviors['is_distracted']:
                        detected_behaviors.append("👀 DISTRACTED")
                    
                    if detected_behaviors:
                        print(f"   🚨 DETECTED: {' | '.join(detected_behaviors)}")
                    else:
                        print(f"   ✅ NORMAL BEHAVIOR")
                
                # Draw debug information on frame
                draw_debug_info(frame, result, frame_processor)
            else:
                # Draw error message
                cv2.putText(frame, f"ERROR: {result.get('error', 'Unknown error')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Improved Detection Test', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n📈 Test Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Duration: {elapsed_time:.2f} seconds")
        print(f"   Average FPS: {fps:.2f}")
        print(f"   Detection method: {'Improved dlib' if frame_processor.use_improved_detection else 'MediaPipe fallback'}")

def draw_debug_info(frame, result, frame_processor):
    """Draw debug information on the frame"""
    try:
        # Draw face box if detected
        if result.get('face_box'):
            x, y, w, h = result['face_box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        metrics = result.get('metrics', {})
        behaviors = result.get('behavior_category', {})
        
        # Draw metrics
        y_offset = 30
        
        # EAR
        ear_color = (0, 0, 255) if behaviors.get('is_drowsy', False) else (255, 255, 255)
        cv2.putText(frame, f"EAR: {metrics.get('ear', 0):.4f} (thresh: {frame_processor.ear_threshold})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        y_offset += 25
        
        # MAR
        mar_color = (0, 165, 255) if behaviors.get('is_yawning', False) else (255, 255, 255)
        cv2.putText(frame, f"MAR: {metrics.get('mar', 0):.4f} (thresh: {frame_processor.mar_threshold})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
        y_offset += 25
        
        # Head pose
        head_pose = metrics.get('head_pose')
        if head_pose is not None and len(head_pose) >= 2:
            pose_color = (255, 0, 0) if behaviors.get('is_distracted', False) else (255, 255, 255)
            yaw = head_pose[1]
            pitch = head_pose[0]
            cv2.putText(frame, f"Yaw: {yaw:+5.1f}° (thresh: ±{frame_processor.yaw_threshold}°)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
            y_offset += 25
            cv2.putText(frame, f"Pitch: {pitch:+5.1f}° (thresh: ±{frame_processor.pitch_threshold}°)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # Draw behavior alerts
        y_offset = frame.shape[0] - 100
        if behaviors.get('is_drowsy', False):
            cv2.putText(frame, "DROWSINESS ALERT", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
        if behaviors.get('is_yawning', False):
            cv2.putText(frame, "YAWNING ALERT", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_offset += 30
        if behaviors.get('is_distracted', False):
            cv2.putText(frame, "DISTRACTION ALERT", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show detection method
        method_text = "Improved dlib" if frame_processor.use_improved_detection else "MediaPipe"
        cv2.putText(frame, f"Method: {method_text}", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    except Exception as e:
        logger.error(f"Error drawing debug info: {e}")

if __name__ == "__main__":
    test_improved_detection() 