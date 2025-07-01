#!/usr/bin/env python3
"""
Direct debug test for frame processing pipeline
"""

import cv2
import numpy as np
from frame_processor import OptimizedFrameProcessor
from config_loader import load_config
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_frame_processing():
    """Test frame processing directly"""
    print("=== Direct Frame Processing Test ===")
    
    # Load config
    config = load_config()
    
    # Initialize frame processor
    processor = OptimizedFrameProcessor(config)
    
    # Capture a frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame")
        return
    
    print(f"Frame captured: {frame.shape}")
    
    # Process frame directly
    result = processor.process_frame(frame)
    
    print("\n=== RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"Face detected: {result['face_box'] is not None}")
    print(f"EAR: {result['metrics']['ear']}")
    print(f"MAR: {result['metrics']['mar']}")
    print(f"Head pose: {result['metrics']['head_pose']}")
    print(f"Behaviors: {result['behavior_category']}")
    print(f"Timestamp: {result['timestamp']}")

if __name__ == "__main__":
    test_frame_processing() 