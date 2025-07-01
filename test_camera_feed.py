#!/usr/bin/env python3
"""
Test script to send camera frames to the backend API for processing.
This simulates what the Android app should be doing.
"""

import cv2
import base64
import requests
import time
import json
from datetime import datetime

def capture_and_send_frame(api_url="http://localhost:5000/api/process_frame"):
    """Capture a frame from camera and send to backend API"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    try:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            return False
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare request data
        data = {
            'frame': frame_base64,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to API
        response = requests.post(api_url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Frame processed successfully")
            print(f"  EAR: {result.get('metrics', {}).get('ear', 'N/A')}")
            print(f"  MAR: {result.get('metrics', {}).get('mar', 'N/A')}")
            print(f"  Behaviors: {result.get('behavior_category', {})}")
            return True
        else:
            print(f"✗ API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        cap.release()

def continuous_feed(api_url="http://localhost:5000/api/process_frame", interval=1.0):
    """Send continuous camera feed to backend API"""
    print(f"Starting continuous camera feed to {api_url}")
    print(f"Sending frame every {interval} seconds")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            success = capture_and_send_frame(api_url)
            if success:
                print(f"Frame sent at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopping camera feed...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            # Send single frame
            print("Sending single frame...")
            capture_and_send_frame()
        elif sys.argv[1] == "continuous":
            # Send continuous feed
            continuous_feed()
        else:
            print("Usage: python test_camera_feed.py [single|continuous]")
    else:
        # Default: send single frame
        print("Sending single test frame...")
        capture_and_send_frame() 