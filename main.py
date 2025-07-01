#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EyeDTrack: Real-Time Driver Attention Monitoring System Backend API
"""

import cv2
import numpy as np
import time
import os
import logging
import base64
from datetime import datetime
from pathlib import Path
import sys
import traceback
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json
from flask_compress import Compress
from typing import Dict, Any, Optional

# Import project modules
from frame_processor import OptimizedFrameProcessor
from event_logger import log_event
from config_loader import load_config, DEFAULT_CONFIG_PATH
# ImprovedFaceAnalyzer is imported by frame_processor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Configure CORS with all methods and headers allowed
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "User-Agent"],
        "expose_headers": ["Content-Type", "Authorization"],
        
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Add response compression
Compress(app)

@app.before_request
def log_request_info():
    """Log details about incoming requests"""
    logger.debug("Request Headers: %s", dict(request.headers))
    logger.debug("Request Method: %s", request.method)
    logger.debug("Request URL: %s", request.url)
    logger.debug("Request Remote Addr: %s", request.remote_addr)
    if request.method == 'OPTIONS':
        logger.debug("Handling OPTIONS request")

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, User-Agent',
        'Access-Control-Expose-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '3600',
    }
    
    for key, value in headers.items():
        response.headers.add(key, value)
    
    logger.debug("Response Status: %s", response.status)
    logger.debug("Response Headers: %s", dict(response.headers))
    return response

def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        return img
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        raise

def initialize_system(config_path=DEFAULT_CONFIG_PATH):
    """Initialize the driver monitoring system"""
    global frame_processor, config, log_dir, session_id
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Configure logging
        logging.getLogger().setLevel(getattr(logging, config["logging"]["level"]))
        
        # Create log directory
        log_dir = Path(config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video directory
        video_dir = log_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize frame processor
        frame_processor = OptimizedFrameProcessor(config)
        
        logger.info(f"Driver monitoring system initialized. Using log directory: {log_dir}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}", exc_info=True)
        raise

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'message': 'Server is running',
            'remote_addr': request.remote_addr
        }
        logger.info(f"Health check successful: {health_data}")
        return jsonify(health_data), 200
    except Exception as e:
        error_data = {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'message': str(e),
            'remote_addr': request.remote_addr
        }
        logger.error(f"Health check failed: {error_data}")
        return jsonify(error_data), 500

@app.route('/api/test_behavior', methods=['GET'])
def test_behavior():
    """Test endpoint that always returns drowsy behavior"""
    logger.info("🔴 TEST ENDPOINT: Always serving DROWSY behavior")
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'behavior_flags': {
            'is_drowsy': True,
            'is_yawning': False,
            'is_distracted': False
        },
        'metrics': {
            'ear': 0.15,  # Low EAR indicates drowsiness
            'mar': 0.6,
            'pitch': 0.0,
            'yaw': 0.0
        },
        'behavior_output': 'TEST ENDPOINT - DROWSY ALWAYS ACTIVE',
        'behavior_confidence': 1.0
    }), 200

@app.route('/api/latest_behavior', methods=['GET'])
def get_latest_behavior():
    """Get the latest behavior detection from driver_monitoring.json"""
    try:
        log_file_path = os.path.join(str(log_dir), "driver_monitoring.json")
        
        if not os.path.exists(log_file_path):
            logger.warning(f"Driver monitoring log file not found: {log_file_path}")
            return jsonify({
                'success': False,
                'error': 'No monitoring data available',
                'behavior_category': {
                    'is_drowsy': False,
                    'is_yawning': False,
                    'is_distracted': False
                }
            }), 404
        
        # Read the latest entry from the log file
        latest_entry = None
        try:
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
                # Get the last non-empty line
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        latest_entry = json.loads(line)
                        break
        except Exception as e:
            logger.error(f"Error reading driver monitoring log: {e}")
            return jsonify({
                'success': False,
                'error': f'Error reading log file: {str(e)}',
                'behavior_category': {
                    'is_drowsy': False,
                    'is_yawning': False,
                    'is_distracted': False
                }
            }), 500
        
        if not latest_entry:
            logger.warning("No entries found in driver monitoring log")
            return jsonify({
                'success': False,
                'error': 'No monitoring entries found',
                'behavior_category': {
                    'is_drowsy': False,
                    'is_yawning': False,
                    'is_distracted': False
                }
            }), 404
        
        # Extract behavior information
        behavior_category = latest_entry.get("behavior_category", {})
        timestamp = latest_entry.get("timestamp", datetime.now().isoformat())
        
        # Log the behavior detection with detailed metrics
        is_drowsy = behavior_category.get("is_drowsy", False)
        is_yawning = behavior_category.get("is_yawning", False)
        is_distracted = behavior_category.get("is_distracted", False)
        
        # Extract metrics for detailed logging
        metrics = latest_entry.get("metrics", {})
        ear = metrics.get("ear", 0)
        mar = metrics.get("mar", 0)
        head_pose = metrics.get("head_pose", [0, 0, 0])
        yaw = head_pose[1] if len(head_pose) > 1 else 0
        pitch = head_pose[0] if len(head_pose) > 0 else 0
        
        if is_drowsy or is_yawning or is_distracted:
            behaviors = []
            if is_drowsy: behaviors.append("DROWSY")
            if is_yawning: behaviors.append("YAWNING")
            if is_distracted: behaviors.append("DISTRACTED")
            logger.warning(f"🚨 RISKY BEHAVIOR DETECTED: {', '.join(behaviors)} | EAR={ear:.3f} MAR={mar:.3f} Yaw={yaw:.1f}° Pitch={pitch:.1f}°")
        else:
            logger.debug(f"✅ Normal behavior | EAR={ear:.3f} MAR={mar:.3f} Yaw={yaw:.1f}° Pitch={pitch:.1f}°")
        
        return jsonify({
            'success': True,
            'behavior_category': behavior_category,
            'behavior_confidence': latest_entry.get("behavior_confidence", 0.0),
            'timestamp': timestamp,
            'metrics': latest_entry.get("metrics", {}),
            'entry_age_seconds': (datetime.now() - datetime.fromisoformat(timestamp.replace('Z', '+00:00').split('.')[0])).total_seconds() if timestamp else 0
        }), 200
            
    except Exception as e:
        logger.error(f"Error getting latest behavior: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'behavior_category': {
                'is_drowsy': False,
                'is_yawning': False,
                'is_distracted': False
            }
        }), 500

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame of video data"""
    global session_id
    
    logger.debug("Received frame processing request")
    logger.debug("Content-Type: %s", request.content_type)
    logger.debug("Content-Length: %s", request.content_length)
    
    try:
        # Get data from request
        data = request.get_json()
        logger.debug("Received data keys: %s", list(data.keys()) if data else None)
        
        if not data or 'frame' not in data:
            logger.error("No frame data provided in request")
            return jsonify({
                'success': False,
                'error': 'No frame data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Log frame data length
        frame_data = data['frame']
        logger.debug("Frame data length: %d", len(frame_data) if frame_data else 0)

        # Convert base64 frame to CV2 image
        frame = base64_to_cv2(data['frame'])
        logger.debug("Successfully converted frame to CV2 image. Shape: %s", frame.shape if frame is not None else None)
        
        # Process the frame
        result = frame_processor.process_frame(frame)
        logger.debug("Frame processing result: %s", result)
        
        # Format behaviors for response
        behaviors = []
        if result.get('is_drowsy', False):
            behaviors.append('drowsy')
        if result.get('is_yawning', False):
            behaviors.append('yawning')
        if result.get('is_distracted', False):
            behaviors.append('distracted')
            
        # Log events if behaviors detected
        if behaviors:
            log_event({
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'behaviors': behaviors,
                'metrics': {
                    'ear': result.get('ear', 0),
                    'mar': result.get('mar', 0),
                    'head_pose': result.get('head_pose', None)
                }
            })
            
        # Format response for frontend
        response = {
            'success': True,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'behaviors': behaviors,
            'metrics': {
                'ear': result.get('ear', 0),
                'mar': result.get('mar', 0),
                'head_pose': result.get('head_pose', None)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        logger.error(f"Request data: {request.get_data(as_text=True)}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/process_frame', methods=['POST'])
def process_frame_redirect():
    """Redirect /process_frame to /api/process_frame for backward compatibility"""
    return process_frame()

if __name__ == '__main__':
    try:
        # Initialize the system
        config = initialize_system()
        
        # Start the server
        host = config["integration"]["api"]["host"]
        port = config["integration"]["api"]["port"]
        
        logger.info(f"Starting server on {host}:{port}")
        app.run(host=host, port=port, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)