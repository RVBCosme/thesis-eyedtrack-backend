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
from behavior_categories import BEHAVIOR_CATEGORIES
from config_loader import load_config, DEFAULT_CONFIG_PATH
from face_analysis import FaceDetector, HeadPoseEstimator

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

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")
    logger.debug("Method: %s", request.method)
    logger.debug("Headers: %s", dict(request.headers))
    logger.debug("Remote addr: %s", request.remote_addr)
    
    if request.method == 'OPTIONS':
        logger.debug("Handling OPTIONS request for health check")
        response = make_response()
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, User-Agent'
        })
        return response
    
    try:
        response_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'message': 'Server is running',
            'remote_addr': request.remote_addr
        }
        logger.info("Health check successful: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'remote_addr': request.remote_addr
        }), 500

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame of video data"""
    global session_id
    
    try:
        # Get data from request
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({
                'success': False,
                'error': 'No frame data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Convert base64 frame to CV2 image
        frame = base64_to_cv2(data['frame'])
        
        # Process the frame
        result = frame_processor.process_frame(frame)
        
        # Add session ID to result
        result['session_id'] = session_id
        
        # Get head pose angles
        head_pose = result.get('head_pose', [0, 0, 0])
        if isinstance(head_pose, (list, tuple, np.ndarray)) and len(head_pose) >= 3:
            pitch, yaw, roll = head_pose[:3]
        else:
            pitch, yaw, roll = 0.0, 0.0, 0.0
        
        # Determine behavior output
        behavior_category = {
            'is_drowsy': bool(result.get('is_drowsy', False)),
            'is_yawning': bool(result.get('is_yawning', False)),
            'is_distracted': bool(result.get('is_distracted', False))
        }
        
        if not result.get('face_box'):
            behavior_output = "NO FACE DETECTED"
        elif any(behavior_category.values()):
            behavior_output = "RISKY BEHAVIOR DETECTED"
        else:
            behavior_output = "NO RISKY BEHAVIOR DETECTED"
        
        # Format response according to desired structure
        response = {
            'timestamp': datetime.now().isoformat(),
            'behavior_category': behavior_category,
            'behavior_output': behavior_output,
            'mar': float(result.get('mar', 0.0)),
            'ear': float(result.get('ear', 0.0)),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'behavior_confidence': float(result.get('behavior_confidence', 0.0)),
            'session_id': session_id
        }
        
        # Log the event if logging is enabled
        if config['logging']['log_events']:
            try:
                # Log the event to the appropriate event file
                log_event(log_dir, 'frame_processed', result)
            except Exception as log_error:
                logger.error(f"Failed to log event: {log_error}")
                logger.error(traceback.format_exc())
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_msg,
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