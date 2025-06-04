"""
Event logging module for driver monitoring system.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

def round_to_4dp(value):
    """Round a value to 4 decimal places"""
    try:
        return round(float(value), 4)
    except (ValueError, TypeError):
        return 0.0

def get_event_type(event_data: Dict[str, Any]) -> str:
    """Determine the event type based on behavior categories"""
    if not event_data.get("face_box"):
        return "no_face"
    elif event_data.get("is_drowsy", False) or event_data.get("is_risky", False):
        return "drowsy"
    elif event_data.get("is_yawning", False):
        return "yawning"
    elif event_data.get("is_distracted", False):
        return "distracted"
    else:
        return "normal"

def log_event(log_dir: Path, event_type: str, event_data: Dict[str, Any]) -> None:
    """Log only risky behavior events to a single JSON file"""
    try:
        # Create the directory if it doesn't exist
        if not isinstance(log_dir, Path):
            log_dir = Path(str(log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a single fixed filename for all events
        events_file = log_dir / 'driver_monitoring.json'
        
        # Extract required data from event_data
        timestamp = datetime.now()
        
        # Get behavior category
        behavior_category = {
            "is_drowsy": bool(event_data.get("is_drowsy", False) or event_data.get("is_risky", False)),
            "is_yawning": bool(event_data.get("is_yawning", False)),
            "is_distracted": bool(event_data.get("is_distracted", False))
        }
        
        # Get MAR and EAR values - rounded to 4 decimal places
        mar = round_to_4dp(event_data.get("mar", 0.0))
        ear = round_to_4dp(event_data.get("ear", 0.0))
        
        # Get head pose angles - rounded to 4 decimal places
        head_pose = event_data.get("head_pose", [0, 0, 0])
        if isinstance(head_pose, (list, tuple, np.ndarray)) and len(head_pose) >= 3:
            pitch, yaw, roll = head_pose[:3]
        else:
            pitch, yaw, roll = 0.0, 0.0, 0.0
            
        pitch = round_to_4dp(float(pitch))
        yaw = round_to_4dp(float(yaw))
        roll = round_to_4dp(float(roll))
        
        # Get behavior confidence - rounded to 4 decimal places
        behavior_confidence = round_to_4dp(event_data.get("behavior_confidence", 0.0))
        
        # Determine behavior output
        if not event_data.get("face_box"):
            behavior_output = "NO FACE DETECTED"
            # Don't log when no face is detected
            return
        elif any(behavior_category.values()):
            behavior_output = "RISKY BEHAVIOR DETECTED"
            # Only create and log the event if risky behavior is detected
            event = {
                "timestamp": timestamp.isoformat(),
                "behavior_category": behavior_category,
                "behavior_output": behavior_output,
                "mar": mar,
                "ear": ear,
                "pitch": pitch,
                "yaw": yaw, 
                "roll": roll,
                "behavior_confidence": behavior_confidence
            }
            
            try:
                # Append event as a single line to the log file
                with open(events_file, 'a', encoding='utf-8') as f:
                    json_str = json.dumps(event)
                    f.write(json_str + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    
            except Exception as e:
                logger.error(f"Failed to log event: {str(e)}", exc_info=True)
        else:
            # Don't log normal behavior
            return
            
    except Exception as e:
        logger.error(f"Failed to process event data: {str(e)}", exc_info=True) 