"""Mouth analysis module for driver monitoring"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def mouth_aspect_ratio(mouth_points):
    """Calculate mouth aspect ratio from mouth landmark points"""
    try:
        # Validate input
        if not isinstance(mouth_points, np.ndarray) or mouth_points.shape[0] != 12:
            return 0.0
            
        # Compute vertical distances
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 51-59
        B = np.linalg.norm(mouth_points[3] - mouth_points[9])   # 52-58
        C = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 53-57
        
        # Compute horizontal distance
        D = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 49-55
        
        # Calculate MAR
        if D == 0:
            return 0.0
            
        mar = (A + B + C) / (3.0 * D)
        return float(mar)
        
    except Exception as e:
        logger.error(f"Error calculating mouth aspect ratio: {e}")
        return 0.0

def get_mouth_state(mar, threshold_closed=0.4, threshold_open=0.7):
    """Determine mouth state from mouth aspect ratio"""
    try:
        if mar <= threshold_closed:
            return "CLOSED", 1.0 - (mar / threshold_closed)
        elif mar >= threshold_open:
            return "WIDE OPEN", mar / threshold_open
        else:
            # Partially open
            ratio = (mar - threshold_closed) / (threshold_open - threshold_closed)
            return "PARTIALLY OPEN", ratio
            
    except Exception as e:
        logger.error(f"Error determining mouth state: {e}")
        return "UNKNOWN", 0.0 