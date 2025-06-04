import numpy as np
import logging
from collections import deque

# Configure module-level logger
logger = logging.getLogger(__name__)

# Global variables for EAR/MAR normalization
_ear_lower = 0.18
_ear_upper = 0.45
_mar_lower = 0.20
_mar_upper = 0.80

# Smoothing buffers
_ear_buffer = deque(maxlen=10)
_mar_buffer = deque(maxlen=10)

def configure_thresholds(ear_lower=0.18, ear_upper=0.45, mar_lower=0.20, mar_upper=0.80):
    """Configure the thresholds for EAR and MAR normalization"""
    global _ear_lower, _ear_upper, _mar_lower, _mar_upper
    _ear_lower = ear_lower
    _ear_upper = ear_upper
    _mar_lower = mar_lower
    _mar_upper = mar_upper

def weighted_temporal_smoothing(new_value, history, alpha=0.3):
    """
    Apply weighted smoothing with more weight on recent values
    
    Args:
        new_value: The newest measurement
        history: List of previous measurements
        alpha: Weight factor for new value (0.3 provides good balance)
        
    Returns:
        Smoothed value
    """
    if not history:
        return new_value
    
    smoothed = alpha * new_value + (1 - alpha) * history[-1]
    return float(smoothed)

def get_point_coordinates(points, index):
    """
    Extract point coordinates from various landmark formats.
    Works with dlib landmarks, numpy arrays, and lists of (x,y) tuples.
    """
    try:
        # For dlib landmarks with part() method
        if hasattr(points, 'part'):
            pt = points.part(index)
            return (pt.x, pt.y)
        # For numpy array
        elif isinstance(points, np.ndarray):
            if len(points.shape) == 2 and points.shape[1] >= 2:
                return (points[index, 0], points[index, 1])
        # For list/tuple of (x,y) coordinates
        elif isinstance(points, (list, tuple)) and index < len(points):
            pt = points[index]
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                return (pt[0], pt[1])
    except Exception as e:
        logger.error(f"Error getting point coordinates at index {index}: {e}")
    
    # Return (0,0) if we can't get the coordinates
    return (0, 0)

def eye_aspect_ratio(eye_points):
    """
    Calculate the eye aspect ratio for a single eye.
    
    Args:
        eye_points: List of 6 points representing the eye landmarks,
                   or a dlib.full_object_detection object.
    
    Returns:
        float: The eye aspect ratio, or 0.25 (neutral value) if calculation fails.
    """
    try:
        # Validate input
        if eye_points is None:
            logger.warning("Eye points are None")
            return 0.25

        # Extract points with validation
        points = []
        try:
            if hasattr(eye_points, 'part'):
                points = [(eye_points.part(i).x, eye_points.part(i).y) for i in range(6)]
            elif isinstance(eye_points, (list, tuple, np.ndarray)):
                if isinstance(eye_points, np.ndarray):
                    if len(eye_points.shape) == 2 and eye_points.shape[1] >= 2:
                        points = [(eye_points[i, 0], eye_points[i, 1]) for i in range(6)]
                    else:
                        logger.warning(f"Invalid eye_points numpy array shape: {eye_points.shape}")
                        return 0.25
                else:
                    points = eye_points[:6]
            else:
                logger.warning(f"Unsupported eye_points type: {type(eye_points)}")
                return 0.25
        except Exception as e:
            logger.error(f"Error extracting eye points: {e}")
            return 0.25

        # Validate points
        if len(points) != 6:
            logger.warning(f"Invalid number of eye points: {len(points)}")
            return 0.25

        # Convert points to numpy arrays with validation
        try:
            points = np.array(points, dtype=np.float64)
            if not np.all(np.isfinite(points)):
                logger.warning("Non-finite values in eye points")
                return 0.25
        except Exception as e:
            logger.error(f"Error converting points to numpy array: {e}")
            return 0.25

        # Calculate distances with validation
        try:
            # Vertical distances
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            # Horizontal distance
            C = np.linalg.norm(points[0] - points[3])

            # Validate distances
            if not all(np.isfinite([A, B, C])) or any(x <= 0 for x in [A, B, C]):
                logger.warning("Invalid distances calculated for EAR")
                return 0.25

            # Compute EAR with validation
            ear = (A + B) / (2.0 * C)
            if not np.isfinite(ear) or ear <= 0:
                logger.warning(f"Invalid EAR value calculated: {ear}")
                return 0.25

            return float(ear)
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.25

    except Exception as e:
        logger.error(f"Error in eye_aspect_ratio: {e}")
        return 0.25

def mouth_aspect_ratio(mouth_points):
    """
    Calculate the mouth aspect ratio.
    
    Args:
        mouth_points: List of 12 points representing the mouth landmarks,
                     or a dlib.full_object_detection object.
    
    Returns:
        float: The mouth aspect ratio, or 0.5 (neutral value) if calculation fails.
    """
    try:
        # Validate input
        if mouth_points is None:
            logger.warning("Mouth points are None")
            return 0.5

        # Extract points with validation
        points = []
        try:
            if hasattr(mouth_points, 'part'):
                # Get specific points for MAR calculation (48, 50, 52, 54, 56, 58)
                indices = [48, 50, 52, 54, 56, 58]
                points = [(mouth_points.part(i).x, mouth_points.part(i).y) for i in indices]
            elif isinstance(mouth_points, (list, tuple, np.ndarray)):
                if isinstance(mouth_points, np.ndarray):
                    if len(mouth_points.shape) == 2 and mouth_points.shape[1] >= 2:
                        # Get specific points for MAR calculation
                        indices = [0, 2, 4, 6, 8, 10]
                        points = [(mouth_points[i, 0], mouth_points[i, 1]) for i in indices]
                    else:
                        logger.warning(f"Invalid mouth_points numpy array shape: {mouth_points.shape}")
                        return 0.5
                else:
                    # Get specific points for MAR calculation
                    indices = [0, 2, 4, 6, 8, 10]
                    points = [mouth_points[i] for i in indices]
            else:
                logger.warning(f"Unsupported mouth_points type: {type(mouth_points)}")
                return 0.5
        except Exception as e:
            logger.error(f"Error extracting mouth points: {e}")
            return 0.5

        # Validate points
        if len(points) != 6:
            logger.warning(f"Invalid number of mouth points: {len(points)}")
            return 0.5

        # Convert points to numpy arrays with validation
        try:
            points = np.array(points, dtype=np.float64)
            if not np.all(np.isfinite(points)):
                logger.warning("Non-finite values in mouth points")
                return 0.5
        except Exception as e:
            logger.error(f"Error converting points to numpy array: {e}")
            return 0.5

        # Calculate distances with validation
        try:
            # Vertical distances
            A = np.linalg.norm(points[1] - points[4])
            B = np.linalg.norm(points[2] - points[5])
            # Horizontal distance
            C = np.linalg.norm(points[0] - points[3])

            # Validate distances
            if not all(np.isfinite([A, B, C])) or any(x <= 0 for x in [A, B, C]):
                logger.warning("Invalid distances calculated for MAR")
                return 0.5

            # Compute MAR with validation
            mar = (A + B) / (2.0 * C)
            if not np.isfinite(mar) or mar <= 0:
                logger.warning(f"Invalid MAR value calculated: {mar}")
                return 0.5

            return float(mar)
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.5

    except Exception as e:
        logger.error(f"Error in mouth_aspect_ratio: {e}")
        return 0.5

def normalize_value(value, lower, upper):
    """Normalize a value between 0 and 1 based on lower and upper bounds"""
    try:
        # Clip the value to the bounds
        clipped = max(lower, min(upper, value))
        
        # Normalize to 0-1 range
        normalized = (clipped - lower) / (upper - lower)
        
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing value: {e}")
        return 0.5  # Return middle value on error

def get_eye_state(ear):
    """Determine eye state based on eye aspect ratio with confidence score"""
    try:
        # Validate input
        if not isinstance(ear, (int, float)) or not np.isfinite(ear):
            logger.warning(f"Invalid EAR value: {ear}")
            return "UNKNOWN", 0.5

        # Normalize EAR value
        normalized_ear = normalize_value(ear, _ear_lower, _ear_upper)
        
        # Invert so that 0 = eyes open, 1 = eyes closed
        normalized_ear = 1.0 - normalized_ear
        
        # Calculate confidence based on distance from thresholds
        if normalized_ear < 0.2:
            confidence = 1.0 - (normalized_ear / 0.2)  # Higher confidence when closer to 0
            return "OPEN", min(1.0, max(0.0, confidence))
        elif normalized_ear < 0.7:
            # Lower confidence in the middle range
            confidence = 0.5
            return "PARTIALLY_CLOSED", confidence
        else:
            confidence = (normalized_ear - 0.7) / 0.3  # Higher confidence when closer to 1
            return "CLOSED", min(1.0, max(0.0, confidence))
    except Exception as e:
        logger.error(f"Error determining eye state: {e}")
        return "UNKNOWN", 0.5

def get_mouth_state(mar):
    """Determine mouth state based on mouth aspect ratio with confidence score"""
    try:
        # Validate input
        if not isinstance(mar, (int, float)) or not np.isfinite(mar):
            logger.warning(f"Invalid MAR value: {mar}")
            return "UNKNOWN", 0.5

        # Normalize MAR value
        normalized_mar = normalize_value(mar, _mar_lower, _mar_upper)
        
        # Calculate confidence based on distance from thresholds
        if normalized_mar < 0.3:
            confidence = 1.0 - (normalized_mar / 0.3)  # Higher confidence when closer to 0
            return "CLOSED", min(1.0, max(0.0, confidence))
        elif normalized_mar < 0.7:
            # Lower confidence in the middle range
            confidence = 0.5
            return "PARTIALLY_OPEN", confidence
        else:
            confidence = (normalized_mar - 0.7) / 0.3  # Higher confidence when closer to 1
            return "WIDE OPEN", min(1.0, max(0.0, confidence))
    except Exception as e:
        logger.error(f"Error determining mouth state: {e}")
        return "UNKNOWN", 0.5

def get_facial_state(right_ear, left_ear, mar, use_smoothing=True):
    """Get the facial state including normalized EAR and MAR values with optional smoothing and confidence scores"""
    try:
        # Validate inputs
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [right_ear, left_ear, mar]):
            logger.warning("Invalid input values for facial state")
            return {
                "right_ear": 0.25,
                "left_ear": 0.25,
                "avg_ear": 0.25,
                "mar": 0.5,
                "normalized_ear": 0.5,
                "normalized_mar": 0.5,
                "ear_confidence": 0.5,
                "mar_confidence": 0.5
            }

        # Calculate average EAR
        avg_ear = (right_ear + left_ear) / 2.0
        
        # Apply temporal smoothing if enabled
        if use_smoothing:
            try:
                # Add current values to buffers
                _ear_buffer.append(avg_ear)
                _mar_buffer.append(mar)
                
                # Calculate smoothed values
                smoothed_ear = weighted_temporal_smoothing(avg_ear, _ear_buffer)
                smoothed_mar = weighted_temporal_smoothing(mar, _mar_buffer)
                
                # Validate smoothed values
                if not all(np.isfinite([smoothed_ear, smoothed_mar])):
                    logger.warning("Invalid smoothed values calculated")
                    return {
                        "right_ear": right_ear,
                        "left_ear": left_ear,
                        "avg_ear": avg_ear,
                        "mar": mar,
                        "normalized_ear": 0.5,
                        "normalized_mar": 0.5,
                        "ear_confidence": 0.5,
                        "mar_confidence": 0.5
                    }
                
                # Use smoothed values
                avg_ear = smoothed_ear
                mar = smoothed_mar
            except Exception as e:
                logger.error(f"Error in temporal smoothing: {e}")
        
        # Normalize EAR and MAR
        normalized_ear = normalize_value(avg_ear, _ear_lower, _ear_upper)
        normalized_mar = normalize_value(mar, _mar_lower, _mar_upper)
        
        # Invert EAR so that 0 = eyes open, 1 = eyes closed
        normalized_ear = 1.0 - normalized_ear
        
        # Calculate confidence scores
        ear_confidence = 1.0 - abs(normalized_ear - 0.5) * 2  # Higher confidence when closer to 0 or 1
        mar_confidence = 1.0 - abs(normalized_mar - 0.5) * 2  # Higher confidence when closer to 0 or 1
        
        return {
            "right_ear": right_ear,
            "left_ear": left_ear,
            "avg_ear": avg_ear,
            "mar": mar,
            "normalized_ear": normalized_ear,
            "normalized_mar": normalized_mar,
            "ear_confidence": ear_confidence,
            "mar_confidence": mar_confidence
        }
    except Exception as e:
        logger.error(f"Error getting facial state: {e}")
        return {
            "right_ear": right_ear,
            "left_ear": left_ear,
            "avg_ear": 0.25,
            "mar": 0.5,
            "normalized_ear": 0.5,
            "normalized_mar": 0.5,
            "ear_confidence": 0.5,
            "mar_confidence": 0.5
        }
