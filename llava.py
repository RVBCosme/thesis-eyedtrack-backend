import logging
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import threading

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# -----------------------------------------------------------------------------
# Constants and defaults
# -----------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_BEHAVIORS: List[str] = [
    "alert and attentive driver",
    "drowsy driver with eyes closing",
    "distracted driver looking away from road",
    "yawning driver showing fatigue",
    "driver using phone while driving",
    "driver talking to passenger",
    "normal driving behavior",
    "driver showing signs of fatigue"
]
DEFAULT_RISK_KEYWORDS: List[str] = ["drowsy", "distracted", "fatigue", "phone", "yawning"]
HIGH_RISK_THRESHOLD = 0.7

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Behavior:
    name: str
    confidence: float

@dataclass(frozen=True)
class BehaviorAnalysisResult:
    all_scores: Dict[str, float]
    top_behaviors: List[Behavior]

# -----------------------------------------------------------------------------
# Model loading with caching
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_clip_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    offline_mode: bool = False
) -> Tuple[CLIPProcessor, CLIPModel, torch.device]:
    """
    Load and cache the CLIP processor and model.
    
    Args:
        model_name: Name of the CLIP model to load
        device: Device to load the model on (cuda or cpu)
        offline_mode: If True, will attempt to load from local cache only
    """
    logger.info(f"Loading CLIP model '{model_name}'...")
    try:
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=offline_mode)
        model = CLIPModel.from_pretrained(model_name, local_files_only=offline_mode)
        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(target_device)
        logger.info(f"Model loaded on {target_device}.")
        return processor, model, target_device
    except Exception as e:
        if not offline_mode:
            logger.warning(f"Failed to load model online: {str(e)}. Trying offline mode...")
            return load_clip_model(model_name, device, True)
        else:
            logger.error(f"Failed to load model in offline mode: {str(e)}")
            raise

# -----------------------------------------------------------------------------
# Preprocessing utility
# -----------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    max_dim: int = 512
) -> np.ndarray:
    """
    Resize the frame to fit within max_dim and convert to RGB.
    Raises ValueError if input is invalid.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty or invalid frame provided.")
    
    # Add more robust error checking
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(frame)}")
    
    if frame.ndim < 2 or frame.ndim > 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {frame.shape}")
    
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        # Use a try-except block for resize operation
        try:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            # Return original if resize fails
            pass
            
    # Handle color conversion with error checking
    try:
        if frame.ndim == 2 or frame.shape[2] == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception as e:
        logger.error(f"Color conversion failed: {e}")
        # Last resort fallback - create a blank RGB image
        if frame.ndim == 2:
            return np.stack([frame, frame, frame], axis=2)
        return frame

# -----------------------------------------------------------------------------
# Core analyzer class
# -----------------------------------------------------------------------------
class BehaviorAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None
    ):
        """
        Initialize the behavior analyzer by loading the CLIP model.
        """
        self.processor, self.model, self.device = load_clip_model(model_name, device)

    def analyze(
        self,
        frame: np.ndarray,
        behavior_categories: Optional[List[str]] = None,
        top_k: int = 3,
        threshold: float = 0.0,
        measure_performance: bool = False
    ) -> Optional[BehaviorAnalysisResult]:
        """
        Analyze driver behavior from a single frame.
        Returns BehaviorAnalysisResult or None if analysis fails.
        
        Args:
            frame: Input image frame
            behavior_categories: List of behavior categories to classify
            top_k: Number of top behaviors to return
            threshold: Minimum confidence threshold for behaviors
            measure_performance: If True, includes performance metrics in result
        """
        import time
        
        try:
            start_time = time.time()
            preprocess_start = start_time
            
            image = preprocess_frame(frame)
            preprocess_time = time.time() - preprocess_start
            
            categories = behavior_categories or DEFAULT_BEHAVIORS
            
            # Prepare inputs for CLIP
            inference_start = time.time()
            inputs = self.processor(
                text=categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image  # shape: [1, num_categories]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            inference_time = time.time() - inference_start
            
            # Construct full score mapping
            all_scores = {cat: float(score) for cat, score in zip(categories, probs)}
            
            # Select top behaviors above threshold
            sorted_indices = np.argsort(probs)[::-1]
            top_behaviors: List[Behavior] = []
            for idx in sorted_indices:
                score = float(probs[idx])
                if score < threshold:
                    break
                top_behaviors.append(Behavior(name=categories[idx], confidence=score))
                if len(top_behaviors) >= top_k:
                    break
            
            total_time = time.time() - start_time
            
            result = BehaviorAnalysisResult(all_scores=all_scores, top_behaviors=top_behaviors)
            
            if measure_performance:
                # Add performance metrics to result
                result.performance_metrics = {
                    "total_time_ms": total_time * 1000,
                    "preprocess_time_ms": preprocess_time * 1000,
                    "inference_time_ms": inference_time * 1000
                }
                
            return result
            
        except Exception as e:
            logger.exception(f"Failed to analyze behavior: {str(e)}")
            return None

    def summarize(
        self,
        result: Optional[BehaviorAnalysisResult],
        risk_keywords: Optional[List[str]] = None,
        high_risk_threshold: float = HIGH_RISK_THRESHOLD
    ) -> str:
        """
        Generate a human-readable summary indicating risk level.
        """
        if not result or not result.top_behaviors:
            return "Behavior analysis unavailable"
        risk_keywords = risk_keywords or DEFAULT_RISK_KEYWORDS
        top = result.top_behaviors[0]
        behavior_lower = top.name.lower()
        is_risky = any(kw in behavior_lower for kw in risk_keywords)
        conf_pct = top.confidence * 100
        if is_risky:
            level = (
                "HIGH RISK" if top.confidence > high_risk_threshold else "MODERATE RISK"
            )
            return f"{level}: {top.name} ({conf_pct:.1f}% confidence)"
        return f"SAFE: {top.name} ({conf_pct:.1f}% confidence)"

    def analyze_batch(
        self,
        frames: List[np.ndarray],
        behavior_categories: Optional[List[str]] = None,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Optional[BehaviorAnalysisResult]]:
        """
        Analyze driver behavior from multiple frames.
        Returns a list of BehaviorAnalysisResult or None for each frame.
        """
        results = []
        for frame in frames:
            try:
                result = self.analyze(frame, behavior_categories, top_k, threshold)
                results.append(result)
            except Exception as e:
                logger.exception(f"Failed to analyze frame in batch: {str(e)}")
                results.append(None)
        return results

# -----------------------------------------------------------------------------
# Module API
# -----------------------------------------------------------------------------
__all__ = [
    "BehaviorAnalyzer",
    "BehaviorAnalysisResult",
    "Behavior",
    "AnalyzerConfig",
    "load_clip_model",
    "preprocess_frame"
]

# Add a singleton pattern similar to mobilenet.py for thread safety
_analyzer: Optional[BehaviorAnalyzer] = None
_lock = threading.Lock()

def get_behavior_analyzer(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None
) -> BehaviorAnalyzer:
    """
    Get or create a singleton BehaviorAnalyzer instance.
    Thread-safe implementation.
    """
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = BehaviorAnalyzer(model_name, device)
    return _analyzer

# -----------------------------------------------------------------------------
# Compatibility functions for batch_processor.py
# -----------------------------------------------------------------------------
def analyze_behavior(
    frame: np.ndarray,
    behavior_categories: Optional[List[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    """
    Compatibility function for analyzing driver behavior.
    
    Returns:
        Tuple of (all_scores, top_behaviors) where top_behaviors is a list of (name, confidence) tuples
    """
    analyzer = BehaviorAnalyzer(model_name=model_name, device=device)
    result = analyzer.analyze(frame, behavior_categories)
    
    if not result:
        return {}, []
        
    # Convert to old format
    top_behaviors = [(b.name, b.confidence) for b in result.top_behaviors]
    return result.all_scores, top_behaviors

def get_behavior_summary(
    behavior_name: str,
    confidence: float,
    risk_keywords: Optional[List[str]] = None,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD
) -> str:
    """
    Compatibility function for generating behavior summary.
    """
    risk_keywords = risk_keywords or DEFAULT_RISK_KEYWORDS
    behavior_lower = behavior_name.lower()
    is_risky = any(kw in behavior_lower for kw in risk_keywords)
    conf_pct = confidence * 100
    
    if is_risky:
        level = "HIGH RISK" if confidence > high_risk_threshold else "MODERATE RISK"
        return f"{level}: {behavior_name} ({conf_pct:.1f}% confidence)"
    return f"SAFE: {behavior_name} ({conf_pct:.1f}% confidence)"

# Update module API to include new functions
__all__ = [
    "BehaviorAnalyzer",
    "BehaviorAnalysisResult",
    "Behavior",
    "load_clip_model",
    "preprocess_frame",
    "analyze_behavior",
    "get_behavior_summary"
]

def analyze_behavior(
    frame: np.ndarray,
    behavior_categories: Optional[List[str]] = None,
    top_k: int = 3,
    threshold: float = 0.0
) -> Optional[BehaviorAnalysisResult]:
    """
    Convenience function for behavior analysis using the singleton analyzer.
    """
    analyzer = get_behavior_analyzer()
    return analyzer.analyze(frame, behavior_categories, top_k, threshold)

def get_behavior_summary(
    result: Optional[BehaviorAnalysisResult],
    risk_keywords: Optional[List[str]] = None,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD
) -> str:
    """
    Convenience function for summarizing behavior using the singleton analyzer.
    """
    analyzer = get_behavior_analyzer()
    return analyzer.summarize(result, risk_keywords, high_risk_threshold)

# Add a cache for behavior analysis results to avoid redundant processing
from functools import lru_cache

@lru_cache(maxsize=32)
def _cached_behavior_analysis(
    frame_hash: str,
    model_name: str,
    categories_hash: str,
    top_k: int,
    threshold: float
) -> Optional[Dict]:
    """
    Internal cache function for behavior analysis results.
    Uses string hashes as keys since numpy arrays aren't hashable.
    """
    # This is just a placeholder for the cache key
    # The actual implementation would need to reconstruct the frame and categories
    pass

def get_frame_hash(frame: np.ndarray) -> str:
    """Generate a hash for a frame to use as cache key."""
    # Downsample for faster hashing
    small = cv2.resize(frame, (32, 32))
    # Convert to grayscale and flatten
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Create hash
    return hashlib.md5(small.tobytes()).hexdigest()

def get_categories_hash(categories: List[str]) -> str:
    """Generate a hash for behavior categories."""
    return hashlib.md5(str(categories).encode()).hexdigest()

'''
llava.py - Vision-Language Driver Behavior Analysis Module

This module provides tools for analyzing driver behavior using CLIP models
to classify images based on textual behavior descriptions.

Key Features:
- Zero-shot classification of driver behaviors
- Risk assessment based on detected behaviors
- Efficient model loading and caching
- Thread-safe singleton implementation

def get_risk_level(
    self,
    result: Optional[BehaviorAnalysisResult],
    risk_keywords: Optional[List[str]] = None,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
    moderate_risk_threshold: float = 0.4
) -> Tuple[str, float]:
    """
    Calculate risk level from behavior analysis result.
    
    Returns:
        Tuple of (risk_level, risk_score) where risk_level is one of:
        "HIGH", "MODERATE", "LOW", or "UNKNOWN"
    """
    if not result or not result.top_behaviors:
        return "UNKNOWN", 0.0
        
    risk_keywords = risk_keywords or DEFAULT_RISK_KEYWORDS
    top = result.top_behaviors[0]
    behavior_lower = top.name.lower()
    
    # Calculate risk score based on keywords and confidence
    risk_score = 0.0
    for kw in risk_keywords:
        if kw in behavior_lower:
            risk_score = max(risk_score, top.confidence)
    
    # Determine risk level
    if risk_score > high_risk_threshold:
        return "HIGH", risk_score
    elif risk_score > moderate_risk_threshold:
        return "MODERATE", risk_score
    elif risk_score > 0:
        return "LOW", risk_score
    else:
        return "SAFE", 0.0
'''


@dataclass
class AnalyzerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    high_risk_threshold: float = HIGH_RISK_THRESHOLD
    moderate_risk_threshold: float = 0.4
    default_behaviors: List[str] = field(default_factory=lambda: DEFAULT_BEHAVIORS)
    risk_keywords: List[str] = field(default_factory=lambda: DEFAULT_RISK_KEYWORDS)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'AnalyzerConfig':
        """Load configuration from JSON file"""
        import json
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {json_path}: {str(e)}")
            return cls()
            
    def to_json(self, json_path: str) -> bool:
        """Save configuration to JSON file"""
        import json
        try:
            with open(json_path, 'w') as f:
                json.dump(self.__dict__, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {json_path}: {str(e)}")
            return False