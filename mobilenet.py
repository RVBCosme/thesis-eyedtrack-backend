"""
Enhanced MobileNet-based behavior classification for driver monitoring with GPU acceleration
"""

import os
import threading
import logging
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

# Configure module-level logger with more detailed formatting
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Expanded mapping of ImageNet classes to driver behaviors
BEHAVIOR_MAPPING = {
    "sleeping_bag": "drowsy driver",
    "pillow": "drowsy driver",
    "sunglasses": "attentive driver with sunglasses",
    "cellular_telephone": "distracted driver using phone",
    "seat_belt": "driver wearing seatbelt",
    "minibus": "driver in vehicle",
    "cab": "driver in vehicle",
    "car_mirror": "driver in vehicle",
    "neck_brace": "driver with head support",
    "mask": "driver wearing mask",
    "microphone": "driver talking",
    "water_bottle": "driver drinking",
    "coffee_mug": "driver drinking",
    "remote_control": "driver distracted",
    "iPod": "driver distracted with device",
    "laptop": "driver severely distracted",
    "digital_watch": "driver checking time",
    "hand_blower": "driver adjusting climate control",
    "cigarette": "driver smoking",
    "book_jacket": "driver reading",
    "menu": "driver reading",
    "notebook": "driver writing",
    "sunscreen": "driver applying cosmetics"
}

# Risk levels for different behaviors
RISK_LEVELS = {
    "high": ["drowsy", "sleeping", "distracted", "phone", "reading", "writing", "laptop"],
    "moderate": ["drinking", "smoking", "talking", "device", "cosmetics"],
    "low": ["sunglasses", "seatbelt", "attentive", "time"]
}

class BehaviorClassifier:
    """
    Enhanced MobileNetV2-based behavior classification for driver monitoring.
    
    Features:
    - Dual model support: TensorFlow and OpenCV DNN (ONNX)
    - GPU acceleration on both frameworks when available
    - Loads a fine-tuned custom model if available (folder: driver_behavior_model)
    - Falls back to ImageNet-pretrained MobileNetV2
    - Provides classify() and summarize() methods
    - Supports temporal smoothing for more stable predictions
    - Includes performance metrics and caching for efficiency
    - Handles various input formats (BGR, RGB, grayscale)
    """
    def __init__(
        self, 
        model_dir: str = 'driver_behavior_model', 
        top_k: int = 5,
        temporal_smoothing: bool = True,
        smoothing_window: int = 5,
        cache_size: int = 128,
        use_opencv_dnn: bool = False
    ):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
        self.top_k = top_k
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.cache_size = cache_size
        self.use_opencv_dnn = use_opencv_dnn
        self._history = deque(maxlen=smoothing_window)
        self._metrics = {
            "inference_times": [],
            "avg_inference_time": 0.0,
            "total_inferences": 0,
            "cache_hits": 0,
            "backend": "unknown"
        }
        
        # Create a manual cache instead of using lru_cache
        self._result_cache = {}
        
        # Load the appropriate model
        self._load_model()
        
        # Warm-up inference to avoid first-call lag
        self._warmup_model()
        
        logger.info(f"Model initialization complete. Using backend: {self._metrics['backend']}")

    def _load_model(self) -> None:
        """Load the appropriate model, with GPU acceleration where available."""
        if hasattr(self, '_model'):
            return
            
        logger.info("Loading behavior classification model...")
        
        # Try OpenCV DNN (ONNX) backend if specified
        if self.use_opencv_dnn:
            self._load_opencv_model()
            if hasattr(self, '_model'):
                return
            logger.warning("OpenCV DNN model loading failed, falling back to TensorFlow")
        
        # Otherwise, use TensorFlow
        self._load_tensorflow_model()

    def _load_opencv_model(self) -> None:
        """Try to load model using OpenCV DNN backend with ONNX format."""
        try:
            # Check for ONNX model file
            onnx_path = os.path.join(self.model_dir, "model.onnx")
            labels_path = os.path.join(self.model_dir, "labels.txt")
            
            if not os.path.exists(onnx_path):
                logger.warning(f"ONNX model file not found: {onnx_path}")
                return
                
            # Load class labels if available
            self._labels = None
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    self._labels = [line.strip() for line in f]
                logger.info(f"Loaded {len(self._labels)} class labels")
            
            # Load the model with GPU acceleration if available
            self._model = cv2.dnn.readNetFromONNX(onnx_path)
            
            # Try to use CUDA backend if available
            try:
                self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self._metrics["backend"] = "opencv_dnn_cuda"
                logger.info("Using OpenCV DNN with CUDA acceleration")
            except Exception as e:
                logger.warning(f"CUDA acceleration not available for OpenCV DNN: {e}")
                self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._metrics["backend"] = "opencv_dnn_cpu"
                logger.info("Using OpenCV DNN with CPU")
            
            self._model_type = "custom_opencv" if self._labels else "imagenet_opencv"
            self._input_size = (224, 224)  # Standard size for MobileNet
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV DNN model: {e}")
            if hasattr(self, '_model'):
                delattr(self, '_model')

    def _load_tensorflow_model(self) -> None:
        """Load model using TensorFlow backend with GPU acceleration if available."""
        model = None
        labels = None
        
        # Check for GPU and optimize if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available for TensorFlow: {len(gpus)} device(s)")
                self._metrics["backend"] = "tensorflow_gpu"
            except RuntimeError as e:
                logger.warning(f"GPU memory configuration failed: {e}")
                self._metrics["backend"] = "tensorflow_cpu"
        else:
            self._metrics["backend"] = "tensorflow_cpu"
        
        # Try loading saved model (TensorFlow SavedModel format)
        if os.path.exists(self.model_dir):
            try:
                # Try loading with TF SavedModel format
                saved_model_path = os.path.join(self.model_dir, "saved_model")
                if os.path.exists(saved_model_path):
                    model = tf.keras.models.load_model(saved_model_path)
                else:
                    # Try loading as direct directory
                    model = tf.keras.models.load_model(self.model_dir)
                
                # Look for labels file
                labels_path = os.path.join(self.model_dir, 'labels.txt')
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        labels = [line.strip() for line in f]
                
                logger.info(f"Loaded custom TensorFlow model with {len(labels) if labels else 'unknown'} classes")
                
                # Try model optimization if supported
                if gpus and len(gpus) > 0:
                    try:
                        # Convert to mixed precision for faster GPU inference
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        logger.info("Mixed precision enabled for faster inference")
                    except Exception as e:
                        logger.warning(f"Model optimization failed: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to load custom TensorFlow model: {e}")
                model = None
                
        # Fallback to ImageNet pretrained MobileNetV2
        if model is None:
            try:
                model = tf.keras.applications.MobileNetV2(weights='imagenet')
                labels = None
                logger.info("Loaded pre-trained MobileNetV2 (ImageNet)")
            except Exception as e:
                logger.error(f"Failed to load MobileNetV2: {e}")
                raise RuntimeError("Failed to load any model") from e
                
        self._model = model
        self._labels = labels
        self._model_type = "custom_tf" if labels else "imagenet_tf"

    def _warmup_model(self) -> None:
        """Perform a warm-up inference to initialize model and avoid first-call lag."""
        try:
            # Create a dummy input for warm-up
            if hasattr(self, "_model_type") and "opencv" in self._model_type:
                # OpenCV DNN warm-up
                dummy = np.zeros((self._input_size[0], self._input_size[1], 3), dtype=np.uint8)
                _ = self._opencv_inference(dummy)
            else:
                # TensorFlow warm-up
                dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = self._model.predict(dummy, verbose=0)
                
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _preprocess_image_tf(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorFlow model input."""
        # Handle different color formats
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            # Grayscale to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # Assume BGR (OpenCV default) and convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Resize & normalize
        img = cv2.resize(rgb, (224, 224))
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return np.expand_dims(img, 0)

    def _preprocess_image_opencv(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for OpenCV DNN model input."""
        # Convert to RGB if needed
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb, self._input_size)
        
        # Convert to blob (batch, channels, height, width)
        blob = cv2.dnn.blobFromImage(
            resized, 
            1.0/255.0,  # Scale factor
            self._input_size, 
            (0, 0, 0),  # Mean subtraction 
            swapRB=False,  # Already converted to RGB
            crop=False
        )
        
        return blob

    def _opencv_inference(self, frame: np.ndarray) -> np.ndarray:
        """Run inference using OpenCV DNN backend."""
        blob = self._preprocess_image_opencv(frame)
        self._model.setInput(blob)
        return self._model.forward()

    def _classify_impl(
        self, 
        frame: np.ndarray, 
        confidence_threshold: float = 0.3,
        frame_id: Optional[str] = None  # For caching purposes
    ) -> Optional[List[Tuple[Any, str, float, bool]]]:
        """
        Implementation of classification logic.
        Returns a list of (id, behavior, confidence, is_risky) tuples.
        """
        if frame is None or frame.size == 0:
            return None
            
        start_time = time.time()
        
        # Run inference based on backend
        if hasattr(self, "_model_type") and "opencv" in self._model_type:
            # OpenCV DNN inference
            preds = self._opencv_inference(frame)[0]
        else:
            # TensorFlow inference
            batch = self._preprocess_image_tf(frame)
            preds = self._model.predict(batch, verbose=0)[0]
        
        # Process results based on model type
        results: List[Tuple[Any, str, float, bool]] = []
        
        if self._labels:  # Custom model
            idxs = np.argsort(preds)[::-1][:self.top_k]
            for i in idxs:
                conf = float(preds[i])
                if conf < confidence_threshold:
                    continue
                name = self._labels[i]
                risk_level = self._determine_risk_level(name.lower())
                is_risky = risk_level in ["high", "moderate"]
                results.append((i, name, conf, is_risky))
        else:  # ImageNet model
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(np.array([preds]))[0]
            for imagenet_id, class_name, conf in decoded:
                if conf < confidence_threshold:
                    continue
                behavior = BEHAVIOR_MAPPING.get(class_name, f"driver with {class_name}")
                risk_level = self._determine_risk_level(behavior.lower())
                is_risky = risk_level in ["high", "moderate"]
                results.append((imagenet_id, behavior, float(conf), is_risky))
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self._metrics["inference_times"].append(inference_time)
        self._metrics["total_inferences"] += 1
        if len(self._metrics["inference_times"]) > 100:
            self._metrics["inference_times"].pop(0)
        self._metrics["avg_inference_time"] = sum(self._metrics["inference_times"]) / len(self._metrics["inference_times"])
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and results:
            self._history.append(results)
            results = self._smooth_predictions()
            
        return results

    def _determine_risk_level(self, behavior_text: str) -> str:
        """Determine risk level based on behavior text."""
        for level, keywords in RISK_LEVELS.items():
            if any(keyword in behavior_text for keyword in keywords):
                return level
        return "low"  # Default to low risk

    def _smooth_predictions(self) -> List[Tuple[Any, str, float, bool]]:
        """Apply temporal smoothing to predictions for more stable results."""
        if not self._history:
            return []
            
        # Count occurrences of each behavior
        behavior_counts: Dict[str, Dict[str, Union[int, float, bool, Any]]] = {}
        
        for frame_results in self._history:
            for id_val, behavior, conf, is_risky in frame_results:
                if behavior not in behavior_counts:
                    behavior_counts[behavior] = {
                        "count": 0, 
                        "total_conf": 0.0, 
                        "id": id_val,
                        "is_risky": is_risky
                    }
                behavior_counts[behavior]["count"] += 1
                behavior_counts[behavior]["total_conf"] += conf
        
        # Sort by count, then by average confidence
        sorted_behaviors = sorted(
            behavior_counts.items(),
            key=lambda x: (x[1]["count"], x[1]["total_conf"] / x[1]["count"]),
            reverse=True
        )
        
        # Return smoothed results
        smoothed_results = []
        for behavior, data in sorted_behaviors:
            avg_conf = data["total_conf"] / data["count"]
            smoothed_results.append((
                data["id"],
                behavior,
                avg_conf,
                data["is_risky"]
            ))
            
        return smoothed_results[:self.top_k]

    def _manage_cache_size(self):
        """Limit the cache size to self.cache_size by removing oldest entries."""
        if len(self._result_cache) > self.cache_size:
            # Remove oldest entries (first 10% of cache)
            items_to_remove = int(self.cache_size * 0.1)
            for _ in range(items_to_remove):
                if self._result_cache:
                    self._result_cache.pop(next(iter(self._result_cache)))

    def classify(
        self, 
        frame: np.ndarray, 
        confidence_threshold: float = 0.3,
        frame_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Public classify method that handles caching using frame_id.
        Returns a dictionary with classification results.
        """
        if frame is None or frame.size == 0:
            return None
            
        # Generate frame_id if not provided, or ensure it's used correctly
        if frame_id is None:
            frame_id = get_frame_hash(frame)
        
        # Check if result is in cache
        cache_key = (frame_id, confidence_threshold)
        if cache_key in self._result_cache:
            self._metrics["cache_hits"] += 1
            return self._result_cache[cache_key]
        
        # Not in cache, perform the classification
        results = self._classify_impl(frame, confidence_threshold, frame_id)
        
        if not results:
            return None
            
        # Convert results to dictionary format
        top_result = results[0]  # Get the highest confidence result
        _, behavior, conf, is_risky = top_result
        
        # Create dictionary result
        dict_result = {
            "top_class": behavior,
            "top_score": float(conf),
            "is_risky": is_risky,
            "all_results": [
                {
                    "class": beh,
                    "score": float(score),
                    "is_risky": risky
                }
                for _, beh, score, risky in results
            ]
        }
        
        # Store in cache
        self._result_cache[cache_key] = dict_result
        
        # Manage cache size
        self._manage_cache_size()
        
        return dict_result

    def summarize(
        self, 
        results: Optional[List[Tuple[Any, str, float, bool]]]
    ) -> Dict[str, Any]:
        """
        Returns an enhanced summary dict with keys: 
        behavior, confidence, is_risky, risk_level, description, metrics.
        """
        if not results:
            return {
                "behavior": "Unknown",
                "confidence": 0.0,
                "is_risky": False,
                "risk_level": "none",
                "description": "Behavior classification unavailable",
                "metrics": self._get_metrics()
            }
            
        _, behavior, conf, is_risky = results[0]
        
        # Determine risk level
        risk_level = self._determine_risk_level(behavior.lower())
        
        # Create description
        if risk_level == "high":
            desc = f"HIGH RISK: {behavior} detected"
        elif risk_level == "moderate":
            desc = f"MODERATE RISK: {behavior} detected"
        else:
            desc = f"SAFE: {behavior} detected"
            
        # Include top 3 behaviors if available
        top_behaviors = []
        for _, beh, conf, _ in results[:3]:
            top_behaviors.append({"behavior": beh, "confidence": conf})
            
        return {
            "behavior": behavior,
            "confidence": conf,
            "is_risky": is_risky,
            "risk_level": risk_level,
            "description": desc,
            "top_behaviors": top_behaviors,
            "metrics": self._get_metrics()
        }
        
    def _get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            "avg_inference_time": self._metrics["avg_inference_time"],
            "total_inferences": self._metrics["total_inferences"],
            "cache_hits": self._metrics["cache_hits"],
            "backend": self._metrics["backend"],
            "model_type": getattr(self, "_model_type", "unknown")
        }
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "model_type": getattr(self, "_model_type", "unknown"),
            "custom_model": self._labels is not None,
            "num_classes": len(self._labels) if self._labels else 1000,
            "model_path": self.model_dir,
            "backend": self._metrics["backend"],
            "using_gpu": "gpu" in self._metrics["backend"].lower(),
            "metrics": self._get_metrics()
        }

# Module-level singleton and lock for thread safety
_classifier: Optional[BehaviorClassifier] = None
_lock = threading.Lock()

def get_frame_hash(frame: np.ndarray) -> str:
    """Generate a hash for a frame to use as cache key."""
    # Downsample for faster hashing
    small = cv2.resize(frame, (32, 32))
    # Convert to grayscale and flatten
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Create hash
    return hashlib.md5(small.tobytes()).hexdigest()

def classify_behavior(
    frame: np.ndarray, 
    confidence_threshold: float = 0.3,
    frame_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function matching original API, delegating to the singleton classifier.
    Added frame_id parameter for caching support.
    """
    global _classifier
    with _lock:
        if _classifier is None:
            _classifier = BehaviorClassifier()
    
    # Always generate a frame_id if not provided
    if frame_id is None and frame is not None:
        frame_id = get_frame_hash(frame)
        
    return _classifier.classify(frame, confidence_threshold, frame_id)

def get_behavior_summary(
    results: Optional[List[Tuple[Any, str, float, bool]]]
) -> Dict[str, Any]:
    """
    Convenience function matching original API, delegating to the singleton classifier.
    Returns enhanced summary with metrics.
    """
    global _classifier
    with _lock:
        if _classifier is None:
            _classifier = BehaviorClassifier()
    return _classifier.summarize(results)

def get_model_info() -> Dict[str, Any]:
    """
    Convenience function to get information about the loaded model.
    """
    global _classifier
    with _lock:
        if _classifier is None:
            _classifier = BehaviorClassifier()
    return _classifier.get_model_info()

def create_classifier(
    model_dir: str = 'driver_behavior_model',
    use_opencv: bool = False
) -> BehaviorClassifier:
    """
    Create a new classifier instance with specific settings.
    Useful when you need multiple classifiers with different configurations.
    
    Args:
        model_dir: Path to model directory
        use_opencv: Whether to use OpenCV DNN backend (if ONNX model available)
        
    Returns:
        A new BehaviorClassifier instance
    """
    return BehaviorClassifier(
        model_dir=model_dir,
        use_opencv_dnn=use_opencv
    )