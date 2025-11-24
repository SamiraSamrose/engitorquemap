#ml_models/inference/edge_inference.py
"""
Edge inference using ONNX Runtime
For deployment on edge devices
"""
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'backend'))

from config import settings

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available")


class EdgeInferenceEngine:
    """ONNX Runtime inference engine for edge deployment"""
    
    def __init__(self):
        self.session = None
        self.load_model()
    
    def load_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available")
            return
        
        model_path = Path(settings.ONNX_MODEL_PATH)
        
        if not model_path.exists():
            print(f"ONNX model not found at {model_path}")
            return
        
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=settings.ONNX_PROVIDERS
            )
            print(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
    
    def predict(self, features):
        """Run inference on features"""
        if self.session is None:
            raise RuntimeError("ONNX model not loaded")
        
        # Prepare input
        input_name = self.session.get_inputs()[0].name
        input_data = features.astype(np.float32)
        
        # Run inference
        result = self.session.run(None, {input_name: input_data})
        
        return result[0]