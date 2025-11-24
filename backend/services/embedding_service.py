## **File: backend/services/embedding_service.py** (Vector Embeddings)

"""
Embedding Service
Generates vector embeddings for driver telemetry, track data, and queries
Used for similarity search and RAG
"""
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Embeddings limited.")

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings for semantic search and RAG
    """
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.model or not TRANSFORMERS_AVAILABLE:
            # Return random embedding as fallback
            return np.random.randn(384)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.random.randn(384)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if not self.model or not TRANSFORMERS_AVAILABLE:
            return [np.random.randn(384) for _ in texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return list(embeddings)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [np.random.randn(384) for _ in texts]
    
    def generate_telemetry_embedding(self, telemetry: Dict) -> np.ndarray:
        """
        Generate embedding for telemetry data
        Converts telemetry dict to text description then embeds
        """
        # Convert telemetry to text description
        text = self._telemetry_to_text(telemetry)
        return self.generate_embedding(text)
    
    def _telemetry_to_text(self, telemetry: Dict) -> str:
        """Convert telemetry dictionary to text description"""
        parts = []
        
        if 'speed' in telemetry:
            parts.append(f"speed {telemetry['speed']:.1f} m/s")
        
        if 'gear' in telemetry:
            parts.append(f"gear {telemetry['gear']}")
        
        if 'pbrake_f' in telemetry:
            parts.append(f"brake pressure {telemetry['pbrake_f']:.1f} bar")
        
        if 'ath' in telemetry:
            parts.append(f"throttle {telemetry['ath']:.1f}%")
        
        if 'accx_can' in telemetry:
            parts.append(f"longitudinal g {telemetry['accx_can']:.2f}")
        
        if 'accy_can' in telemetry:
            parts.append(f"lateral g {telemetry['accy_can']:.2f}")
        
        return ", ".join(parts)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar(self, query_embedding: np.ndarray, 
                    candidate_embeddings: List[np.ndarray], 
                    top_k: int = 5) -> List[int]:
        """
        Find most similar embeddings
        Returns indices of top-k most similar candidates
        """
        similarities = [
            self.compute_similarity(query_embedding, candidate)
            for candidate in candidate_embeddings
        ]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return list(top_indices)
