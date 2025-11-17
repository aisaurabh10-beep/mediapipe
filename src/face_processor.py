import cv2
import numpy as np
import logging
from deepface import DeepFace

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, config):
        self.arcface_model = "ArcFace"
        
        # Warm up the model
        try:
            logger.info("Warming up ArcFace model...")
            DeepFace.represent(
                np.zeros((112, 112, 3), dtype=np.uint8),
                model_name=self.arcface_model,
                detector_backend='skip'
            )
            logger.info("ArcFace model ready.")
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            raise

    def calculate_blur(self, image):
        """Calculates blurriness using Laplacian variance."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def get_embedding(self, face_image_bgr):
        """
        Gets L2-normalized ArcFace embedding from RGB face.
        No preprocessing - just pass the aligned RGB face directly.
        """
        try:
            # DeepFace handles resizing and normalization internally
            embedding_obj = DeepFace.represent(
                face_image_bgr,
                model_name=self.arcface_model,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            embedding = embedding_obj[0]['embedding']
            return np.array(embedding)
        
        except Exception as e:
            logger.warning(f"Could not get embedding: {e}")
            return None

    def apply_ema(self, current_embedding, prev_embedding, alpha):
        """Applies Exponential Moving Average to the embedding."""
        if prev_embedding is None:
            return current_embedding
        
        smoothed_embedding = (alpha * current_embedding) + ((1 - alpha) * prev_embedding)
        
        # Re-normalize to L2
        l2_norm = np.linalg.norm(smoothed_embedding)
        return smoothed_embedding / l2_norm