import cv2
import numpy as np
import logging
from deepface import DeepFace
import os

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, config):
        self.debug_path = config.get('Paths', 'debug_aligned_faces_path')
        self.debug_enabled = config.getboolean('Logging', 'debug_save_faces', fallback=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

    def preprocess(self, face_image_bgr, tracker_id, frame_num):
        """Applies CLAHE and saves debug images."""
        
        # 1. Convert to grayscale for CLAHE
        face_gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE
        clahe_face = self.clahe.apply(face_gray)
        
        # 3. Convert back to BGR for ArcFace (it expects 3 channels)
        preprocessed_face = cv2.cvtColor(clahe_face, cv2.COLOR_GRAY2BGR)

        # 4. Save to debug folder
        if self.debug_enabled:
            filename = os.path.join(self.debug_path, f"track_{tracker_id}_frame_{frame_num}.jpg")
            cv2.imwrite(filename, preprocessed_face)
            
        return preprocessed_face

    def get_embedding(self, preprocessed_face):
        """Gets L2-normalized ArcFace embedding."""
        try:
            # DeepFace's 'represent' function handles resizing, normalization,
            # and returns a L2-normalized embedding by default for ArcFace.
            embedding_obj = DeepFace.represent(
                preprocessed_face,
                model_name=self.arcface_model,
                enforce_detection=False, # We've already aligned
                detector_backend='skip'
            )
            
            # embedding_obj is a list of dicts, we want the first
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