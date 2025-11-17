import cv2
import numpy as np
import logging
from deepface import DeepFace

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, config):
        self.arcface_model = "ArcFace"

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

    # def calculate_blur(self, image):
    #     """Calculate sharpness using Sobel gradient magnitude (better for motion blur)."""
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    #     gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    #     magnitude = np.sqrt(gx**2 + gy**2)
    #     return np.mean(magnitude)

    def calculate_blur_laplacian(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_sobel_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        return np.mean(magnitude)


    def get_embedding(self, face_image_bgr):
        try:
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
        if prev_embedding is None:
            return current_embedding
        smoothed_embedding = (alpha * current_embedding) + ((1 - alpha) * prev_embedding)
        l2_norm = np.linalg.norm(smoothed_embedding)
        return smoothed_embedding / l2_norm
