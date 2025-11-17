import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceValidator:
    """
    Geometric validator for frontal face detection.
    Filters based on brightness, sharpness, and contrast only.
    Removed edge density, symmetry, and centering.
    """

    @staticmethod
    @staticmethod
    def validate_brightness_distribution(aligned_face_bgr):
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        mean_brightness = np.mean(gray)
        # allow slightly darker overall scenes
        if mean_brightness < 50:   # relaxed from 60
            return False, f"Too dark (brightness={mean_brightness:.0f})"

        top_third = gray[:h//3, :]
        middle_third = gray[h//3:2*h//3, :]
        bottom_third = gray[2*h//3:, :]

        top_mean = np.mean(top_third)
        middle_mean = np.mean(middle_third)
        bottom_mean = np.mean(bottom_third)

        if middle_mean < 8 or bottom_mean < 8:
            return False, "Extremely dark regions"

        top_to_middle = top_mean / (middle_mean + 1e-6)
        middle_to_bottom = middle_mean / (bottom_mean + 1e-6)

        # Tolerances (relaxed a bit for webcams)
        if top_to_middle < 0.65:   # from 0.7 -> 0.65
            return False, f"Dark top (ratio={top_to_middle:.2f})"

        # Allow slightly higher middle_to_bottom when image is overall bright (selfie glare)
        relaxed_cutoff = 1.75
        if middle_to_bottom > relaxed_cutoff:
            # If image is very bright overall, permit a bit more ratio
            if mean_brightness > 110 and middle_to_bottom <= 1.95:
                return True, "Brightness OK (relaxed high mean)"
            return False, f"Bright middle (ratio={middle_to_bottom:.2f})"

        if middle_to_bottom < 0.60:  # from 0.65 -> 0.60
            return False, f"Dark middle (ratio={middle_to_bottom:.2f})"

        return True, "Brightness OK"


    @staticmethod
    def validate_sharpness(aligned_face_bgr, min_sharpness=50):
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        sharpness = np.mean(magnitude)
        if sharpness < min_sharpness:
            return False, f"Low sharpness ({sharpness:.0f})"
        return True, f"Sharp ({sharpness:.0f})"

    @staticmethod
    def validate_contrast(aligned_face_bgr, min_contrast=30):
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        if contrast < min_contrast:
            return False, f"Low contrast ({contrast:.0f})"
        return True, f"Contrast OK"

    @staticmethod
    def comprehensive_validation(aligned_face_bgr):
        h, w = aligned_face_bgr.shape[:2]
        if h < 80 or w < 80:
            return False, f"Too small ({w}x{h})"

        image_aspect_ratio = h / w
        if image_aspect_ratio < 1.15:
            return False, f"Too wide (AR={image_aspect_ratio:.2f})"
        if image_aspect_ratio > 1.85:
            return False, f"Too tall (AR={image_aspect_ratio:.2f})"

        brightness_ok, brightness_msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
        if not brightness_ok:
            return False, brightness_msg

        sharpness_ok, sharpness_msg = FaceValidator.validate_sharpness(aligned_face_bgr)
        if not sharpness_ok:
            return False, sharpness_msg

        contrast_ok, contrast_msg = FaceValidator.validate_contrast(aligned_face_bgr)
        if not contrast_ok:
            return False, contrast_msg

        return True, f"PASS (AR={image_aspect_ratio:.2f})"
