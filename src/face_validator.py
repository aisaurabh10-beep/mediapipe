# src/face_validator.py
import cv2
import numpy as np
import logging
from src.config_loader import load_config

logger = logging.getLogger(__name__)

# Cached config object so we don't reload repeatedly
_CONFIG = None

def _get_config():
    global _CONFIG
    if _CONFIG is None:
        try:
            _CONFIG = load_config('config.ini')
        except Exception as e:
            logger.debug(f"Failed to load config.ini in FaceValidator: {e}")
            _CONFIG = None
    return _CONFIG

class FaceValidator:
    """
    Geometric/photometric validator for aligned faces.
    All thresholds are read from config.ini under [Quality_Filters].
    Returns (is_valid: bool, reason: str).
    """

    @staticmethod
    def _get_th(name, default, cast=float):
        cfg = _get_config()
        if cfg is None:
            return default
        try:
            if cast is int:
                return cfg.getint('Quality_Filters', name, fallback=default)
            if cast is float:
                return cfg.getfloat('Quality_Filters', name, fallback=default)
            if cast is bool:
                return cfg.getboolean('Quality_Filters', name, fallback=default)
            return cfg.get('Quality_Filters', name, fallback=default)
        except Exception:
            return default

    @staticmethod
    def validate_brightness_distribution(aligned_face_bgr):
        """
        Uses configurable thresholds:
          - min_brightness
          - top_middle_min_ratio
          - middle_bottom_min_ratio (lower bound)
          - middle_bottom_max_ratio (upper bound)
          - middle_bottom_relaxed_mean (if mean brightness > this, allow relaxed upper)
          - middle_bottom_relaxed_max
        """
        cfg = _get_config()
        min_brightness = FaceValidator._get_th('min_brightness', 50.0, float)
        top_middle_min_ratio = FaceValidator._get_th('top_middle_min_ratio', 0.45, float)
        middle_bottom_min_ratio = FaceValidator._get_th('middle_bottom_min_ratio', 0.60, float)
        middle_bottom_max_ratio = FaceValidator._get_th('middle_bottom_max_ratio', 1.75, float)
        middle_bottom_relaxed_mean = FaceValidator._get_th('middle_bottom_relaxed_mean', 110.0, float)
        middle_bottom_relaxed_max = FaceValidator._get_th('middle_bottom_relaxed_max', 1.95, float)

        try:
            gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = aligned_face_bgr.copy()

        h, w = gray.shape
        mean_brightness = float(np.mean(gray))

        if mean_brightness < min_brightness:
            return False, f"Too dark (brightness={mean_brightness:.0f} < {min_brightness:.0f})"

        top_third = gray[:h//3, :]
        middle_third = gray[h//3:2*h//3, :]
        bottom_third = gray[2*h//3:, :]

        top_mean = float(np.mean(top_third))
        middle_mean = float(np.mean(middle_third))
        bottom_mean = float(np.mean(bottom_third))

        if middle_mean < 8 or bottom_mean < 8:
            return False, "Extremely dark regions"

        top_to_middle = top_mean / (middle_mean + 1e-6)
        middle_to_bottom = middle_mean / (bottom_mean + 1e-6)

        if top_to_middle < top_middle_min_ratio:
            return False, f"Dark top (ratio={top_to_middle:.2f} < {top_middle_min_ratio:.2f})"

        if middle_to_bottom > middle_bottom_max_ratio:
            # allow relaxed max when overall bright
            if mean_brightness > middle_bottom_relaxed_mean and middle_to_bottom <= middle_bottom_relaxed_max:
                return True, "Brightness OK (relaxed high mean)"
            return False, f"Bright middle (ratio={middle_to_bottom:.2f} > {middle_bottom_max_ratio:.2f})"

        if middle_to_bottom < middle_bottom_min_ratio:
            return False, f"Dark middle (ratio={middle_to_bottom:.2f} < {middle_bottom_min_ratio:.2f})"

        return True, "Brightness OK"

    @staticmethod
    def validate_sharpness(aligned_face_bgr):
        """
        Sharpness via Sobel magnitude. Config keys:
          - min_sharpness
        """
        min_sharpness = FaceValidator._get_th('min_sharpness', 50.0, float)
        try:
            gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = aligned_face_bgr.copy()
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        sharpness = float(np.mean(magnitude))
        if sharpness < min_sharpness:
            return False, f"Low sharpness ({sharpness:.0f} < {min_sharpness:.0f})"
        return True, f"Sharp ({sharpness:.0f})"

    @staticmethod
    def validate_contrast(aligned_face_bgr):
        """
        Contrast check using standard deviation. Config keys:
          - min_contrast
        """
        min_contrast = FaceValidator._get_th('min_contrast', 30.0, float)
        try:
            gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = aligned_face_bgr.copy()
        contrast = float(np.std(gray))
        if contrast < min_contrast:
            return False, f"Low contrast ({contrast:.0f} < {min_contrast:.0f})"
        return True, f"Contrast OK ({contrast:.0f})"

    @staticmethod
    def comprehensive_validation(aligned_face_bgr):
        """
        Full validation pipeline. Reads thresholds from config:
          - min_face_width, min_face_height
          - aspect_ratio_min (width/height)
          - aspect_ratio_max
        Returns (True, "PASS...") or (False, "reason")
        """
        # minimum dims
        min_width = FaceValidator._get_th('min_face_width', 80, int)
        min_height = FaceValidator._get_th('min_face_height', 80, int)

        aspect_min = FaceValidator._get_th('aspect_ratio_min', 1.0, float)
        aspect_max = FaceValidator._get_th('aspect_ratio_max', 1.85, float)

        if aligned_face_bgr is None:
            return False, "No face"

        try:
            h, w = aligned_face_bgr.shape[:2]
        except Exception:
            return False, "Bad image shape"

        if w < min_width or h < min_height:
            return False, f"Too small ({w}x{h} < {min_width}x{min_height})"

        # Compute aspect ratio as width/height (this matches common expectation)
        aspect = float(w) / (h + 1e-8)

        if aspect < aspect_min:
            return False, f"Aspect ratio too small (AR={aspect:.2f} < {aspect_min:.2f})"
        if aspect_max is not None and aspect > aspect_max:
            return False, f"Aspect ratio too large (AR={aspect:.2f} > {aspect_max:.2f})"

        # Now other photometric checks
        ok, msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
        if not ok:
            return False, msg

        ok, msg = FaceValidator.validate_sharpness(aligned_face_bgr)
        if not ok:
            return False, msg

        ok, msg = FaceValidator.validate_contrast(aligned_face_bgr)
        if not ok:
            return False, msg

        return True, f"PASS (AR={aspect:.2f}, {w}x{h})"
