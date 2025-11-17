import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceValidator:
    """
    Geometric validator that checks if a face is frontal and suitable for recognition.
    Uses aspect ratio, brightness distribution, edge density, and symmetry.
    """
    
    @staticmethod
    def validate_brightness_distribution(aligned_face_bgr):
        """
        Check if brightness is evenly distributed across face regions.
        Detects when top-of-head (hair) is visible vs actual face.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split into THREE sections
        top_third = gray[:h//3, :]        # Forehead/hair region
        middle_third = gray[h//3:2*h//3, :]  # Eyes/nose region
        bottom_third = gray[2*h//3:, :]   # Mouth/chin region
        
        top_mean = np.mean(top_third)
        middle_mean = np.mean(middle_third)
        bottom_mean = np.mean(bottom_third)
        
        top_to_middle = top_mean / (middle_mean + 1e-6)
        middle_to_bottom = middle_mean / (bottom_mean + 1e-6)
        
        # Looking down: dark hair on top, face in middle/bottom
        if top_to_middle < 0.65:
            return False, f"Dark top (ratio={top_to_middle:.2f})"
        
        # Looking down: face foreshortened, bright middle
        if middle_to_bottom > 1.8:
            return False, f"Bright middle (ratio={middle_to_bottom:.2f})"
        
        # Looking up: rare but check anyway
        if middle_to_bottom < 0.6:
            return False, f"Dark middle (ratio={middle_to_bottom:.2f})"
        
        return True, f"Brightness OK"
    
    @staticmethod
    def validate_edge_density(aligned_face_bgr):
        """
        Frontal faces have consistent edge distribution.
        Too few edges = featureless (side view or blur)
        Too many edges = noise or bad alignment
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.02:
            return False, f"Too few features ({edge_density*100:.1f}%)"
        if edge_density > 0.35:
            return False, f"Too noisy ({edge_density*100:.1f}%)"
        
        return True, f"Edges OK"
    
    @staticmethod
    def validate_face_symmetry(aligned_face_bgr):
        """
        Check left/right symmetry. Side poses fail this check.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = gray[:, mid:w]
        
        right_flipped = cv2.flip(right_half, 1)
        
        # Ensure same dimensions
        if left_half.shape[1] != right_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
        
        diff = cv2.absdiff(left_half, right_flipped)
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        if symmetry_score < 0.5:
            return False, f"Not symmetric ({symmetry_score:.2f})"
        
        return True, f"Symmetric OK"
    
    @staticmethod
    def comprehensive_validation(aligned_face_bgr):
        """
        Runs all validation checks with smart decision logic.
        Returns: (is_valid, failure_reason)
        """
        # Check 0: Minimum face size
        h, w = aligned_face_bgr.shape[:2]
        if h < 80 or w < 80:
            return False, f"Too small ({w}x{h})"
        
        # Check 1: Aspect Ratio - More permissive now
        image_aspect_ratio = h / w
        
        if image_aspect_ratio < 1.10:
            return False, f"Too wide (AR={image_aspect_ratio:.2f})"
        if image_aspect_ratio > 2.0:
            return False, f"Too tall (AR={image_aspect_ratio:.2f})"
        
        # Check 2: Brightness Distribution (Critical for detecting looking down)
        brightness_ok, brightness_msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
        
        # Check 3: Edge Density
        edge_ok, edge_msg = FaceValidator.validate_edge_density(aligned_face_bgr)
        
        # Check 4: Symmetry
        symmetry_ok, symmetry_msg = FaceValidator.validate_face_symmetry(aligned_face_bgr)
        
        # SMART LOGIC: If AR is borderline (1.10-1.20), require at least 2 other checks to pass
        if image_aspect_ratio < 1.20:
            checks_passed = sum([brightness_ok, edge_ok, symmetry_ok])
            if checks_passed < 2:
                # Find which check failed
                if not brightness_ok:
                    return False, f"Low AR + {brightness_msg}"
                if not edge_ok:
                    return False, f"Low AR + {edge_msg}"
                if not symmetry_ok:
                    return False, f"Low AR + {symmetry_msg}"
        
        # For normal AR (≥1.20), apply standard checks
        if not brightness_ok:
            return False, brightness_msg
        
        if not edge_ok:
            return False, edge_msg
        
        if not symmetry_ok:
            return False, symmetry_msg
        
        return True, f"Valid (AR={image_aspect_ratio:.2f})"