import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceValidator:
    """
    STRICT geometric validator for frontal face detection.
    Multiple checks with HIGH thresholds to prevent bad faces.
    """
    
    @staticmethod
    def validate_brightness_distribution(aligned_face_bgr):
        """
        Check brightness across face regions.
        STRICTER than before to catch dark/side faces.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check overall brightness first
        mean_brightness = np.mean(gray)
        if mean_brightness < 60:  # Very dark image
            return False, f"Too dark (brightness={mean_brightness:.0f})"
        
        # Split into regions
        top_third = gray[:h//3, :]
        middle_third = gray[h//3:2*h//3, :]
        bottom_third = gray[2*h//3:, :]
        
        top_mean = np.mean(top_third)
        middle_mean = np.mean(middle_third)
        bottom_mean = np.mean(bottom_third)
        
        # Avoid division by zero
        if middle_mean < 10 or bottom_mean < 10:
            return False, "Extremely dark regions"
        
        top_to_middle = top_mean / (middle_mean + 1e-6)
        middle_to_bottom = middle_mean / (bottom_mean + 1e-6)
        
        # Stricter thresholds
        if top_to_middle < 0.7:  # Raised from 0.65
            return False, f"Dark top (ratio={top_to_middle:.2f})"
        
        if middle_to_bottom > 1.6:  # Lowered from 1.8 - stricter
            return False, f"Bright middle (ratio={middle_to_bottom:.2f})"
        
        if middle_to_bottom < 0.65:  # Raised from 0.6
            return False, f"Dark middle (ratio={middle_to_bottom:.2f})"
        
        return True, "Brightness OK"
    
    @staticmethod
    def validate_sharpness(aligned_face_bgr, min_sharpness=100):
        """
        NEW: Additional sharpness check using gradient magnitude.
        Catches blurry faces that Laplacian might miss.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        sharpness = np.mean(magnitude)
        
        if sharpness < min_sharpness:
            return False, f"Low sharpness ({sharpness:.0f})"
        
        return True, f"Sharp ({sharpness:.0f})"
    
    @staticmethod
    def validate_contrast(aligned_face_bgr, min_contrast=30):
        """
        NEW: Check if image has sufficient contrast.
        Low contrast = washed out/dark/poor quality.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
        contrast = gray.std()
        
        if contrast < min_contrast:
            return False, f"Low contrast ({contrast:.0f})"
        
        return True, f"Contrast OK"
    
    @staticmethod
    def validate_edge_density(aligned_face_bgr):
        """
        Edge density with STRICTER thresholds.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # STRICTER
        if edge_density < 0.03:  # Raised from 0.02
            return False, f"Too few edges ({edge_density*100:.1f}%)"
        if edge_density > 0.30:  # Lowered from 0.35
            return False, f"Too noisy ({edge_density*100:.1f}%)"
        
        return True, "Edges OK"
    
    @staticmethod
    def validate_face_symmetry(aligned_face_bgr):
        """
        Symmetry check with STRICTER threshold.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = gray[:, mid:w]
        
        right_flipped = cv2.flip(right_half, 1)
        
        if left_half.shape[1] != right_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
        
        diff = cv2.absdiff(left_half, right_flipped)
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        # STRICTER
        if symmetry_score < 0.55:  # Raised from 0.5
            return False, f"Asymmetric ({symmetry_score:.2f})"
        
        return True, "Symmetric"
    
    @staticmethod
    def validate_face_centering(aligned_face_bgr):
        """
        NEW: Check if face features are centered.
        Side profiles will have features off-center.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate center of mass of bright regions (face features)
        _, bright_regions = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        if np.sum(bright_regions) == 0:
            return False, "No face features detected"
        
        moments = cv2.moments(bright_regions)
        if moments['m00'] == 0:
            return False, "No feature mass"
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Center should be near image center
        center_x = w / 2
        center_y = h / 2
        
        offset_x = abs(cx - center_x) / w
        offset_y = abs(cy - center_y) / h
        
        # Allow 20% offset from center
        if offset_x > 0.2:
            return False, f"Face off-center X ({offset_x*100:.0f}%)"
        if offset_y > 0.2:
            return False, f"Face off-center Y ({offset_y*100:.0f}%)"
        
        return True, "Centered"
    
    @staticmethod
    def comprehensive_validation(aligned_face_bgr):
        """
        STRICT multi-stage validation.
        ALL checks must pass.
        """
        # Check 0: Size
        h, w = aligned_face_bgr.shape[:2]
        if h < 80 or w < 80:
            return False, f"Too small ({w}x{h})"
        
        # Check 1: Aspect Ratio - STRICTER
        image_aspect_ratio = h / w
        
        if image_aspect_ratio < 1.15:  # Raised from 1.10
            return False, f"Too wide (AR={image_aspect_ratio:.2f})"
        if image_aspect_ratio > 1.85:  # Lowered from 2.0
            return False, f"Too tall (AR={image_aspect_ratio:.2f})"
        
        # Check 2: Overall Brightness & Distribution
        brightness_ok, brightness_msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
        if not brightness_ok:
            return False, brightness_msg
        
        # Check 3: Sharpness (NEW)
        sharpness_ok, sharpness_msg = FaceValidator.validate_sharpness(aligned_face_bgr)
        if not sharpness_ok:
            return False, sharpness_msg
        
        # Check 4: Contrast (NEW)
        contrast_ok, contrast_msg = FaceValidator.validate_contrast(aligned_face_bgr)
        if not contrast_ok:
            return False, contrast_msg
        
        # Check 5: Edge Density
        edge_ok, edge_msg = FaceValidator.validate_edge_density(aligned_face_bgr)
        if not edge_ok:
            return False, edge_msg
        
        # Check 6: Symmetry
        symmetry_ok, symmetry_msg = FaceValidator.validate_face_symmetry(aligned_face_bgr)
        if not symmetry_ok:
            return False, symmetry_msg
        
        # Check 7: Centering (NEW)
        centering_ok, centering_msg = FaceValidator.validate_face_centering(aligned_face_bgr)
        if not centering_ok:
            return False, centering_msg
        
        # If borderline AR, require perfect scores on other checks
        if image_aspect_ratio < 1.25:
            # All other checks must have passed to reach here
            pass  # Already validated above
        
        return True, f"PASS (AR={image_aspect_ratio:.2f})"