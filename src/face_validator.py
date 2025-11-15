# import cv2
# import numpy as np
# import logging

# logger = logging.getLogger(__name__)

# class FaceValidator:
#     """
#     Simple geometric validator that doesn't rely on pose angles.
#     Uses aspect ratio + brightness distribution to detect non-frontal faces.
#     """
    
#     @staticmethod
#     def validate_aspect_ratio(aligned_face_bgr, min_ratio=1.15, max_ratio=2.0):
#         """
#         Check if face has proper height/width ratio.
#         Looking down = squashed face (ratio < 1.15)
#         Looking up = elongated face (ratio > 2.0)
#         """
#         h, w = aligned_face_bgr.shape[:2]
#         ratio = h / w
        
#         if ratio < min_ratio:
#             return False, f"Face too wide (AR={ratio:.2f}, looking down?)"
#         if ratio > max_ratio:
#             return False, f"Face too tall (AR={ratio:.2f}, looking up?)"
        
#         return True, f"Aspect ratio OK ({ratio:.2f})"
    
#     @staticmethod
#     def validate_brightness_distribution(aligned_face_bgr):
#         """
#         Check if brightness is evenly distributed.
        
#         When looking down:
#         - Top of head (dark hair) dominates
#         - Bottom half is darker (shadow under face)
        
#         When looking straight:
#         - Face is evenly lit
#         """
#         gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
#         h, w = gray.shape
        
#         # Split into top and bottom halves
#         top_half = gray[:h//2, :]
#         bottom_half = gray[h//2:, :]
        
#         top_mean = np.mean(top_half)
#         bottom_mean = np.mean(bottom_half)
        
#         # For a frontal face, top and bottom should be similar
#         # (eyes/forehead vs nose/mouth/chin)
#         brightness_ratio = top_mean / (bottom_mean + 1e-6)
        
#         # RELAXED: Allow more variation in lighting
#         if brightness_ratio > 1.6 or brightness_ratio < 0.6:
#             return False, f"Uneven brightness (ratio={brightness_ratio:.2f})"
        
#         return True, f"Brightness OK ({brightness_ratio:.2f})"
    
#     @staticmethod
#     def validate_edge_density(aligned_face_bgr):
#         """
#         Frontal faces have edges (features) distributed across the face.
#         Side/down faces have fewer edges or concentrated edges.
#         """
#         gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
#         # Detect edges
#         edges = cv2.Canny(gray, 50, 150)
        
#         # Count edge pixels
#         edge_density = np.sum(edges > 0) / edges.size
        
#         # RELAXED: Accept wider range of edge densities
#         if edge_density < 0.02:
#             return False, f"Too few features ({edge_density*100:.1f}% edges)"
#         if edge_density > 0.35:
#             return False, f"Too many edges ({edge_density*100:.1f}%)"
        
#         return True, f"Edge density OK ({edge_density*100:.1f}%)"
    
#     @staticmethod
#     def validate_face_symmetry(aligned_face_bgr):
#         """
#         Check if face is symmetric (left/right).
#         Side poses are highly asymmetric.
#         """
#         gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
#         h, w = gray.shape
        
#         # Split into left and right halves
#         mid = w // 2
#         left_half = gray[:, :mid]
#         right_half = gray[:, mid:w]
        
#         # Flip right half
#         right_flipped = cv2.flip(right_half, 1)
        
#         # Resize to match if needed
#         if left_half.shape[1] != right_flipped.shape[1]:
#             min_width = min(left_half.shape[1], right_flipped.shape[1])
#             left_half = left_half[:, :min_width]
#             right_flipped = right_flipped[:, :min_width]
        
#         # Calculate difference
#         diff = cv2.absdiff(left_half, right_flipped)
#         symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
#         # RELAXED: Lower threshold for symmetry
#         if symmetry_score < 0.5:
#             return False, f"Face not symmetric ({symmetry_score:.2f})"
        
#         return True, f"Symmetry OK ({symmetry_score:.2f})"
    
#     @staticmethod
#     def comprehensive_validation(aligned_face_bgr, mp_aspect_ratio=None):
#         """
#         Runs all validation checks.
#         Returns: (is_valid, failure_reason)
        
#         Args:
#             aligned_face_bgr: The aligned face image
#             mp_aspect_ratio: Optional MediaPipe's aspect ratio (more reliable than image dimensions)
#         """
#         # Check 1: Aspect Ratio - Use MediaPipe's if available
#         if mp_aspect_ratio is not None:
#             # MediaPipe's aspect ratio is more reliable
#             if mp_aspect_ratio < 1.15:
#                 return False, f"Face too wide (AR={mp_aspect_ratio:.2f}, looking down?)"
#             if mp_aspect_ratio > 2.0:
#                 return False, f"Face too tall (AR={mp_aspect_ratio:.2f})"
#         else:
#             # Fallback to image dimensions
#             ar_ok, ar_msg = FaceValidator.validate_aspect_ratio(aligned_face_bgr)
#             if not ar_ok:
#                 return False, ar_msg
        
#         # Check 2: Brightness Distribution
#         brightness_ok, brightness_msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
#         if not brightness_ok:
#             return False, brightness_msg
        
#         # Check 3: Edge Density (optional but helpful)
#         edge_ok, edge_msg = FaceValidator.validate_edge_density(aligned_face_bgr)
#         if not edge_ok:
#             return False, edge_msg
        
#         # Check 4: Symmetry (can be strict)
#         symmetry_ok, symmetry_msg = FaceValidator.validate_face_symmetry(aligned_face_bgr)
#         if not symmetry_ok:
#             return False, symmetry_msg
        
#         return True, "All geometric checks passed"


import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceValidator:
    """
    Simple geometric validator that doesn't rely on pose angles.
    Uses aspect ratio + brightness distribution to detect non-frontal faces.
    """
    
    @staticmethod
    def validate_aspect_ratio(aligned_face_bgr, min_ratio=1.15, max_ratio=2.0):
        """
        Check if face has proper height/width ratio.
        Looking down = squashed face (ratio < 1.15)
        Looking up = elongated face (ratio > 2.0)
        """
        h, w = aligned_face_bgr.shape[:2]
        ratio = h / w
        
        if ratio < min_ratio:
            return False, f"Face too wide (AR={ratio:.2f}, looking down?)"
        if ratio > max_ratio:
            return False, f"Face too tall (AR={ratio:.2f}, looking up?)"
        
        return True, f"Aspect ratio OK ({ratio:.2f})"
    
    @staticmethod
    def validate_brightness_distribution(aligned_face_bgr):
        """
        Check if brightness is evenly distributed.
        
        When looking down:
        - Top of head (dark hair) dominates
        - Bottom half is darker (shadow under face)
        
        When looking straight:
        - Face is evenly lit
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split into top and bottom halves
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        
        # For a frontal face, top and bottom should be similar
        # (eyes/forehead vs nose/mouth/chin)
        brightness_ratio = top_mean / (bottom_mean + 1e-6)
        
        # RELAXED: Allow more variation in lighting
        if brightness_ratio > 1.4 or brightness_ratio < 0.7:
            return False, f"Uneven brightness (ratio={brightness_ratio:.2f})"
        
        return True, f"Brightness OK ({brightness_ratio:.2f})"
    
    @staticmethod
    def validate_edge_density(aligned_face_bgr):
        """
        Frontal faces have edges (features) distributed across the face.
        Side/down faces have fewer edges or concentrated edges.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels
        edge_density = np.sum(edges > 0) / edges.size
        
        # RELAXED: Accept wider range of edge densities
        if edge_density < 0.02:
            return False, f"Too few features ({edge_density*100:.1f}% edges)"
        if edge_density > 0.35:
            return False, f"Too many edges ({edge_density*100:.1f}%)"
        
        return True, f"Edge density OK ({edge_density*100:.1f}%)"
    
    @staticmethod
    def validate_face_symmetry(aligned_face_bgr):
        """
        Check if face is symmetric (left/right).
        Side poses are highly asymmetric.
        """
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split into left and right halves
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = gray[:, mid:w]
        
        # Flip right half
        right_flipped = cv2.flip(right_half, 1)
        
        # Resize to match if needed
        if left_half.shape[1] != right_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
        
        # Calculate difference
        diff = cv2.absdiff(left_half, right_flipped)
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        # RELAXED: Lower threshold for symmetry
        if symmetry_score < 0.5:
            return False, f"Face not symmetric ({symmetry_score:.2f})"
        
        return True, f"Symmetry OK ({symmetry_score:.2f})"
    
    @staticmethod
    def comprehensive_validation(aligned_face_bgr, mp_aspect_ratio=None):
        """
        Runs all validation checks.
        Returns: (is_valid, failure_reason)
        
        Args:
            aligned_face_bgr: The aligned face image
            mp_aspect_ratio: Optional MediaPipe's aspect ratio (more reliable than image dimensions)
        """
        # Check 1: Aspect Ratio - Use MediaPipe's if available
        if mp_aspect_ratio is not None:
            # MediaPipe's aspect ratio is more reliable
            if mp_aspect_ratio < 1.25:
                return False, f"Face too wide (AR={mp_aspect_ratio:.2f}, looking down?)"
            if mp_aspect_ratio > 2.0:
                return False, f"Face too tall (AR={mp_aspect_ratio:.2f})"
        else:
            # Fallback to image dimensions
            ar_ok, ar_msg = FaceValidator.validate_aspect_ratio(aligned_face_bgr)
            if not ar_ok:
                return False, ar_msg
        
        # Check 2: Brightness Distribution
        brightness_ok, brightness_msg = FaceValidator.validate_brightness_distribution(aligned_face_bgr)
        if not brightness_ok:
            return False, brightness_msg
        
        # Check 3: Edge Density (optional but helpful)
        edge_ok, edge_msg = FaceValidator.validate_edge_density(aligned_face_bgr)
        if not edge_ok:
            return False, edge_msg
        
        # Check 4: Symmetry (can be strict)
        symmetry_ok, symmetry_msg = FaceValidator.validate_face_symmetry(aligned_face_bgr)
        if not symmetry_ok:
            return False, symmetry_msg
        
        return True, "All geometric checks passed"