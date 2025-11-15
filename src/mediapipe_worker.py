import sys
import struct
import pickle
import numpy as np
import cv2
import traceback

def log_stderr(message):
    """Helper function to write messages to stderr for the parent to log."""
    print(message, file=sys.stderr, flush=True)

try:
    # --- Helper Functions for Communication ---
    def read_message():
        """Reads a length-prefixed message from stdin."""
        len_bytes = sys.stdin.buffer.read(4)
        if not len_bytes:
            sys.exit(0)
        msg_len = struct.unpack('!I', len_bytes)[0]
        msg_data = sys.stdin.buffer.read(msg_len)
        return msg_data

    def write_message(data):
        """Writes a length-prefixed message to stdout."""
        len_bytes = struct.pack('!I', len(data))
        sys.stdout.buffer.write(len_bytes)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    # --- MediaPipe Setup ---
    log_stderr("WORKER: Importing MediaPipe...")
    import mediapipe as mp
    log_stderr("WORKER: MediaPipe imported successfully.")

    log_stderr("WORKER: Initializing FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    log_stderr("WORKER: FaceMesh initialization successful.")

    # --- 3D Model Points (Canonical Face) ---
    MP_LANDMARKS = {
        "nose": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291
    }

    MODEL_3D = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, 330.0, -65.0),     # Chin
        (-225.0, -170.0, -135.0), # Left eye
        (225.0, -170.0, -135.0),  # Right eye
        (-150.0, 150.0, -125.0),  # Left mouth
        (150.0, 150.0, -125.0)   # Right mouth
    ], dtype=np.float64)

    def get_pose_and_alignment(image_rgb):
        """
        HYBRID APPROACH:
        1. Use solvePnP for accurate pose (with fallback)
        2. Add geometric validation (aspect ratio, eye visibility)
        3. Return alignment + pose + aspect_ratio
        """
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, 0, 0, 0, 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image_rgb.shape[:2]

        # --- Get Key Landmarks ---
        def get_px(index):
            lm = landmarks[index]
            return lm.x * w, lm.y * h

        try:
            left_eye = get_px(33)
            right_eye = get_px(263)
            nose = get_px(1)
            chin = get_px(152)
            left_mouth = get_px(61)
            right_mouth = get_px(291)
            forehead = get_px(10)
            
        except IndexError:
            log_stderr("WORKER_ERROR: Landmark index out of range.")
            return None, 0, 0, 0, 0.0

# --- 1. Calculate Aspect Ratio (Height/Width) ---
        try:
            left_face_edge = get_px(130)  # Left cheek
            right_face_edge = get_px(359) # Right cheek

            face_height_dist = forehead_to_chin_dist = chin[1] - forehead[1]
            face_width_dist = right_face_edge[0] - left_face_edge[0]
        
            if face_width_dist < 1e-6:
                return None, 0, 0, 0, 0.0
                
            aspect_ratio = face_height_dist / face_width_dist
        
        except IndexError:
            log_stderr("WORKER_ERROR: Landmark index out of range for AR.")
            return None, 0, 0, 0, 0.0

        # --- 2. Calculate ROLL (for alignment) ---
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll_angle = np.degrees(np.arctan2(dy, dx))

        # --- 3. Try solvePnP for Accurate Pose ---
        yaw, pitch, roll = 0.0, 0.0, 0.0
        
        try:
            # Build 2D points in same order as MODEL_3D
            landmarks_2d = np.array([
                (landmarks[MP_LANDMARKS["nose"]].x * w, landmarks[MP_LANDMARKS["nose"]].y * h),
                (landmarks[MP_LANDMARKS["chin"]].x * w, landmarks[MP_LANDMARKS["chin"]].y * h),
                (landmarks[MP_LANDMARKS["left_eye"]].x * w, landmarks[MP_LANDMARKS["left_eye"]].y * h),
                (landmarks[MP_LANDMARKS["right_eye"]].x * w, landmarks[MP_LANDMARKS["right_eye"]].y * h),
                (landmarks[MP_LANDMARKS["left_mouth"]].x * w, landmarks[MP_LANDMARKS["left_mouth"]].y * h),
                (landmarks[MP_LANDMARKS["right_mouth"]].x * w, landmarks[MP_LANDMARKS["right_mouth"]].y * h)
            ], dtype=np.float64)
            
            # Camera matrix
            focal_length = (w + h) / 2.0
            center = (w / 2.0, h / 2.0)
            cam_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Run solvePnP
            success, rvec, tvec = cv2.solvePnP(
                MODEL_3D,
                landmarks_2d,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Get Euler angles
                R, _ = cv2.Rodrigues(rvec)
                euler_angles = cv2.RQDecomp3x3(R)[0]
                
                pitch = float(euler_angles[0])
                yaw = -float(euler_angles[1])  # Flip for intuitive direction
                roll = float(euler_angles[2])
                
                # Normalize roll to [-90, 90]
                if abs(roll) > 90:
                    roll = 180 - abs(roll) if roll > 0 else -(180 - abs(roll))
            else:
                log_stderr("WORKER_WARNING: solvePnP failed, using fallback pose estimation")
                # Fallback to geometric heuristics
                pitch = (1.5 - aspect_ratio) * 60.0
                
                # Simple yaw from nose position
                left_face = get_px(130)
                right_face = get_px(359)
                total_width = right_face[0] - left_face[0]
                if total_width > 1e-6:
                    yaw_ratio = (nose[0] - left_face[0]) / total_width
                    yaw = (yaw_ratio - 0.5) * 180.0
                
                roll = roll_angle
                
        except Exception as e:
            log_stderr(f"WORKER_ERROR: Pose estimation failed: {e}")
            # Return geometric fallback
            pitch = (1.5 - aspect_ratio) * 60.0
            yaw = 0.0
            roll = roll_angle

        # --- 4. Face Alignment ---
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )
        
        M = cv2.getRotationMatrix2D(eye_center, roll_angle, 1)
        aligned_face = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), yaw, pitch, roll, aspect_ratio

    # --- Main Worker Loop ---
    def main_loop():
        """Reads an image, processes it, and writes back the result."""
        log_stderr("WORKER: Entering main loop, waiting for data...")
        while True:
            try:
                in_data = read_message()
                image_bgr = pickle.loads(in_data)
                
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                result_tuple = get_pose_and_alignment(image_rgb)
                
                out_data = pickle.dumps(result_tuple)
                write_message(out_data)
                
            except (EOFError, struct.error, BrokenPipeError):
                log_stderr("WORKER: Pipe closed by main process. Exiting.")
                break
            except Exception as e:
                try:
                    err_str = traceback.format_exc()
                    log_stderr(f"WORKER_ERROR: {err_str}")
                    error_response = (None, 0.0, 0.0, 0.0, 0.0, err_str)
                    out_data = pickle.dumps(error_response)
                    write_message(out_data)
                except Exception:
                    log_stderr("WORKER_ERROR: Failed to send error message. Exiting.")
                    break

    if __name__ == '__main__':
        main_loop()

except Exception as e:
    log_stderr("WORKER_FATAL_ERROR: Worker failed during initialization.")
    log_stderr(traceback.format_exc())
    sys.exit(1)