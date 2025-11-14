import sys
import struct
import pickle
import numpy as np
import cv2
import traceback # Import traceback for detailed error logging

def log_stderr(message):
    """Helper function to write messages to stderr for the parent to log."""
    print(message, file=sys.stderr, flush=True)

try:
    # --- Helper Functions for Communication ---

    def read_message():
        """Reads a length-prefixed message from stdin."""
        len_bytes = sys.stdin.buffer.read(4)
        if not len_bytes:
            sys.exit(0) # Exit cleanly if stdin is closed
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
    log_stderr("WORKER: Checkpoint 1 - Importing MediaPipe...")
    import mediapipe as mp
    log_stderr("WORKER: Checkpoint 2 - MediaPipe imported successfully.")

    log_stderr("WORKER: Checkpoint 3 - Initializing FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,     # <-- ADD THIS LINE
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    log_stderr("WORKER: Checkpoint 4 - FaceMesh initialization successful.")

    # --- *** NEW: FROM YOUR OLD FILE *** ---
    # These are the correct 3D model points and landmark indices
    MP_LANDMARKS = {
        "nose": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291
    }

    # MODEL_3D = np.array([
    #     (0.0, 0.0, 0.0),       # Nose tip
    #     (0.0, 63.6, -12.5),    # Chin
    #     (-43.3, -32.7, -26.0), # Left eye corner
    #     (43.3, -32.7, -26.0),   # Right eye corner
    #     (-28.9, 28.9, -24.1),   # Left mouth corner
    #     (28.9, 28.9, -24.1)    # Right mouth corner
    # ], dtype=np.float64)



    MODEL_3D = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, 330.0, -65.0),     # Chin
        (-225.0, -170.0, -135.0), # Left eye
        (225.0, -170.0, -135.0),  # Right eye
        (-150.0, 150.0, -125.0),   # Left mouth
        (150.0, 150.0, -125.0)    # Right mouth
    ], dtype=np.float64)


    # --- *** END OF NEW SECTION *** ---
# # # 
#     def get_pose_and_alignment(image_rgb):
#         """
#         Runs MediaPipe Face Mesh to get aligned face and pose angles.
#         Returns: (aligned_face, yaw, pitch, roll) or (None, 0, 0, 0)
#         """
#         results = face_mesh.process(image_rgb)
        
#         if not results.multi_face_landmarks:
#             return None, 0, 0, 0

#         landmarks = results.multi_face_landmarks[0].landmark
        
#         # --- *** REPLACED: POSE ESTIMATION WITH YOUR OLD, STABLE LOGIC *** ---
#         h, w = image_rgb.shape[:2]

#         # Build 2D points in the SAME order as MODEL_3D
#         try:
#             landmarks_2d = np.array([
#                 (landmarks[MP_LANDMARKS["nose"]].x * w, landmarks[MP_LANDMARKS["nose"]].y * h),
#                 (landmarks[MP_LANDMARKS["chin"]].x * w, landmarks[MP_LANDMARKS["chin"]].y * h),
#                 (landmarks[MP_LANDMARKS["left_eye"]].x * w, landmarks[MP_LANDMARKS["left_eye"]].y * h),
#                 (landmarks[MP_LANDMARKS["right_eye"]].x * w, landmarks[MP_LANDMARKS["right_eye"]].y * h),
#                 (landmarks[MP_LANDMARKS["left_mouth"]].x * w, landmarks[MP_LANDMARKS["left_mouth"]].y * h),
#                 (landmarks[MP_LANDMARKS["right_mouth"]].x * w, landmarks[MP_LANDMARKS["right_mouth"]].y * h)
#             ], dtype=np.float64)
#         except Exception as e:
#             log_stderr(f"WORKER_ERROR: Landmark index error: {e}")
#             return None, 0, 0, 0 # Return None if a landmark is missing
        
#         # Camera matrix (from your old _get_pose function)
#         focal_length = (w + h) / 2.0
#         center = (w / 2.0, h / 2.0)
#         cam_matrix = np.array([
#             [focal_length, 0, center[0]],
#             [0, focal_length, center[1]],
#             [0, 0, 1]
#         ], dtype=np.float64)

#         dist_coeffs = np.zeros((4, 1), dtype=np.float64) # No distortion

#         # Run solvePnP
#         success, rvec, tvec = cv2.solvePnP(
#             MODEL_3D,
#             landmarks_2d,
#             cam_matrix,
#             dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE
#         )

#         if not success:
#             log_stderr("WORKER_ERROR: solvePnP failed to find pose.")
#             return None, 0, 0, 0
            
#         # Get Euler angles from rotation matrix
#         R, _ = cv2.Rodrigues(rvec)
#         euler_angles = cv2.RQDecomp3x3(R)[0]
        
#         pitch = float(euler_angles[0])
#         yaw = float(euler_angles[1])
#         roll = float(euler_angles[2])
        
#         # Flip Yaw sign to be intuitive (from your old code)
#         yaw = -yaw

#         # pitch = -pitch
#         # roll = -roll
#         if abs(roll) > 90:
#             roll = 180 - abs(roll) if roll > 0 else -(180 - abs(roll))
        
#         # --- *** END OF REPLACEMENT *** ---


#         # --- Face Alignment (This part was correct) ---
#         left_eye = landmarks[33]
#         right_eye = landmarks[263]
        
#         dx = right_eye.x - left_eye.x
#         dy = right_eye.y - left_eye.y
#         angle = np.degrees(np.arctan2(dy, dx))
        
#         eye_center = (
#             (left_eye.x + right_eye.x) * w / 2,
#             (left_eye.y + right_eye.y) * h / 2
#         )
        
#         M = cv2.getRotationMatrix2D(eye_center, angle, 1)
#         aligned_face = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_CUBIC)
        
#         return cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), yaw, pitch, roll

# --- DELETE THE OLD MODEL_3D and MP_LANDMARKS VARIABLES ---

    def get_pose_and_alignment(image_rgb):
        """
        NEW VERSION: Uses robust geometric heuristics for pose,
        which is much more stable than solvePnP.
        """
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, 0, 0, 0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image_rgb.shape[:2]

        # --- 1. Get Key Landmarks ---
        # We need pixel coordinates
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
            
            # Points for Yaw
            left_face_edge = get_px(130)  # Left cheek
            right_face_edge = get_px(359) # Right cheek
            
            # Points for Pitch
            forehead = get_px(10) # Top of forehead
            
        except IndexError:
            log_stderr("WORKER_ERROR: Landmark index out of range.")
            return None, 0, 0, 0

        # --- 2. Calculate ROLL (Tilt) ---
        # This is the same logic we use for alignment
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = np.degrees(np.arctan2(dy, dx))

# --- 3. Calculate PITCH (Up/Down) ---
        # NEW HEURISTIC: Compare face "height" (forehead-to-chin) 
        # to face "width" (eye-to-eye).
        try:
            # Horizontal distance between eyes
            eye_to_eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            
            # Vertical distance from forehead to chin
            forehead_to_chin_dist = chin[1] - forehead[1]

            # This ratio is the "aspect ratio" of the face.
            # A normal face is ~1.5. When looking down, it gets < 1.0.
            aspect_ratio = forehead_to_chin_dist / (eye_to_eye_dist + 1e-6)
            
            # We'll map this ratio to an angle.
            # (These are "magic numbers" that can be tuned.)
            # Let's say a normal ratio of 1.5 is 0 degrees.
            # A ratio of 1.0 (foreshortened) is 30 degrees.
            pitch = (1.5 - aspect_ratio) * 60.0

        except Exception:
            pitch = 0.0

        # --- 4. Calculate YAW (Left/Right) ---
        # Compare horizontal distance of nose-to-left vs nose-to-right
        try:
            total_face_width = right_face_edge[0] - left_face_edge[0]
            # Get ratio of how far nose is from the left edge
            yaw_ratio = (nose[0] - left_face_edge[0]) / (total_face_width + 1e-6)
            # Convert ratio (0.0 to 1.0) to angle (-90 to +90)
            # 0.5 ratio = 0 degrees.
            yaw = (yaw_ratio - 0.5) * 180.0 
        except Exception:
            yaw = 0.0

        # --- 5. Face Alignment (Same as before) ---
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2
        )
        
        M = cv2.getRotationMatrix2D(eye_center, roll, 1) # Use the roll we just calculated
        aligned_face = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), yaw, pitch, roll

    # --- Main Worker Loop ---
    def main_loop():
        """Reads an image, processes it, and writes back the result."""
        log_stderr("WORKER: Checkpoint 5 - Entering main loop, waiting for data...")
        while True:
            try:
                # 1. Read image from main process
                in_data = read_message()
                image_bgr = pickle.loads(in_data)
                
                # 2. Process with MediaPipe
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                result_tuple = get_pose_and_alignment(image_rgb)
                
                # 3. Create response and send back
                out_data = pickle.dumps(result_tuple)
                write_message(out_data)
                
            except (EOFError, struct.error, BrokenPipeError):
                log_stderr("WORKER: Pipe closed by main process. Exiting.")
                break
            except Exception as e:
                # Try to send a structured error back to the main process
                try:
                    err_str = traceback.format_exc()
                    log_stderr(f"WORKER_ERROR: Error during processing: {err_str}")
                    error_response = (None, 0.0, 0.0, 0.0, err_str)
                    out_data = pickle.dumps(error_response)
                    write_message(out_data)
                except Exception:
                    log_stderr("WORKER_ERROR: Failed to send error message. Exiting.")
                    break # Failed to send error, just exit

    if __name__ == '__main__':
        main_loop()

except Exception as e:
    # --- THIS IS THE CRITICAL ADDITION ---
    # If *anything* fails during setup (imports, init), this will catch it
    # and write the full traceback to stderr for the parent to log.
    log_stderr("WORKER_FATAL_ERROR: Worker failed during initialization.")
    log_stderr(traceback.format_exc())
    sys.exit(1) # Exit with a non-zero code to signal failure