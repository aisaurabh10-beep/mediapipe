import cv2
import numpy as np
import time
import logging
import subprocess
import os
import atexit
import sys

# Import project modules
from src.config_loader import load_config
from src.logger_setup import setup_logging
from src.video_stream import VideoStream
from src.model_loader import load_yolo_model, build_reference_database
from src.mediapipe_client import MediaPipeClient
from src.face_processor import FaceProcessor
from src.attendance_manager import AttendanceManager
import cv2


# # ByteTrack arguments
# class ByteTrackArgs:
#     track_thresh = 0.5
#     track_buffer = 30
#     match_thresh = 0.8
#     aspect_ratio_thresh = 1.6
#     min_box_area = 10
#     mot20 = False

# Global variable for the MediaPipe server process
mp_server_process = None


def main():
    try:
        # 1. Load Config and Logger
        config = load_config('config.ini')
        logger = setup_logging(config)
    except Exception as e:
        print(f"FATAL: Failed to load config or setup logger: {e}")
        sys.exit(1)

    try:


        # 3. Load Models
        yolo_model = load_yolo_model(config)
        face_processor = FaceProcessor(config)
        reference_db = build_reference_database(config, face_processor)
        
        # 4. Init Managers
        stream = VideoStream(config)
        mp_client = MediaPipeClient(config)
        att_manager = AttendanceManager(config, reference_db)
        # tracker = ByteTracker(ByteTrackArgs())
        
        # 5. Load Config Values
        yolo_conf = config.getfloat('Model_Settings', 'yolo_conf_threshold')
        padding = config.getint('Model_Settings', 'padding')
        
        blur_thresh = config.getfloat('Quality_Filters', 'blur_threshold')
        yaw_thresh = config.getfloat('Quality_Filters', 'yaw_threshold')
        pitch_thresh = config.getfloat('Quality_Filters', 'pitch_threshold')
        roll_thresh = config.getfloat('Quality_Filters', 'roll_threshold')
        aspect_ratio_thresh = config.getfloat('Quality_Filters', 'aspect_ratio_threshold') # <-- ADD THIS
        
        ema_alpha = config.getfloat('Performance', 'ema_alpha')
        
        frame_num = 0
        tracked_faces = {} # {tracker_id: {'ema_embedding': ndarray, 'name': str}}

        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue
                
            frame_num += 1
            
            # --- 1. YOLOv8n-face (Detection + Tracking) ---
            results = yolo_model.track(
                frame,
                conf=yolo_conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            current_track_ids = []
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                current_track_ids = list(track_ids)
                
                # --- 3. Process each tracked face ---
                for box, track_id in zip(boxes, track_ids):
                    
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # --- *** NEW: Initialize draw variables *** ---
                    status_text = ""
                    pose_text = ""
                    status_color = (0, 0, 255) # Red (Bad) by default
                    
                    p_x1 = max(0, x1 - padding)
                    p_y1 = max(0, y1 - padding)
                    p_x2 = min(frame.shape[1], x2 + padding)
                    p_y2 = min(frame.shape[0], y2 + padding)
                    
                    face_crop = frame[p_y1:p_y2, p_x1:p_x2]

                    if face_crop.size == 0:
                        continue
                    
                    # --- 4. Quality Filter: Blur ---
                    blur_score = face_processor.calculate_blur(face_crop)
                    if blur_score < blur_thresh:
                        status_text = f"Too Blurry ({blur_score:.0f})"
                    
                    else:
                        # --- 5. MediaPipe (Alignment & Pose) ---
                        aligned_face, yaw, pitch, roll,aspect_ratio = mp_client.get_alignment_and_pose(face_crop)
                        
                        if aligned_face is None:
                            status_text = "MP Failed"
                        else:
                            # --- *** NEW: Always show pose *** ---
                            pose_text = f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}"

                            # --- 6. Quality Filter: Pose ---
                            if aspect_ratio < aspect_ratio_thresh:
                                status_text = f"Bad Aspect Ratio ({aspect_ratio:.2f})"


                            elif (abs(yaw) > yaw_thresh or
                                abs(pitch) > pitch_thresh or
                                abs(roll) > roll_thresh):
                                
                                # --- *** This is your request *** ---
                                status_text = "Face is not aligned"
                            
                            else:
                                # --- 7. CLAHE (Preprocessing) ---
                                preprocessed_face = face_processor.preprocess(aligned_face, track_id, frame_num)
                                
                                # --- 8. ArcFace (Embedding) ---
                                current_embedding = face_processor.get_embedding(preprocessed_face)
                                if current_embedding is None:
                                    status_text = "ArcFace Failed"
                                else:
                                    # --- 9. Temporal EMA (Smoothing) ---
                                    if track_id not in tracked_faces:
                                        tracked_faces[track_id] = {'ema_embedding': None, 'name': 'Unknown'}
                                    
                                    prev_ema = tracked_faces[track_id]['ema_embedding']
                                    smoothed_embedding = face_processor.apply_ema(current_embedding, prev_ema, ema_alpha)
                                    tracked_faces[track_id]['ema_embedding'] = smoothed_embedding

                                    # --- 10. Match (Cosine/SVM) ---
                                    name, similarity = att_manager.find_match(smoothed_embedding)
                                    
                                    # --- 11. Attendance Trigger ---
                                    if name != "Unknown":
                                        tracked_faces[track_id]['name'] = name
                                        att_manager.mark_attendance(name)
                                        status_text = f"{name} ({similarity:.2f})"
                                        status_color = (0, 255, 0) # Green (Good)
                                    else:
                                        att_manager.handle_unknown(face_crop, track_id)
                                        status_text = f"Unknown ({similarity:.2f})"
                                        status_color = (0, 255, 255) # Yellow (Unknown)
                    
                    # --- 12. Visualization (Universal Drawing) ---
                    # Draw the main bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                    
                    # Draw the text background
                    (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), status_color, -1)
                    
                    # Draw the status text (e.g., Name, "Too Blurry", "Not Aligned")
                    cv2.putText(frame, status_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Draw the Track ID
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Draw the Pose Angles (if they exist)
                    if pose_text:
                        cv2.putText(frame, pose_text, (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- Clean up old tracks ---
            for track_id in list(tracked_faces.keys()):
                if track_id not in current_track_ids:
                    logger.debug(f"Removing stale track ID: {track_id}")
                    del tracked_faces[track_id]

            # --- Display Frame ---
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        stream.stop()
        cv2.destroyAllWindows()
        # The atexit hook will stop the mp_server
        logger.info("Application shutting down.")

if __name__ == '__main__':
    main()