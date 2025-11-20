# src/main.py
import cv2
import numpy as np
import time
import logging
import sys
import os
from datetime import date

from src.config_loader import load_config
from src.logger_setup import setup_logging
from src.video_stream import VideoStream
from src.model_loader import load_yolo_model, build_reference_database
from src.mediapipe_client import MediaPipeClient
from src.face_processor import FaceProcessor
from src.attendance_manager import AttendanceManager
from src.face_validator import FaceValidator

def main():
    try:
        config = load_config('config.ini')
        os.makedirs(config.get('Paths','debug_aligned_faces_path', fallback='debug_aligned_faces/'), exist_ok=True)

        logger = setup_logging(config)
    except Exception as e:
        print(f"FATAL: Failed to load config or setup logger: {e}")
        sys.exit(1)

    try:
        yolo_model = load_yolo_model(config)
        face_processor = FaceProcessor(config)
        reference_db = build_reference_database(config, face_processor)

        stream = VideoStream(config)
        mp_client = MediaPipeClient(config)
        att_manager = AttendanceManager(config, reference_db)

        # Config params
        yolo_conf = config.getfloat('Model_Settings', 'yolo_conf_threshold')
        padding = config.getint('Model_Settings', 'padding')
        blur_thresh = config.getfloat('Quality_Filters', 'blur_threshold')
        ema_alpha = config.getfloat('Performance', 'ema_alpha')
        min_valid_frames = config.getint('Performance', 'min_valid_frames', fallback=2)
        required_matches = config.getint('Performance', 'required_match_count', fallback=3)
        min_sobel = config.getfloat('Quality_Filters', 'min_sobel_sharpness', fallback=25.0)
        embedding_every = config.getint('Performance', 'embedding_every_frames', fallback=2)

        # Persistent ID mapping (YOLO internal -> persistent PID)
        yolo_to_pid = {}              # {yolo_id: pid}
        pid_next = 1                  # monotonic counter, reset daily
        pid_assigned_date = date.today()

        # tracked_faces keyed by PID (persistent id)
        # structure per PID:
        # {
        #   'yolo_id': int,
        #   'ema_embedding': np.array or None,
        #   'name': str,
        #   'valid_count': int,
        #   'match_count': int,
        #   'last_confirmed': timestamp (float)
        # }
        tracked_faces = {}

        frame_num = 0

        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            # daily reset of PID mapping
            today = date.today()
            if today != pid_assigned_date:
                logger.info("Day changed - resetting persistent PID mapping and counter.")
                yolo_to_pid.clear()
                tracked_faces.clear()
                pid_next = 1
                pid_assigned_date = today

            frame_num += 1

            results = yolo_model.track(
                frame,
                conf=yolo_conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            current_yolo_ids = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                current_yolo_ids = list(track_ids)

                # Ensure mapping exists for each YOLO track id -> persistent pid
                for yid in current_yolo_ids:
                    if yid not in yolo_to_pid:
                        yolo_to_pid[yid] = pid_next
                        pid_next += 1

                # Build reverse map for quick lookup of current pids
                current_pids = [yolo_to_pid[yid] for yid in current_yolo_ids]

                for box, yolo_id in zip(boxes, track_ids):
                    pid = yolo_to_pid[int(yolo_id)]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                    status_text = ""
                    pose_text = ""
                    status_color = (0, 0, 255)

                    p_x1 = max(0, x1 - padding)
                    p_y1 = max(0, y1 - padding)
                    p_x2 = min(frame.shape[1], x2 + padding)
                    p_y2 = min(frame.shape[0], y2 + padding)

                    face_crop = frame[p_y1:p_y2, p_x1:p_x2]
                    if face_crop.size == 0:
                        continue

                    # Initialize tracked_faces entry for pid if missing
                    if pid not in tracked_faces:
                        tracked_faces[pid] = {
                            'yolo_id': int(yolo_id),
                            'ema_embedding': None,
                            'name': 'Unknown',
                            'valid_count': 0,
                            'match_count': 0,
                            'last_confirmed': None
                        }
                    else:
                        # keep yolo mapping up-to-date (YOLO internal id may change if tracker re-assigned)
                        tracked_faces[pid]['yolo_id'] = int(yolo_id)

                    # === FILTER 1: Initial Blur Check (Laplacian primary) ===
                    try:
                        if hasattr(face_processor, 'calculate_blur_laplacian'):
                            initial_blur = face_processor.calculate_blur_laplacian(face_crop)
                        else:
                            initial_blur = face_processor.calculate_blur(face_crop)
                    except Exception:
                        initial_blur = face_processor.calculate_blur(face_crop)

                    if initial_blur < blur_thresh:
                        status_text = f"Too Blurry ({initial_blur:.0f})"
                        status_color = (0, 0, 255)
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === FILTER 2: MediaPipe Alignment ===
                    aligned_face, yaw, pitch, roll, aspect_ratio = mp_client.get_alignment_and_pose(face_crop)
                    if aligned_face is None:
                        status_text = "MP Failed"
                        status_color = (0, 0, 255)
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    pose_text = f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}"

                    # === FILTER 3: Pose check ===
                    yaw_thresh = config.getfloat('Quality_Filters', 'yaw_threshold', fallback=15.0)
                    pitch_thresh = config.getfloat('Quality_Filters', 'pitch_threshold', fallback=15.0)
                    roll_thresh = config.getfloat('Quality_Filters', 'roll_threshold', fallback=15.0)
                    if abs(yaw) > yaw_thresh or abs(pitch) > pitch_thresh or abs(roll) > roll_thresh:
                        status_text = f"Tilted (Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f})"
                        status_color = (0, 0, 255)
                        tracked_faces[pid]['ema_embedding'] = None
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === FILTER 4: Re-check blur on aligned face ===
                    try:
                        if hasattr(face_processor, 'calculate_blur_laplacian'):
                            aligned_blur = face_processor.calculate_blur_laplacian(aligned_face)
                        else:
                            aligned_blur = face_processor.calculate_blur(aligned_face)
                    except Exception:
                        aligned_blur = face_processor.calculate_blur(aligned_face)

                    if aligned_blur < blur_thresh:
                        status_text = f"Aligned Blur ({aligned_blur:.0f})"
                        status_color = (0, 0, 255)
                        tracked_faces[pid]['ema_embedding'] = None
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === FILTER 5: Sobel/motion check (if available) ===
                    sobel_ok = True
                    sobel_score = None
                    try:
                        if hasattr(face_processor, 'calculate_sobel_sharpness'):
                            sobel_score = face_processor.calculate_sobel_sharpness(aligned_face)
                            if sobel_score < min_sobel:
                                sobel_ok = False
                                status_text = f"Motion Blur ({sobel_score:.0f})"
                                status_color = (0, 0, 255)
                    except Exception:
                        sobel_ok = True

                    if not sobel_ok:
                        tracked_faces[pid]['ema_embedding'] = None
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0
                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === Geometric validation ===
                    is_valid_face, validation_reason = FaceValidator.comprehensive_validation(aligned_face)
                    if not is_valid_face:
                        status_text = f"REJECT: {validation_reason}"
                        status_color = (0, 0, 255)
                        tracked_faces[pid]['ema_embedding'] = None
                        tracked_faces[pid]['valid_count'] = 0
                        tracked_faces[pid]['match_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, pose_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        continue

                    # === Stabilization count ===
                    tracked_faces[pid]['valid_count'] += 1
                    if tracked_faces[pid]['valid_count'] < min_valid_frames:
                        status_text = f"Stabilizing ({tracked_faces[pid]['valid_count']}/{min_valid_frames})"
                        status_color = (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === ALL CHECKS PASSED: Recognition processing ===
                    preprocessed_face = cv2.resize(aligned_face, (112, 112))

                    # Compute embedding only every `embedding_every` valid frames
                    compute_embedding_now = (tracked_faces[pid]['valid_count'] % embedding_every == 0)

                    current_embedding = None
                    if compute_embedding_now:
                        current_embedding = face_processor.get_embedding(preprocessed_face)

                    if current_embedding is None and not compute_embedding_now:
                        smoothed_embedding = tracked_faces[pid]['ema_embedding']
                    elif current_embedding is not None:
                        prev_ema = tracked_faces[pid]['ema_embedding']
                        smoothed_embedding = face_processor.apply_ema(current_embedding, prev_ema, ema_alpha)
                        tracked_faces[pid]['ema_embedding'] = smoothed_embedding
                    else:
                        # embedding failed
                        tracked_faces[pid]['match_count'] = 0
                        smoothed_embedding = None

                    # --- DEBUG SAVE: write aligned/preprocessed face for inspection ---
                    if config.getboolean('Logging', 'debug_save_faces', fallback=False):
                        try:
                            debug_path = config.get('Paths', 'debug_aligned_faces_path', fallback='debug_aligned_faces/')
                            os.makedirs(debug_path, exist_ok=True)
                            # safe yolo_id retrieval
                            yid_for_name = int(yolo_id) if 'yolo_id' in locals() else int(track_ids[0]) if len(track_ids)>0 else -1
                            lap = int(aligned_blur) if 'aligned_blur' in locals() else 0
                            sob = int(sobel_score) if ('sobel_score' in locals() and sobel_score is not None) else 0
                            dbg_fn = os.path.join(debug_path, f"pid_{pid}_yid_{yid_for_name}_frame_{frame_num}_lap{lap}_sob{sob}.jpg")
                            ok = cv2.imwrite(dbg_fn, preprocessed_face)
                            # if ok:
                            #     logger.info(f"Wrote debug face: {dbg_fn}")
                            # else:
                            #     logger.warning(f"Failed to write debug face: {dbg_fn}")
                        except Exception as e:
                            logger.warning(f"Exception while saving debug face: {e}")

                    if smoothed_embedding is not None:
                        name, similarity = att_manager.find_match(smoothed_embedding)

                        # If matched to a known person
                        if name != "Unknown" and similarity >= att_manager.threshold:
                            # If previous stored name differs, reset its match_count
                            if tracked_faces[pid].get('name') != name:
                                tracked_faces[pid]['match_count'] = 0

                            tracked_faces[pid]['name'] = name
                            tracked_faces[pid]['match_count'] += 1

                            # Only mark attendance when we reached required consecutive matches
                            if tracked_faces[pid]['match_count'] >= required_matches:
                                last_confirmed = tracked_faces[pid].get('last_confirmed')
                                now_ts = time.time()
                                if last_confirmed is None or (now_ts - last_confirmed) > max(1.0, att_manager.mark_cooldown - 1):
                                    success = att_manager.mark_attendance(name)
                                    if success:
                                        tracked_faces[pid]['last_confirmed'] = now_ts
                                # reset match_count after marking
                                tracked_faces[pid]['match_count'] = 0

                            status_text = f"{name} ({similarity:.2f})"
                            status_color = (0, 255, 0)
                        else:
                            # Unknown result: reset match_count and optionally handle unknown capture
                            tracked_faces[pid]['match_count'] = 0
                            tracked_faces[pid]['name'] = 'Unknown'
                            # pass persistent PID to unknown handler
                            att_manager.handle_unknown(preprocessed_face, pid)
                            status_text = f"Unknown ({similarity:.2f})"
                            status_color = (0, 255, 255)
                    else:
                        status_text = "No Embedding"
                        status_color = (0, 165, 255)

                    # === Visualization ===
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                    (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), status_color, -1)
                    cv2.putText(frame, status_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Show persistent PID on screen
                    cv2.putText(frame, f"PID: {pid}", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if pose_text:
                        cv2.putText(frame, pose_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Clean up removed tracks:
            # Compute set of PIDs that are currently present (from current YOLO ids)
            current_pids_set = set()
            for yid in current_yolo_ids:
                if yid in yolo_to_pid:
                    current_pids_set.add(yolo_to_pid[yid])

            # Remove tracked_faces for PIDs not present in current frame
            for pid in list(tracked_faces.keys()):
                if pid not in current_pids_set:
                    del tracked_faces[pid]

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        try:
            stream.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        logger.info("Application shutting down.")

if __name__ == '__main__':
    main()
