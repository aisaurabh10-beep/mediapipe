import cv2
import numpy as np
import time
import logging
import sys
import os

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

        yolo_conf = config.getfloat('Model_Settings', 'yolo_conf_threshold')
        padding = config.getint('Model_Settings', 'padding')
        blur_thresh = config.getfloat('Quality_Filters', 'blur_threshold')
        ema_alpha = config.getfloat('Performance', 'ema_alpha')
        min_valid_frames = config.getint('Performance', 'min_valid_frames', fallback=2)

        frame_num = 0
        tracked_faces = {}

        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            frame_num += 1

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

                for box, track_id in zip(boxes, track_ids):
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

                    # === FILTER 1: Initial Blur Check (use Laplacian) ===
                    try:
                        aligned_lap = None
                        if hasattr(face_processor, 'calculate_blur_laplacian'):
                            aligned_lap = face_processor.calculate_blur_laplacian(face_crop)
                        else:
                            aligned_lap = face_processor.calculate_blur(face_crop)
                    except Exception:
                        aligned_lap = face_processor.calculate_blur(face_crop)

                    if aligned_lap < blur_thresh:
                        status_text = f"Too Blurry ({aligned_lap:.0f})"
                        status_color = (0, 0, 255)

                        if track_id in tracked_faces:
                            tracked_faces[track_id]['ema_embedding'] = None
                            tracked_faces[track_id]['valid_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), status_color, -1)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        continue

                    # === FILTER 2: MediaPipe Alignment ===
                    aligned_face, yaw, pitch, roll, aspect_ratio = mp_client.get_alignment_and_pose(face_crop)

                    if aligned_face is None:
                        status_text = "MP Failed"
                        status_color = (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        continue

                    pose_text = f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}"

                    # === FILTER 3: Pose check BEFORE further processing ===
                    yaw_thresh = config.getfloat('Quality_Filters', 'yaw_threshold', fallback=15.0)
                    pitch_thresh = config.getfloat('Quality_Filters', 'pitch_threshold', fallback=15.0)
                    roll_thresh = config.getfloat('Quality_Filters', 'roll_threshold', fallback=15.0)

                    if abs(yaw) > yaw_thresh or abs(pitch) > pitch_thresh or abs(roll) > roll_thresh:
                        status_text = f"Tilted (Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f})"
                        status_color = (0, 0, 255)
                        if track_id in tracked_faces:
                            tracked_faces[track_id]['ema_embedding'] = None
                            tracked_faces[track_id]['valid_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        continue

                    # === FILTER 4: Re-check Blur on Aligned Face (Laplacian primary) ===
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
                        logger.debug(f"Track {track_id} REJECTED: Aligned face blur {aligned_blur:.0f}")

                        if track_id in tracked_faces:
                            tracked_faces[track_id]['ema_embedding'] = None
                            tracked_faces[track_id]['valid_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        continue

                    # === FILTER 5: Secondary Sobel motion check (if available) ===
                    sobel_ok = True
                    try:
                        if hasattr(face_processor, 'calculate_sobel_sharpness'):
                            sobel_score = face_processor.calculate_sobel_sharpness(aligned_face)
                            min_sobel = config.getfloat('Quality_Filters', 'min_sobel_sharpness', fallback=40.0)
                            if sobel_score < min_sobel:
                                sobel_ok = False
                                status_text = f"Motion Blur ({sobel_score:.0f})"
                                status_color = (0, 0, 255)
                    except Exception:
                        sobel_ok = True

                    if not sobel_ok:
                        if track_id in tracked_faces:
                            tracked_faces[track_id]['ema_embedding'] = None
                            tracked_faces[track_id]['valid_count'] = 0
                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        continue

                    # === FILTER 6: Brightness & Contrast validation ===
                    is_valid_face, validation_reason = FaceValidator.comprehensive_validation(aligned_face)

                    if not is_valid_face:
                        status_text = f"REJECT: {validation_reason}"
                        status_color = (0, 0, 255)
                        logger.debug(f"Track {track_id} failed geometric validation: {validation_reason}")

                        if track_id in tracked_faces:
                            tracked_faces[track_id]['ema_embedding'] = None
                            tracked_faces[track_id]['valid_count'] = 0

                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, pose_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        continue

                    # === PASSED QUALITY CHECKS: stabilization count ===
                    if track_id not in tracked_faces:
                        tracked_faces[track_id] = {'ema_embedding': None, 'name': 'Unknown', 'valid_count': 0}

                    tracked_faces[track_id]['valid_count'] += 1

                    if tracked_faces[track_id]['valid_count'] < min_valid_frames:
                        status_text = f"Stabilizing ({tracked_faces[track_id]['valid_count']}/{min_valid_frames})"
                        status_color = (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(frame, status_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        continue

                    # === ALL CHECKS PASSED - Process for Recognition ===
                    preprocessed_face = cv2.resize(aligned_face, (112, 112))

                    # ArcFace embedding - ONLY save if this succeeds
                    current_embedding = face_processor.get_embedding(preprocessed_face)
                    if current_embedding is None:
                        status_text = "ArcFace Failed"
                        status_color = (0, 165, 255)
                    else:
                        # Optional: Save for debugging (ONLY after successful embedding)
                        if config.getboolean('Logging', 'debug_save_faces', fallback=False):
                            debug_path = config.get('Paths', 'debug_aligned_faces_path')
                            filename = os.path.join(debug_path, f"track_{track_id}_frame_{frame_num}_lap{aligned_blur:.0f}_sob{int(sobel_score) if 'sobel_score' in locals() else 0}.jpg")
                            cv2.imwrite(filename, preprocessed_face)

                        # Temporal EMA
                        if track_id not in tracked_faces:
                            tracked_faces[track_id] = {'ema_embedding': None, 'name': 'Unknown', 'valid_count': 0}

                        prev_ema = tracked_faces[track_id]['ema_embedding']
                        smoothed_embedding = face_processor.apply_ema(current_embedding, prev_ema, ema_alpha)
                        tracked_faces[track_id]['ema_embedding'] = smoothed_embedding

                        # Match
                        name, similarity = att_manager.find_match(smoothed_embedding)

                        logger.debug(f"Track {track_id}: {name} (sim={similarity:.3f})")

                        if name != "Unknown":
                            tracked_faces[track_id]['name'] = name
                            att_manager.mark_attendance(name)
                            status_text = f"{name} ({similarity:.2f})"
                            status_color = (0, 255, 0)
                        else:
                            att_manager.handle_unknown(face_crop, track_id)
                            status_text = f"Unknown ({similarity:.2f})"
                            status_color = (0, 255, 255)

                    # === Visualization ===
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)

                    (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), status_color, -1)

                    cv2.putText(frame, status_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if pose_text:
                        cv2.putText(frame, pose_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Clean up old tracks
            for track_id in list(tracked_faces.keys()):
                if track_id not in current_track_ids:
                    del tracked_faces[track_id]

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Application shutting down.")


if __name__ == '__main__':
    main()
