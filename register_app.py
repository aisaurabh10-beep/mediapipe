#!/usr/bin/env python3
# register_app.py
# Save full face crop (raw) to dataset_path/<person_name>/ as registration image.
# Press 'c' to capture (when READY), 'q' to quit.

import cv2
import os
import time
import argparse
from datetime import datetime
import numpy as np

from src.config_loader import load_config
from src.model_loader import load_yolo_model
from src.face_processor import FaceProcessor
from src.mediapipe_client import MediaPipeClient
from src.face_validator import FaceValidator

def ensure_person_folder(dataset_path, person_name):
    person_dir = os.path.join(dataset_path, person_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir

def draw_overlay(frame, box, metrics, ready):
    x1, y1, x2, y2 = box
    color = (0, 200, 0) if ready else (0, 0, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    txt_lines = [
        f"Yaw:{metrics.get('yaw',0):.1f} Pitch:{metrics.get('pitch',0):.1f} Roll:{metrics.get('roll',0):.1f}",
        f"Blur:{metrics.get('aligned_blur',0):.1f} Sobel:{metrics.get('sobel',0):.1f} Aspect:{metrics.get('aspect',0):.2f}",
        f"Valid:{metrics.get('is_valid', False)}  READY:{'YES' if ready else 'NO'}"
    ]
    x = x1
    y = max(0, y1 - 60)
    for i, line in enumerate(txt_lines):
        cv2.putText(frame, line, (x+4, y + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # big READY/ADJUST
    label = "READY" if ready else "ADJUST"
    lc = (0,200,0) if ready else (0,0,200)
    cv2.putText(frame, label, (x2-110, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lc, 2, cv2.LINE_AA)
    return frame

def expand_box(x1,y1,x2,y2, w, h, pad_ratio=0.25):
    # expand by pad_ratio fraction of box size
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w-1, x2 + pad_w)
    ny2 = min(h-1, y2 + pad_h)
    return nx1, ny1, nx2, ny2

def get_largest_box(boxes):
    if len(boxes) == 0:
        return None
    areas = [(int((x2-x1)*(y2-y1)), (x1,y1,x2,y2)) for (x1,y1,x2,y2) in boxes]
    areas.sort(key=lambda x: x[0], reverse=True)
    return areas[0][1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name','-n', default=None)
    p.add_argument('--cam','-c', type=int, default=0)
    args = p.parse_args()

    config = load_config('config.ini')
    person = args.name or input("Enter name for registration: ").strip()
    dataset_path = config.get('Paths','dataset_path', fallback='dataset')
    debug_save_flag = config.getboolean('Logging','debug_save_faces', fallback=False)

    # Load helpers (YOLO optional but preferred for robust face bbox)
    try:
        yolo_model = load_yolo_model(config)
    except Exception:
        yolo_model = None

    mp_client = MediaPipeClient(config)
    face_processor = FaceProcessor(config)
    person_dir = ensure_person_folder(dataset_path, person)

    # thresholds (mirror production)
    blur_thresh = config.getfloat('Quality_Filters','blur_threshold', fallback=50.0)
    yaw_thresh = config.getfloat('Quality_Filters','yaw_threshold', fallback=15.0)
    pitch_thresh = config.getfloat('Quality_Filters','pitch_threshold', fallback=15.0)
    roll_thresh = config.getfloat('Quality_Filters','roll_threshold', fallback=15.0)
    min_sobel = config.getfloat('Quality_Filters','min_sobel_sharpness', fallback=25.0)
    embedding_every = config.getint('Performance','embedding_every_frames', fallback=2)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Camera not available")
        return

    saved = 0
    print("Preview: Press 'c' to capture (saves full padded face crop), 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        h, w = frame.shape[:2]

        # 1) Detect face bbox via YOLO if available else use center heuristics
        boxes = []
        if yolo_model is not None:
            try:
                # use single-frame predict for detection
                res = yolo_model.predict(frame, conf=config.getfloat('Model_Settings','yolo_conf_threshold', fallback=0.45), device=0 if hasattr(yolo_model, "model") else None)
                # ultralytics returns list results, each has boxes.xyxy
                if hasattr(res[0], 'boxes') and res[0].boxes is not None:
                    xyxy = res[0].boxes.xyxy.cpu().numpy()
                    for b in xyxy:
                        x1,y1,x2,y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        boxes.append((x1,y1,x2,y2))
            except Exception:
                boxes = []

        if boxes:
            box = get_largest_box(boxes)
            x1,y1,x2,y2 = expand_box(*box, w=w, h=h, pad_ratio=0.25)
        else:
            # fallback center region
            cx1 = int(w*0.25); cy1 = int(h*0.15); cx2 = int(w*0.75); cy2 = int(h*0.85)
            x1,y1,x2,y2 = cx1, cy1, cx2, cy2

        face_crop = frame[y1:y2, x1:x2].copy()
        aligned_face, yaw, pitch, roll, aspect = mp_client.get_alignment_and_pose(face_crop)
        metrics = {'yaw':0., 'pitch':0., 'roll':0., 'aligned_blur':0., 'sobel':0., 'aspect':0., 'is_valid':False}

        if aligned_face is not None:
            metrics['yaw'] = float(yaw); metrics['pitch'] = float(pitch); metrics['roll'] = float(roll)
            try:
                metrics['aligned_blur'] = float(face_processor.calculate_blur_laplacian(aligned_face))
            except Exception:
                metrics['aligned_blur'] = 0.0
            try:
                metrics['sobel'] = float(face_processor.calculate_sobel_sharpness(aligned_face))
            except Exception:
                metrics['sobel'] = 0.0
            try:
                ah, aw = aligned_face.shape[:2]
                metrics['aspect'] = float(aw / (ah + 1e-8))
            except Exception:
                metrics['aspect'] = 0.0
            try:
                is_valid, reason = FaceValidator.comprehensive_validation(aligned_face)
                metrics['is_valid'] = bool(is_valid)
                metrics['validation_reason'] = reason
            except Exception as e:
                metrics['is_valid'] = False
                metrics['validation_reason'] = str(e)

        # decide readiness
        ready = False
        if aligned_face is not None:
            pose_ok = (abs(metrics['yaw']) <= yaw_thresh and abs(metrics['pitch']) <= pitch_thresh and abs(metrics['roll']) <= roll_thresh)
            blur_ok = metrics['aligned_blur'] >= blur_thresh
            sobel_ok = metrics['sobel'] >= min_sobel
            valid_ok = metrics['is_valid']
            ready = pose_ok and blur_ok and sobel_ok and valid_ok

        # display overlay-copy (do not modify original)
        disp = frame.copy()
        disp = draw_overlay(disp, (x1,y1,x2,y2), metrics, ready)
        cv2.imshow("Register - Preview", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('c'):
            if face_crop is None or face_crop.size == 0:
                print("No face/crop available to save.")
                continue
            # Save full raw padded crop (this is the canonical registration image)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_name = f"raw_{person}_{ts}.jpg"
            raw_path = os.path.join(person_dir, raw_name)
            try:
                cv2.imwrite(raw_path, face_crop)
                # also save aligned preview (112x112) for quick QA (optional)
                if aligned_face is not None:
                    try:
                        aligned_preview = cv2.resize(aligned_face, (112,112))
                        aligned_name = f"aligned_{person}_{ts}.jpg"
                        aligned_path = os.path.join(person_dir, aligned_name)
                        # convert RGB->BGR if mediapipe returns RGB (heuristic)
                        save_img = aligned_preview
                        if aligned_preview.shape[2] == 3 and aligned_preview[...,0].mean() > aligned_preview[...,2].mean():
                            save_img = cv2.cvtColor(aligned_preview, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(aligned_path, save_img)
                    except Exception:
                        pass
                saved += 1
                print(f"[SAVED] raw: {raw_path}  ready={ready}")
            except Exception as e:
                print(f"Failed to save raw crop: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} images to {person_dir}")

if __name__ == "__main__":
    main()
