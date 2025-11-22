import os
import csv
import cv2
import logging
import time
import numpy as np
from datetime import datetime
import torch
import threading
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
import pymongo
import requests
import json

logger = logging.getLogger(__name__)

class AttendanceManager:
    def __init__(self, config, reference_db):
        self.config = config
        self.attendance_file = config.get('Paths', 'attendance_file')
        self.unknown_path = config.get('Paths', 'unknown_faces_path')
        self.threshold = config.getfloat('Model_Settings', 'arcface_threshold')
        self.mark_cooldown = config.getfloat('Performance', 'attendance_mark_minutes') * 60
        self.unknown_cooldown = config.getfloat('Performance', 'unknown_capture_cooldown')
        
        self.embedding_lock = threading.Lock()
        self.reference_db = reference_db
        self.known_names = list(reference_db.keys())
        self.known_embeddings = np.array(list(reference_db.values()))
        
        self.last_attendance_time = {} # {name: timestamp}
        self.last_unknown_capture_time = {} # {tracker_id: timestamp}

        
        self._db_lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
        """Creates the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'name'])
            logger.info(f"Created attendance file: {self.attendance_file}")

    def find_match(self, query_embedding):
        """Finds the best match in the reference database."""
        if not self.known_names:
            return "Unknown", 0.0
        
        if torch.cuda.is_available():
                q = torch.from_numpy(query_embedding).to('cuda:0').unsqueeze(0) # 1xd
                refs = torch.from_numpy(self.known_embeddings).to('cuda:0')    # nxd
                sims = torch.matmul(q, refs.T).squeeze(0).cpu().numpy()
        else:
                sims = np.dot(query_embedding, self.known_embeddings.T)
            
        # Calculate cosine similarity
        # sims = np.dot(query_embedding, self.known_embeddings.T)
        
        best_match_idx = np.argmax(sims)
        best_similarity = sims[best_match_idx]
        
        if best_similarity >= self.threshold:
            name = self.known_names[best_match_idx]
            return name, best_similarity
        else:
            return "Unknown", best_similarity

    # def mark_attendance(self, name):
        """Marks attendance for a known person if cooldown has passed."""
        current_time = time.time()
        last_time = self.last_attendance_time.get(name, 0)
        
        if (current_time - last_time) > self.mark_cooldown:
            self.last_attendance_time[name] = current_time
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name])
                
            logger.info(f"ATTENDANCE: {name} marked at {timestamp}")
            return True
        return False
   
    def mark_attendance(self, name, yolo_conf: float = None, deepface_dist: float = None):
        """
        Mark attendance for a studentId `name`.
        Workflow:
        1. Local cooldown check to avoid frequent DB/API calls.
        2. Look up student in students collection to get _id and firstName.
        3. Attempt to POST attendance to external Attendance API (config: AttendanceAPI.url).
            - If API returns 2xx -> success (CSV backup still written).
            - If API fails or times out -> FALLBACK to direct MongoDB insert.
        Returns True if attendance recorded (either via API or fallback DB), False otherwise.
        """
        current_time = time.time()
        last_time = self.last_attendance_time.get(name, 0)
        if (current_time - last_time) <= self.mark_cooldown:
            return False

        try:
            now_dt = datetime.now()
            mark_minutes = self.config.getint('Performance', 'attendance_mark_minutes', fallback=0)
            time_limit = now_dt - timedelta(minutes=mark_minutes)

            # MongoDB config (used for lookup + fallback)
            mongo_uri = self.config.get('MongoDB', 'uri')
            db_name = self.config.get('MongoDB', 'database_name')
            students_coll_name = self.config.get('MongoDB', 'students_collection_name', fallback='students')
            attendance_coll_name = self.config.get('MongoDB', 'attendance_collection_name', fallback='attendance')

            client = MongoClient(mongo_uri)
            db = client[db_name]
            students_coll = db[students_coll_name]
            attendance_coll = db[attendance_coll_name]

            # Find student doc by studentId (name)
            student_doc = students_coll.find_one({"studentId": name})
            if student_doc is None:
                logger.warning(f"Student '{name}' not found in '{students_coll_name}' collection — skipping attendance.")
                client.close()
                return False

            student_id_str = str(student_doc["_id"])
            first_name = student_doc.get("firstName") or student_doc.get("name") or "Unknown"

            # Prevent duplicate entries within mark_minutes window (DB check)
            recent = attendance_coll.find_one({
                "studentId": student_id_str,
                "entryTime": {"$gt": time_limit}
            })
            if recent is not None:
                client.close()
                self.last_attendance_time[name] = current_time
                return False

            # Build attendance payload/doc
            attendance_payload = {
                "student_id": student_id_str,                 # old API field name
                "yolo_confidence": float(yolo_conf) if yolo_conf is not None else None,
                "deepface_distance": float(deepface_dist) if deepface_dist is not None else None
            }

            # --- Try Attendance API ---
            api_url = None
            try:
                import requests
                api_url = self.config.get('AttendanceAPI', 'url', fallback=None)
                if api_url:
                    timeout = self.config.getint('AttendanceAPI', 'timeout_seconds', fallback=5)
                    api_key = self.config.get('AttendanceAPI', 'api_key', fallback=None)
                    headers = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                    resp = requests.post(api_url, json=attendance_payload, headers=headers, timeout=timeout)
                    if 200 <= resp.status_code < 300:
                        # success via API -> update local cooldown and CSV fallback
                        self.last_attendance_time[name] = current_time
                        timestamp = now_dt.strftime('%Y-%m-%d %H:%M:%S')
                        try:
                            with open(self.attendance_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([timestamp, name, first_name,
                                                f"{attendance_payload['yolo_confidence']}",
                                                f"{attendance_payload['deepface_distance']}"])
                        except Exception as e:
                            logger.warning(f"Failed to write attendance CSV backup after API success: {e}")

                        logger.info(f"ATTENDANCE API: {first_name} ({name}) recorded via API {api_url} at {timestamp}")
                        client.close()
                        return True
                    else:
                        logger.warning(f"Attendance API returned {resp.status_code}: {resp.text}. Falling back to DB.")
                else:
                    logger.debug("AttendanceAPI.url not configured — skipping API call and using DB fallback.")
            except Exception as e:
                logger.warning(f"Attendance API call failed (url={api_url}): {e}. Falling back to direct DB insert.")

            # --- FALLBACK: direct MongoDB insert ---
            try:
                attendance_doc_db = {
                    "studentId": student_id_str,
                    "firstName": first_name,
                    "entryTime": now_dt,
                    "detector_confidence": attendance_payload['detector_confidence'],
                    "face_similarity": attendance_payload['face_similarity'],
                    "source": "camera_pipeline_fallback"
                }
                attendance_coll.insert_one(attendance_doc_db)
            except Exception as e:
                logger.exception(f"Fallback DB insert failed for {name}: {e}")
                client.close()
                return False

            # update local state + CSV fallback
            self.last_attendance_time[name] = current_time
            timestamp = now_dt.strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(self.attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, name, first_name,
                                    f"{attendance_doc_db['detector_confidence']}",
                                    f"{attendance_doc_db['face_similarity']}"])
            except Exception as e:
                logger.warning(f"Failed to write attendance CSV backup after fallback DB insert: {e}")

            logger.info(f"ATTENDANCE DB (fallback): {first_name} ({name}) marked at {timestamp}")
            client.close()
            return True

        except Exception as e:
            logger.exception(f"Error while marking attendance for {name}: {e}")
            return False


    def handle_unknown(self, face_crop, tracker_id):
        """Saves an image of an unknown face if cooldown has passed."""
        current_time = time.time()
        last_time = self.last_unknown_capture_time.get(tracker_id, 0)
        
        if (current_time - last_time) > self.unknown_cooldown:
            self.last_unknown_capture_time[tracker_id] = current_time
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.unknown_path, f"unknown_{tracker_id}_{timestamp}.jpg")
            
            cv2.imwrite(filename, face_crop)
            logger.info(f"Saved unknown face to {filename}")


    def add_identity(self, student_id: str, embedding_np: np.ndarray, first_name: str = None, persist_to_db: bool = True):
        """
        Atomically append a new identity to known_names / known_embeddings.
        - student_id: the folder / studentId string (used as name key in DB and for matching)
        - embedding_np: 1-D float32 L2-normalized numpy array
        - first_name: optional, used for logs
        - persist_to_db: if True, write embedding to students collection (embedding as list)
        """
        if embedding_np is None:
            logger.warning("add_identity called with None embedding; skipping.")
            return False

        # ensure shape
        embedding_np = embedding_np.astype(np.float32).reshape(-1)

        # persist to MongoDB (best-effort)
        try:
            if persist_to_db:
                mongo_uri = self.config.get('MongoDB', 'uri')
                db_name = self.config.get('MongoDB', 'database_name')
                students_coll_name = self.config.get('MongoDB', 'students_collection_name', fallback='students')
                client = pymongo.MongoClient(mongo_uri)
                db = client[db_name]
                students_coll = db[students_coll_name]
                # update embedding field (store as plain list)
                students_coll.update_one({"studentId": student_id},
                                        {"$set": {"embedding": embedding_np.tolist()}},
                                        upsert=False)
                client.close()
        except Exception as e:
            logger.warning(f"Failed to persist embedding for {student_id}: {e}")

        # atomic in-memory append
        with self.embedding_lock:
            if student_id in self.known_names:
                logger.info(f"add_identity: {student_id} already exists — replacing embedding.")
                idx = self.known_names.index(student_id)
                self.known_embeddings[idx] = embedding_np
            else:
                self.known_names.append(student_id)
                if self.known_embeddings.size == 0:
                    self.known_embeddings = embedding_np[np.newaxis, :]
                else:
                    self.known_embeddings = np.vstack([self.known_embeddings, embedding_np[np.newaxis, :]])
        logger.info(f"Added identity in-memory: {student_id} ({first_name or ''}), total={len(self.known_names)}")
        return True




