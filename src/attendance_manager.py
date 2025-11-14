import os
import csv
import cv2
import logging
import time
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AttendanceManager:
    def __init__(self, config, reference_db):
        self.config = config
        self.attendance_file = config.get('Paths', 'attendance_file')
        self.unknown_path = config.get('Paths', 'unknown_faces_path')
        self.threshold = config.getfloat('Model_Settings', 'arcface_threshold')
        self.mark_cooldown = config.getfloat('Performance', 'attendance_mark_minutes') * 60
        self.unknown_cooldown = config.getfloat('Performance', 'unknown_capture_cooldown')
        
        self.reference_db = reference_db
        self.known_names = list(reference_db.keys())
        self.known_embeddings = np.array(list(reference_db.values()))
        
        self.last_attendance_time = {} # {name: timestamp}
        self.last_unknown_capture_time = {} # {tracker_id: timestamp}
        
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
            
        # Calculate cosine similarity
        sims = np.dot(query_embedding, self.known_embeddings.T)
        
        best_match_idx = np.argmax(sims)
        best_similarity = sims[best_match_idx]
        
        if best_similarity >= self.threshold:
            name = self.known_names[best_match_idx]
            return name, best_similarity
        else:
            return "Unknown", best_similarity

    def mark_attendance(self, name):
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