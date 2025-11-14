import os
import glob
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

logger = logging.getLogger(__name__)

def load_yolo_model(config):
    """Loads the YOLOv8-face model."""
    model_path = config.get('Paths', 'yolo_model_path')
    if not os.path.exists(model_path):
        logger.error(f"YOLO model not found at {model_path}")
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
        
    logger.info(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    logger.info("YOLO model loaded.")
    return model

def build_reference_database(config, face_processor):
    """
    Scans the dataset_path and builds a database of reference embeddings.
    Structure:
        dataset/
            person_A/
                1.jpg
                2.jpg
            person_B/
                1.jpg
    """
    logger.info("Building reference face database...")
    dataset_path = config.get('Paths', 'dataset_path')
    reference_db = {}
    
    person_folders = [f for f in os.scandir(dataset_path) if f.is_dir()]
    
    for person_dir in person_folders:
        person_name = person_dir.name
        embeddings = []
        image_files = glob.glob(os.path.join(person_dir.path, '*.[jp][pn]g'))
        
        if not image_files:
            logger.warning(f"No images found for {person_name}, skipping.")
            continue
            
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not read {img_path}")
                    continue
                
                # Use DeepFace's built-in alignment for database building
                # This is more reliable than our live pipeline
                embedding_obj = DeepFace.represent(
                    img_path,
                    model_name=face_processor.arcface_model,
                    enforce_detection=True,
                    detector_backend='retinaface' # Use a good detector
                )
                
                embeddings.append(embedding_obj[0]['embedding'])
                
            except Exception as e:
                logger.warning(f"Failed to process {img_path} for {person_name}: {e}")
        
        if embeddings:
            # Average all embeddings for this person to get a single, robust vector
            avg_embedding = np.mean(embeddings, axis=0)
            l2_norm = np.linalg.norm(avg_embedding)
            reference_db[person_name] = avg_embedding / l2_norm
            logger.info(f"Processed {person_name} with {len(embeddings)} images.")
            
    logger.info(f"Reference database built. {len(reference_db)} people found.")
    return reference_db