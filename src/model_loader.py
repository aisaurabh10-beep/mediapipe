import os
import glob
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import torch, logging

logger = logging.getLogger(__name__)

def load_yolo_model(config):
    model_path = config.get('Paths', 'yolo_model_path')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    logger.info(f"Loading YOLO model from {model_path} ...")
    model = YOLO(model_path)
    # prefer device 0 if cuda available
    if torch.cuda.is_available():
        try:
            model.to("cuda:0")
            logger.info("Moved YOLO model to cuda:0")
        except Exception:
            logger.warning("Failed to call model.to('cuda:0') - Ultralytics may manage device internally")
    else:
        logger.info("CUDA not available, using CPU")
    return model

def build_reference_database(config, face_processor):
    """
    Scans dataset_path and builds a database of reference embeddings using
    face_processor.get_embedding(...) so runtime and DB use identical embeddings.
    Structure:
        dataset/
            person_A/
                1.jpg
                2.jpg
    Returns: dict {person_name: avg_embedding (L2-normalized numpy array)}
    """
    import os, glob, logging, cv2, numpy as np
    logger = logging.getLogger(__name__)
    dataset_path = config.get('Paths', 'dataset_path')
    reference_db = {}

    person_folders = [f for f in os.scandir(dataset_path) if f.is_dir()]

    logger.info("Building reference face database (using face_processor.get_embedding)...")
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

                # Use face_processor.get_embedding (this ensures same model/preproc as runtime)
                emb = face_processor.get_embedding(img)
                if emb is None:
                    logger.warning(f"Failed to get embedding for {img_path} (skipping)")
                    continue
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to process {img_path} for {person_name}: {e}")

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            l2_norm = np.linalg.norm(avg_embedding) + 1e-8
            reference_db[person_name] = (avg_embedding / l2_norm).astype(np.float32)
            logger.info(f"Processed {person_name} with {len(embeddings)} images.")
        else:
            logger.warning(f"No valid embeddings for {person_name}, skipping.")

    logger.info(f"Reference database built. {len(reference_db)} people found.")
    return reference_db
