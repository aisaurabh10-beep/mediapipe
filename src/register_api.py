# register_api.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os, glob, logging
import numpy as np
import cv2
import pymongo
from typing import Optional

logger = logging.getLogger("register_api")
app = FastAPI(title="Register API")

# these will be injected by main at startup
PROCESSOR = None   # FaceProcessor instance
ATT_MANAGER = None # AttendanceManager instance
CONFIG = None

class RegisterResponse(BaseModel):
    status: str
    studentId: str
    message: Optional[str] = None

def init_api(processor, att_manager, config):
    global PROCESSOR, ATT_MANAGER, CONFIG
    PROCESSOR = processor
    ATT_MANAGER = att_manager
    CONFIG = config
    logger.info("Register API initialized with processor & att_manager")

def _compute_avg_embedding_from_folder(student_id: str):
    dataset_base = CONFIG.get('Paths', 'dataset_path')
    person_path = os.path.join(dataset_base, student_id)
    if not os.path.isdir(person_path):
        raise FileNotFoundError(f"Folder not found: {person_path}")

    image_files = sorted([p for p in glob.glob(os.path.join(person_path, "*")) if p.lower().endswith(('.jpg','.jpeg','.png'))])
    if not image_files:
        raise FileNotFoundError("No images found in folder.")

    embeddings = []
    for p in image_files:
        img = cv2.imread(p)
        if img is None:
            logger.warning(f"Can't read {p}")
            continue
        emb = PROCESSOR.get_embedding(img)  # returns normalized np.array or None
        if emb is None:
            logger.warning(f"Embedding failed for {p}")
            continue
        embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No valid embeddings for any image in folder.")

    avg = np.mean(embeddings, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg.astype(np.float32), len(embeddings)

def _persist_embedding_to_db(student_id: str, embedding_np: np.ndarray):
    try:
        mongo_uri = CONFIG.get('MongoDB', 'uri')
        db_name = CONFIG.get('MongoDB', 'database_name')
        students_coll_name = CONFIG.get('MongoDB', 'students_collection_name', fallback='students')
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        students = db[students_coll_name]
        students.update_one({"studentId": student_id},
                            {"$set": {"embedding": embedding_np.tolist()}},
                            upsert=False)
        client.close()
    except Exception as e:
        logger.warning(f"Persist embedding failed: {e}")
        raise

def process_and_add(student_id: str):
    logger.info(f"Register: processing {student_id}")
    # compute average embedding
    avg_emb, count = _compute_avg_embedding_from_folder(student_id)
    # persist embedding to students collection
    _persist_embedding_to_db(student_id, avg_emb)
    # fetch firstName for logging
    try:
        mongo_uri = CONFIG.get('MongoDB', 'uri')
        db_name = CONFIG.get('MongoDB', 'database_name')
        students_coll_name = CONFIG.get('MongoDB', 'students_collection_name', fallback='students')
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        students = db[students_coll_name]
        doc = students.find_one({"studentId": student_id}, {"firstName": 1})
        first = doc.get("firstName") if doc else None
        client.close()
    except Exception:
        first = None

    # add to in-memory DB used by attendance manager
    ATT_MANAGER.add_identity(student_id, avg_emb, first_name=first, persist_to_db=False)
    logger.info(f"Register: done for {student_id} (images_used={count})")

# @app.post("/register/{student_id}", response_model=RegisterResponse, status_code=202)
# def register(student_id: str, background_tasks: BackgroundTasks):
#     if PROCESSOR is None or ATT_MANAGER is None or CONFIG is None:
#         raise HTTPException(status_code=503, detail="API not initialized")
#     # quick validation of folder
#     base = CONFIG.get('Paths', 'dataset_path')
#     folder = os.path.join(base, student_id)
#     if not os.path.isdir(folder):
#         raise HTTPException(status_code=404, detail="Student folder not found")
#     background_tasks.add_task(process_and_add, student_id)
#     return {"status": "accepted", "studentId": student_id, "message": "processing in background"}


@app.post("/register/{student_id}", response_model=RegisterResponse, status_code=200)
def register(student_id: str):
    if PROCESSOR is None or ATT_MANAGER is None or CONFIG is None:
        raise HTTPException(status_code=503, detail="API not initialized")

    # check folder exists
    base = CONFIG.get('Paths', 'dataset_path')
    folder = os.path.join(base, student_id)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail="Student folder not found")

    try:
        # compute embedding directly (NO background task)
        avg_emb, count = _compute_avg_embedding_from_folder(student_id)

        # save embedding into MongoDB
        _persist_embedding_to_db(student_id, avg_emb)

        # fetch firstName for logs
        mongo_uri = CONFIG.get('MongoDB', 'uri')
        db_name = CONFIG.get('MongoDB', 'database_name')
        students_coll = CONFIG.get('MongoDB', 'students_collection_name', fallback='students')
        import pymongo
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        doc = db[students_coll].find_one({"studentId": student_id}, {"firstName": 1})
        first = doc.get("firstName") if doc else None
        client.close()

        # add to in-memory embeddings
        ATT_MANAGER.add_identity(student_id, avg_emb, first_name=first, persist_to_db=False)

        return {
            "status": "success",
            "studentId": student_id,
            "message": f"Registered successfully. {count} images processed.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

