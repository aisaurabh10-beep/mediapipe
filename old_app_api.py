# import asyncio
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# from bson import ObjectId
# from pydantic import BaseModel, Field
# import os
# from typing import List, Optional

# # --- Machine Learning Imports ---
# from deepface import DeepFace
# import tensorflow as tf
# # --- Add the standard PyMongo client for background tasks ---
# import pymongo


# # --- Configuration ---
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# # MONGO_DB_NAME = "face_detection"
# # MONGO_COLLECTION_NAME = "students"

# MONGO_DB_NAME = "attendance_poc"
# MONGO_COLLECTION_NAME = "students"
# # DATASET_PATH = os.getenv("DATASET_PATH", "dataset")
# DATASET_PATH = os.getenv("DATASET_PATH", "C:/vaibhav/face_detection/backend/uploads/students")





# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="Face Recognition Embedding Sync API",
#     description="API to trigger face embedding generation for existing students.",
#     version="1.3.0"
# )

# # --- CORS Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Database Connection ---
# # This async client is for the main API endpoints
# client = AsyncIOMotorClient(MONGO_URI)
# db = client[MONGO_DB_NAME]
# collection = db[MONGO_COLLECTION_NAME]

# # --- Background Task Logic ---
# def generate_and_store_embedding(student_id: str):
#     """
#     Finds images for a student, generates face embedding, and updates MongoDB.
#     This function is designed to be run in the background.
#     """
#     print(f"BACKGROUND_TASK: Starting embedding generation for student_id: {student_id}")
    
#     person_path = os.path.join(DATASET_PATH, student_id)
    
#     if not os.path.isdir(person_path):
#         print(f"BACKGROUND_TASK_ERROR: Directory not found for student_id: {student_id} at {person_path}")
#         return

#     image_path_to_process = None
#     for img_file in os.listdir(person_path):
#         if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path_to_process = os.path.join(person_path, img_file)
#             break

#     if not image_path_to_process:
#         print(f"BACKGROUND_TASK_WARNING: No images found for student_id: {student_id}")
#         return

#     try:
#         print(f"Processing image: {image_path_to_process}")
#         embedding_obj = DeepFace.represent(
#             img_path=image_path_to_process, 
#             model_name="ArcFace", 
#             enforce_detection=False
#         )
        
#         embedding = embedding_obj[0]["embedding"]
        
#         # --- FIX: Use the synchronous pymongo client for background tasks ---
#         # This is because background tasks run in a separate thread without an asyncio event loop.
#         sync_client = pymongo.MongoClient(MONGO_URI)
#         sync_db = sync_client[MONGO_DB_NAME]
#         sync_collection = sync_db[MONGO_COLLECTION_NAME]
        
#         # # Update the student's document in MongoDB with the new embedding
#         # sync_collection.update_one(
#         #     {"studentId": student_id},
#         #     {"$set": {"embedding": embedding}}
#         # )


#         # Check if embedding field is missing OR empty array
#         student_doc = sync_collection.find_one(
#             {"studentId": student_id},
#             {"embedding": 1}
#         )

#         if (
#             student_doc is None or
#             "embedding" not in student_doc or
#             not student_doc["embedding"]  # covers [] or None
#         ):
#             # Update or add the embedding
#             sync_collection.update_one(
#                 {"studentId": student_id},
#                 {"$set": {"embedding": embedding}}
#             )
        
#         print(f"BACKGROUND_TASK_SUCCESS: Successfully updated embedding for {student_id}")

#         success = f"True"
#         return JSONResponse(status_code=202, content={ "success": success})
#         sync_client.close()

#     except Exception as e:
#         print(f"BACKGROUND_TASK_ERROR: Failed to process or update embedding for {student_id}. Error: {e}")


# # --- FastAPI Lifespan Events ---
# @app.on_event("startup")
# async def startup_event():
#     print("Application startup...")


# # --- API Endpoints ---
# @app.post("/sync-embeddings", summary="Generate embeddings for all students who are missing them")
# async def sync_embeddings(background_tasks: BackgroundTasks):
#     """
#     Finds all students in the database where the 'embedding' field does not exist
#     and starts a background task to generate it for each one.
#     """
#     students_to_process = 0
#     cursor = collection.find({"embedding": {"$exists": False}})
    
#     async for student_doc in cursor:
#         student_id = student_doc.get("studentId")
#         if student_id:
#             background_tasks.add_task(generate_and_store_embedding, student_id)
#             students_to_process += 1
            
#     message = f"Started embedding generation for {students_to_process} student(s)."
#     success = f"True"

#     print(message)
#     print(success)
    

#     return JSONResponse(status_code=202, content={ "success": success})

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pydantic import BaseModel, Field
import os
from typing import List, Optional, Tuple

# --- Machine Learning Imports ---
from deepface import DeepFace
import tensorflow as tf
# --- Add the standard PyMongo client for background tasks ---
import pymongo


# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "attendance_poc"
MONGO_COLLECTION_NAME = "students"
DATASET_PATH = os.getenv("DATASET_PATH", "C:/vaibhav/face_detection/backend/uploads/students")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Face Recognition Embedding Sync API",
    description="An API to trigger and wait for face embedding generation to complete.",
    version="1.6.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]

# --- Processing Logic ---
def generate_and_store_embedding(student_id: str) -> Tuple[str, str]:
    """
    Finds images for a student, generates face embedding, and updates MongoDB.
    Returns a tuple of (status, message) for summarizing results.
    """
    print(f"TASK_STARTED: Starting embedding generation for student_id: {student_id}")
    
    # Use a synchronous client for thread safety
    sync_client = pymongo.MongoClient(MONGO_URI)
    sync_db = sync_client[MONGO_DB_NAME]
    sync_collection = sync_db[MONGO_COLLECTION_NAME]

    try:
        person_path = os.path.join(DATASET_PATH, student_id)
        if not os.path.isdir(person_path):
            raise FileNotFoundError(f"Directory not found for student_id: {student_id} at {person_path}")

        image_path_to_process = None
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path_to_process = os.path.join(person_path, img_file)
                break

        if not image_path_to_process:
            raise FileNotFoundError(f"No images found for student_id: {student_id}")

        print(f"Processing image: {image_path_to_process}")
        embedding_obj = DeepFace.represent(
            img_path=image_path_to_process, 
            model_name="ArcFace", 
            enforce_detection=False
        )
        embedding = embedding_obj[0]["embedding"]
        
        # On success, update the document with the embedding
        sync_collection.update_one(
            {"studentId": student_id},
            {"$set": {"embedding": embedding}}
        )
        success_message = f"Successfully updated embedding for {student_id}"
        print(f"TASK_SUCCESS: {success_message}")
        return ("success", success_message)

    except Exception as e:
        # On failure, return the error details
        error_message = f"Failed to process embedding for {student_id}. Error: {e}"
        print(f"TASK_ERROR: {error_message}")
        return ("error", error_message)
    finally:
        sync_client.close()


# --- API Endpoints ---
@app.post("/sync-embeddings", summary="Generate embeddings and wait for completion")
async def sync_embeddings():
    """
    Finds all students where the 'embedding' field does not exist,
    generates the embedding for each one, and waits for all tasks to complete
    before sending a final response. This can take a long time.
    """
    students_to_process_ids = []
    query = {"embedding": {"$exists": False}}
    cursor = collection.find(query)
    
    async for student_doc in cursor:
        student_id = student_doc.get("studentId")
        if student_id:
            students_to_process_ids.append(student_id)
            
    if not students_to_process_ids:
        return JSONResponse(
            status_code=200,
            content={"message": "All student embeddings are already up-to-date."}
        )

    # Create a list of tasks to run in parallel threads
    tasks = [asyncio.to_thread(generate_and_store_embedding, student_id) for student_id in students_to_process_ids]
    
    # Run all tasks concurrently and wait for them to finish
    results = await asyncio.gather(*tasks)
    
    # Process the results into success and error lists
    success_details = [res[1] for res in results if res[0] == "success"]
    error_details = [res[1] for res in results if res[0] == "error"]

    msg = ['True' for res in results if res[0] == "success"]
    
    response_content = {
        "message": "Embedding synchronization process completed.",
        "success" : msg[0],
        "processed_count": len(results),
        "success_count": len(success_details),
        "error_count": len(error_details),
        "details": {
            "successes": success_details,
            "errors": error_details,
        }
    }
    
    # Use 200 OK if all succeed, or 207 Multi-Status if there are mixed results
    status_code = 200 if not error_details else 207
    
    return JSONResponse(status_code=status_code, content=response_content)