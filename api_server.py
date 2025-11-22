from fastapi import FastAPI
from pydantic import BaseModel
import datetime

app = FastAPI()

class AttendanceRecord(BaseModel):
    student_id: str
    yolo_confidence: float
    deepface_distance: float

@app.post("/mark_attendance")
def mark_attendance(record: AttendanceRecord):
    """A simple endpoint to receive and print attendance data."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] SUCCESS: Received Attendance for Student ID: {record.student_id}")
    # In a real application, you would save this record to your main database.
    return {"status": "success", "message": "Attendance marked", "data": record.dict()}