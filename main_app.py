from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import json
import datetime

from model import detect_objects, get_model
from utils import decode_base64_image, read_uploaded_file

# main.py (video imports)
from utils import save_uploaded_video, extract_frames
from model import verify_car_video
import os


db = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize in-memory database
    print("[INFO] Starting up and Preloading model...", datetime.datetime.now())
    get_model()
    print("[INFO] YOLO model ready")
    db["start_time"] = datetime.datetime.now()
    yield
    # Shutdown: Cleanup if necessary
    print("[INFO] Shutting down...", datetime.datetime.now())



app = FastAPI(title="YOLOv11 Fast Detection API", lifespan=lifespan)


# @app.on_event("startup")
# def preload_model():
#     # Load model once when API starts
#     print("[INFO] Preloading YOLO model...")
#     get_model()
#     print("[INFO] YOLO model ready")

@app.get("/api")
async def read_root():
    return {"message": "App is running", "started_at": db.get("start_time")}


@app.post("/api/detect")
async def detect(
    image_base64: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    classes: Optional[str] = Form(None)
):
    if not image_base64 and not file:
        return {"error": "No image provided, please provide base64 image or a file."}

    DEFAULT_CLASSES = ["car"]

    # Parse classes if provided
    try:
        allowed_classes = json.loads(classes) if classes else DEFAULT_CLASSES
        if not isinstance(allowed_classes, list):
            raise ValueError()
    except Exception:
        return {"error": "Invalid 'classes' value. Must be a JSON array, e.g., [\"car\",\"bus\"]"}

    # Decode image
    image = decode_base64_image(image_base64) if image_base64 else read_uploaded_file(file)

    # Detect objects
    detections = detect_objects(image, allowed_classes)

    # If no object found, return useful message
    if not detections:
        return {
            "success": True,
            "count": 0,
            "detections": [],
            "message": f"No objects detected such as: {allowed_classes}"
        }

    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
        "message": f"Valid image."
    }


@app.post("/api/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    frame_skip: Optional[int] = Form(5)):
    if not file:
        return {"error": "No video file provided"}

    video_path = save_uploaded_video(file)

    try:
        frames = extract_frames(video_path, frame_skip=frame_skip)
        result = verify_car_video(frames)

        return {
            "success": True,
            "video_validation": result
        }

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
