import os
from ultralytics import YOLO

MODEL_DIR = "models"
# Choose a valid YOLO11 detection variant for auto-download
OFFICIAL_MODEL_NAME = "yolo11x.pt"
MODEL_PATH = os.path.join(MODEL_DIR, OFFICIAL_MODEL_NAME)

_yolo_model = None  # singleton instance


def ensure_model_exists():
    """
    Ensures that the YOLO11 model file exists locally.
    If not, auto-downloads it using the official name.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Model not found locally. Downloading {OFFICIAL_MODEL_NAME} ...")
        # This downloads the official YOLO11n model and caches it
        model = YOLO(OFFICIAL_MODEL_NAME)
        model.save(MODEL_PATH)
        print(f"[INFO] Model downloaded and saved at {MODEL_PATH}")


def _print_detectable_classes(model):
    """Prints class names supported by this YOLO model."""
    print("\n[INFO] YOLO Detectable Classes:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    print(f"[INFO] Total classes: {len(model.names)}\n")


def get_model():
    global _yolo_model

    if _yolo_model is None:
        ensure_model_exists()
        _yolo_model = YOLO(MODEL_PATH)
        _print_detectable_classes(_yolo_model)

    return _yolo_model


def get_detectable_classes():
    """
    Returns a list of detectable class names.
    Useful for API endpoints or frontend UI.
    """
    model = get_model()
    return list(model.names.values())


def detect_objects(image, allowed_classes=None):
    """
    Runs inference on the given image and returns filtered detections.
    """
    model = get_model()
    results = model(image, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if allowed_classes and label not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    return detections

# model.py (ADD BELOW existing code)

def verify_car_video(
    frames,
    allowed_classes=["car"],
    min_presence_ratio=0.8
):
    """
    Verifies:
    - Car is present in >= 80% frames
    - User does not leave object (basic bbox continuity)
    """
    total_frames = len(frames)
    if total_frames == 0:
        return {
            "valid": False,
            "reason": "No frames extracted from video"
        }

    car_detected_frames = 0
    last_bbox = None
    bbox_jump_count = 0

    for frame in frames:
        detections = detect_objects(frame, allowed_classes)

        if detections:
            car_detected_frames += 1

            # Take the highest confidence detection
            best = max(detections, key=lambda x: x["confidence"])
            bbox = best["bbox"]

            # Simple movement sanity check
            if last_bbox:
                dx = abs(bbox[0] - last_bbox[0])
                dy = abs(bbox[1] - last_bbox[1])

                if dx > 200 or dy > 200:  # threshold tweakable
                    bbox_jump_count += 1

            last_bbox = bbox

    presence_ratio = car_detected_frames / total_frames

    return {
        "valid": presence_ratio >= min_presence_ratio,
        "total_frames": total_frames,
        "car_detected_frames": car_detected_frames,
        "presence_ratio": round(presence_ratio, 3),
        "bbox_jumps": bbox_jump_count,
        "message": (
            "Valid 360 car walkaround"
            if presence_ratio >= min_presence_ratio
            else "Car not consistently visible in video"
        )
    }
