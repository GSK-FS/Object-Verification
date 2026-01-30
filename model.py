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
