# utils.py
import base64
import cv2
import numpy as np
import tempfile
import os


def decode_base64_image(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def read_uploaded_file(file):
    content = file.file.read()
    np_arr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def save_uploaded_video(file):
    """
    Saves uploaded video to a temp file and returns its path
    """
    suffix = os.path.splitext(file.filename)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name


def extract_frames(video_path, frame_skip=5):
    """
    Extract frames from video every N frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            frames.append(frame)

        idx += 1

    cap.release()
    return frames
