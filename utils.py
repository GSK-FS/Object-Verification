import base64
import cv2
import numpy as np

def decode_base64_image(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def read_uploaded_file(file):
    content = file.file.read()
    np_arr = np.frombuffer(content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
