import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640
DEVICE = "cpu"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =====================
# GLOBAL SHARED FRAME
# =====================
latest_frame = None
lock = threading.Lock()

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")


# =====================
# CAMERA THREAD
# =====================
def camera_loop():
    global latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame


# =====================
# YOLO THREAD
# =====================
def inference_loop():
    global latest_frame

    while True:

        if latest_frame is None:
            continue

        with lock:
            frame = latest_frame.copy()

        # YOLO inference
        results = model(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]

        annotated = results.plot()

        # FIX: ensure correct format
        annotated = cv2.resize(annotated, (FRAME_WIDTH, FRAME_HEIGHT))
        annotated = np.ascontiguousarray(annotated)

        with lock:
            latest_frame = annotated


# =====================
# START THREADS
# =====================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()

print("[INFO] YOLO inference running (frame producer)")

# keep alive
while True:
    time.sleep(1)