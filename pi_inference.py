import cv2
import time
import threading
import numpy as np
import subprocess
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

CAMERA = "/dev/video1"
RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# GLOBAL FRAME
# =====================
latest_frame = None
lock = threading.Lock()

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA (ONLY ONE OWNER)
# =====================
cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Camera opened")

# =====================
# FFmpeg (TAKES RAW YOLO FRAMES)
# =====================
ffmpeg = subprocess.Popen([
    "ffmpeg",
    "-loglevel", "error",

    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
    "-r", str(FPS),
    "-i", "-",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL
], stdin=subprocess.PIPE)

print("[INFO] RTSP streaming started:", RTSP_URL)

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
# YOLO + STREAM THREAD
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

        # resize for consistency
        annotated = cv2.resize(annotated, (FRAME_WIDTH, FRAME_HEIGHT))

        # send to FFmpeg
        try:
            ffmpeg.stdin.write(annotated.tobytes())
        except Exception as e:
            print("[FFMPEG ERROR]", e)
            break

# =====================
# START
# =====================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()

print("[INFO] YOLO + RTSP pipeline running")

while True:
    time.sleep(1)