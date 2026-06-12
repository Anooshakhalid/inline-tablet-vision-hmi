import cv2
import time
import threading
import numpy as np
import subprocess
from ultralytics import YOLO
from queue import Queue

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640

WIDTH = 640
HEIGHT = 480
FPS = 15

CAMERA = "/dev/video0"
RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Camera opened")

# =====================
# FRAME QUEUE (IMPORTANT FIX)
# =====================
frame_queue = Queue(maxsize=1)

# =====================
# FFmpeg PROCESS
# =====================
ffmpeg = subprocess.Popen([
    "ffmpeg",
    "-loglevel", "error",

    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",

    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL
], stdin=subprocess.PIPE)

print("[INFO] RTSP streaming started")

# =====================
# CAMERA THREAD
# =====================
def camera_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get()

        frame_queue.put(frame)

# =====================
# YOLO THREAD
# =====================
def inference_loop():
    while True:
        if frame_queue.empty():
            time.sleep(0.001)
            continue

        frame = frame_queue.get()

        results = model(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]

        annotated = results.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated = cv2.resize(annotated, (WIDTH, HEIGHT))
        annotated = np.ascontiguousarray(annotated)

        # store back for streaming
        frame_queue.put(annotated)

# =====================
# STREAM THREAD (SEPARATED)
# =====================
def stream_loop():
    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        try:
            ffmpeg.stdin.write(frame.tobytes())
        except Exception as e:
            print("[FFMPEG ERROR]", e)
            break

        time.sleep(1 / FPS)

# =====================
# START
# =====================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()

print("[INFO] Pipeline running")

while True:
    time.sleep(1)