import cv2
import time
import threading
import numpy as np
import subprocess
from ultralytics import YOLO
from queue import Queue

from processing.analyzer import process
from database.db import save_to_influx
from utils.batch_manager import BatchManager

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640

WIDTH = 640
HEIGHT = 480
FPS = 15

CAMERA = "/dev/video0"
RTSP_URL = "rtsp://192.168.100.121:8554/live"

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# QUEUES
# =====================
camera_queue = Queue(maxsize=1)
stream_queue = Queue(maxsize=1)

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
# FFmpeg PIPE (FIXED)
# =====================
ffmpeg = subprocess.Popen([
    "ffmpeg",
    "-loglevel", "info",

    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",

    # 🔥 IMPORTANT RTSP SETTINGS
    "-rtsp_transport", "tcp",
    "-muxdelay", "0",
    "-muxpreload", "0",

    "-f", "rtsp",
    RTSP_URL
], stdin=subprocess.PIPE)

print("[INFO] FFmpeg started")

time.sleep(2)  # allow MediaMTX to stabilize

# =====================
# QC STATE
# =====================
batch_manager = BatchManager()
batch_id = batch_manager.new_batch()
frame_count = 0
FRAME_LIMIT = 30

# =====================
# CAMERA THREAD
# =====================
def camera_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 🔥 FORCE SIZE (VERY IMPORTANT)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        if camera_queue.full():
            try:
                camera_queue.get_nowait()
            except:
                pass

        camera_queue.put(frame)

# =====================
# INFERENCE THREAD
# =====================
def inference_loop():
    global batch_id, frame_count

    while True:
        if camera_queue.empty():
            time.sleep(0.001)
            continue

        frame = camera_queue.get()

        results = model(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = results.names[cls_id]

            detections.append({
                "class": name,
                "confidence": conf
            })

        result = process(detections, batch_id)

        try:
            save_to_influx(result)
        except Exception as e:
            print("[WARN] DB error:", e)

        print("QC RESULT:", result)

        frame_count += 1
        if frame_count >= FRAME_LIMIT:
            frame_count = 0
            batch_id = batch_manager.new_batch()
            print(f"[NEW BATCH] {batch_id}")

        annotated = results.plot()
        annotated = cv2.resize(annotated, (WIDTH, HEIGHT))
        annotated = np.ascontiguousarray(annotated)

        if stream_queue.full():
            try:
                stream_queue.get_nowait()
            except:
                pass

        stream_queue.put(annotated)

# =====================
# STREAM THREAD
# =====================
def stream_loop():
    while True:
        if stream_queue.empty():
            time.sleep(0.001)
            continue

        frame = stream_queue.get()

        try:
            ffmpeg.stdin.write(frame.tobytes())
            ffmpeg.stdin.flush()
        except BrokenPipeError:
            print("[FFMPEG ERROR] Pipe broken — FFmpeg crashed")
            break
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

print("[INFO] FULL PIPELINE RUNNING")

while True:
    time.sleep(1)