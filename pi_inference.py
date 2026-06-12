import cv2
import time
import threading
import numpy as np
import subprocess
from queue import Queue
from ultralytics import YOLO

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
RTSP_URL = "rtsp://127.0.0.1:8554/live"

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
# FFmpeg PROCESS (FIXED)
# =====================
ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
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

        "-g", str(FPS * 2),

        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        RTSP_URL
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0
)

print("[INFO] FFmpeg started")

# =====================
# FFmpeg LOG THREAD (IMPORTANT)
# =====================
def ffmpeg_logger():
    for line in ffmpeg.stderr:
        print("[FFMPEG]", line.decode().strip())

threading.Thread(target=ffmpeg_logger, daemon=True).start()

time.sleep(2)

# =====================
# STATE
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

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        if camera_queue.full():
            camera_queue.get_nowait()

        camera_queue.put(frame)

# =====================
# INFERENCE THREAD
# =====================
def inference_loop():
    global batch_id, frame_count

    while True:
        if camera_queue.empty():
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
        annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

        if stream_queue.full():
            stream_queue.get_nowait()

        stream_queue.put(annotated)

# =====================
# STREAM THREAD (FIXED)
# =====================
def stream_loop():
    while True:
        if stream_queue.empty():
            continue

        frame = stream_queue.get()

        if ffmpeg.poll() is not None:
            print("[FFMPEG DEAD]")
            break

        try:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            ffmpeg.stdin.write(frame.tobytes())

        except BrokenPipeError:
            print("[FFMPEG PIPE BROKEN]")
            break

        except Exception as e:
            print("[FFMPEG ERROR]", e)
            break

# =====================
# START THREADS
# =====================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()

print("[INFO] PIPELINE RUNNING")

# =====================
# MAIN LOOP
# =====================
while True:
    time.sleep(1)