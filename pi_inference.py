import cv2
import socket
import struct
import threading
import time
from ultralytics import YOLO

from processing.analyzer import process
from database.db import save_to_influx
from utils.batch_manager import BatchManager

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640
DEVICE = "cpu"

PC_IP = "192.168.100.175"
PORT = 9999

FRAME_LIMIT = 30
prev = time.time()

# =========================
# GLOBALS (LATEST ONLY)
# =========================
latest_frame = None
latest_annotated = None
latest_result = None

lock = threading.Lock()

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

while True:
    try:
        client_socket.connect((PC_IP, PORT))
        break
    except:
        print("[INFO] Waiting for PC...")

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

print("[INFO] Connected. Industrial pipeline started")

# =========================
# THREAD 1 - CAMERA (FAST)
# =========================
def camera_loop():
    global latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame

# =========================
# THREAD 2 - YOLO (SLOW SAFE)
# =========================
def inference_loop():
    global latest_frame, latest_annotated, latest_result, batch_id

    frame_count = 0

    while True:

        if latest_frame is None:
            continue

        with lock:
            frame = latest_frame.copy()

        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=0.25,
            device=DEVICE,
            verbose=False
        )[0]

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

        # async-safe DB write (no blocking stream)
        try:
            save_to_influx(result)
        except Exception as e:
            print("[WARN] DB error:", e)

        latest_result = result
        latest_annotated = results.plot()

        print("QC RESULT:", result)
        print("FPS:", 1 / (time.time() - prev))
        prev = time.time()
        # batch update
        frame_count += 1
        if frame_count >= FRAME_LIMIT:
            frame_count = 0
            batch_id = batch_manager.new_batch()
            print(f"\nNEW BATCH: {batch_id}\n")

# =========================
# THREAD 3 - STREAM (ULTRA FAST)
# =========================
def stream_loop():
    global latest_annotated

    while True:

        if latest_annotated is None:
            continue

        with lock:
            frame = latest_annotated.copy()

        try:
            _, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            data = buffer.tobytes()
            message = struct.pack("Q", len(data)) + data

            client_socket.sendall(message)

        except Exception as e:
            print("[STREAM ERROR]", e)

# =========================
# START THREADS
# =========================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()

# =========================
# KEEP ALIVE
# =========================
while True:
    time.sleep(1)