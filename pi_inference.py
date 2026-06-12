import cv2
import time
import threading
import numpy as np

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

RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA (RTSP INPUT - FROM FFMPEG PIPELINE)
# =====================
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot open RTSP stream")

print("[INFO] RTSP stream opened for inference")

# =====================
# QC STATE
# =====================
batch_manager = BatchManager()
batch_id = batch_manager.new_batch()
frame_count = 0
FRAME_LIMIT = 30

# =====================
# INFERENCE LOOP ONLY
# =====================
def inference_loop():
    global batch_id, frame_count

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # YOLO inference
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

        # process QC
        result = process(detections, batch_id)

        try:
            save_to_influx(result)
        except Exception as e:
            print("[WARN] DB error:", e)

        print("QC RESULT:", result)

        # batch handling
        frame_count += 1
        if frame_count >= FRAME_LIMIT:
            frame_count = 0
            batch_id = batch_manager.new_batch()
            print(f"[NEW BATCH] {batch_id}")

        # optional visualization
        annotated = results.plot()
        cv2.imshow("AI Stream", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# =====================
# START
# =====================
print("[INFO] Starting inference pipeline...")
inference_loop()