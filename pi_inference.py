import cv2
import time
import threading
import subprocess
from ultralytics import YOLO

# =========================
# YOUR MODULES
# =========================
from processing.analyzer import process
from database.db import save_to_influx
from utils.batch_manager import BatchManager

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/bestt.pt"
FRAME_SIZE = 320
DEVICE = "cpu"
FRAME_LIMIT = 50

CAM_URL = "http://192.168.100.49:8080/video"

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] System Started")

# =========================
# RTSP STREAM (FFMPEG)
# =========================
ffmpeg = subprocess.Popen([
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '320x320',
    '-r', '30',
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'rtsp',
    'rtsp://localhost:8554/live'
], stdin=subprocess.PIPE)

# =========================
# STATE
# =========================
batch_manager = BatchManager()
batch_id = batch_manager.new_batch()
frame_count = 0

# =========================
# INFERENCE LOOP
# =========================
def inference_loop():
    global batch_id, frame_count

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        results = model(frame, imgsz=320, device=DEVICE)
        r = results[0]

        detections = []

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls_id]

            detections.append({
                "class": name,
                "confidence": conf
            })

        result = process(detections, batch_id)

        try:
            save_to_influx(result)
        except Exception as e:
            print("[WARN] Influx write failed:", e)

        print("QC RESULT:", result)

        # =========================
        # DRAW BOXES
        # =========================
        annotated = frame.copy()

        for d, box in zip(detections, r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = d["class"]

            color = (0, 255, 0) if label == "normal" else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # =========================
        # PUSH TO RTSP (IMPORTANT PART)
        # =========================
        ffmpeg.stdin.write(annotated.tobytes())

        # =========================
        # BATCH CONTROL
        # =========================
        frame_count += 1

        if frame_count >= FRAME_LIMIT:
            frame_count = 0
            batch_id = batch_manager.new_batch()
            print(f"\nNEW BATCH: {batch_id}\n")

        time.sleep(0.03)

# =========================
# START
# =========================
if __name__ == "__main__":

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    while True:
        time.sleep(1)