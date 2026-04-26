import cv2
import time
import threading
from flask import Flask, Response
from ultralytics import YOLO
from datetime import datetime

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
FRAME_LIMIT = 50   # batch size

# =========================
# INIT
# =========================
app = Flask(__name__)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# camera stability
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] System Started")

latest_frame = None
lock = threading.Lock()

# batch system
batch_manager = BatchManager()
batch_id = batch_manager.new_batch()
frame_count = 0

# =========================
# INFERENCE THREAD
# =========================
def inference_loop():
    global latest_frame, frame_count, batch_id

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        results = model(frame, imgsz=320, device=DEVICE)
        r = results[0]

        # =========================
        # CONVERT YOLO → detections list
        # =========================
        detections = []

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls_id]

            detections.append({
                "class": name,
                "confidence": conf
            })

        # =========================
        # PROCESS LOGIC
        # =========================
        result = process(detections, batch_id)

        # =========================
        # SAVE TO INFLUX
        # =========================
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
        # UPDATE FRAME (THREAD SAFE)
        # =========================
        with lock:
            latest_frame = annotated

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
# FLASK STREAM
# =========================
def generate():
    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.03)

# =========================
# ROUTE
# =========================
@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# START SYSTEM
# =========================
if __name__ == "__main__":

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)