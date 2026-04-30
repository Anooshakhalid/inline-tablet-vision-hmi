import cv2
import socket
import struct
import time
from ultralytics import YOLO

from processing.analyzer import process
from database.db import save_to_influx
from utils.batch_manager import BatchManager

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/bestt.pt"
FRAME_SIZE = 320
DEVICE = "cpu"

PC_IP = "192.168.100.213"
PORT = 9999

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# Socket (RELIABLE VERSION)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        client_socket.connect((PC_IP, PORT))
        break
    except:
        print("[INFO] Waiting for PC...")
        time.sleep(2)

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

print("[INFO] Connected. Sending stream...")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    # =========================
    # YOLO
    # =========================
    results = model(frame, imgsz=320, device=DEVICE, verbose=False)
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

    # =========================
    # BUSINESS LOGIC
    # =========================
    result = process(detections, batch_id)

    try:
        save_to_influx(result)
    except Exception as e:
        print("[WARN] DB error:", e)

    # =========================
    # DRAW
    # =========================
    for d, box in zip(detections, r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = d["class"]

        color = (0, 255, 0) if label == "normal" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # =========================
    # SEND FRAME (FAST VERSION)
    # =========================
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 45])

        # send raw jpeg (NO pickle)
        data = buffer.tobytes()

        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)

    except Exception as e:
        print("[ERROR] Send failed:", e)
        break