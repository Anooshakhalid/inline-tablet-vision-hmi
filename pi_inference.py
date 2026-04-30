import cv2
import socket
import struct
import threading
from ultralytics import YOLO

from processing.analyzer import process
from database.db import save_to_influx
from utils.batch_manager import BatchManager

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/best.pt"
IMG_SIZE = 640
DEVICE = "cpu"

PC_IP = "192.168.100.213"
PORT = 9999

FRAME_LIMIT = 30
frame_count = 0

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture("http://192.168.100.49:8080/video", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =========================
# SOCKET
# =========================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        client_socket.connect((PC_IP, PORT))
        break
    except:
        print("[INFO] Waiting for PC...")

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

print("[INFO] Connected. Streaming...")

# =========================
# ASYNC SEND
# =========================
def send_frame(sock, message):
    try:
        sock.sendall(message)
    except:
        pass

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=0.25,
        device=DEVICE,
        verbose=False
    )

    r = results[0]

    # =========================
    # YOUR LOGIC (UNCHANGED)
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

    result = process(detections, batch_id)

    try:
        save_to_influx(result)
    except Exception as e:
        print("[WARN] DB error:", e)

    print("QC RESULT:", result)

    # =========================
    # BATCH CONTROL
    # =========================
    frame_count += 1
    if frame_count >= FRAME_LIMIT:
        frame_count = 0
        batch_id = batch_manager.new_batch()
        print(f"\nNEW BATCH: {batch_id}\n")

    # =========================
    # EXACT COLAB VISUALIZATION
    # =========================
    annotated_frame = r.plot()


    # =========================
    # SEND TO LAPTOP (ASYNC)
    # =========================
    try:
        _, buffer = cv2.imencode(
            '.jpg',
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        data = buffer.tobytes()
        message = struct.pack("Q", len(data)) + data

        try:
            client_socket.sendall(message)
        except Exception as e:
            print("[ERROR] Send failed:", e)

    except Exception as e:
        print("[ERROR] Send failed:", e)
        break

    print("\n--- YOLO RAW OUTPUT ---")

    if len(r.boxes) == 0:
        print("No detections")
    else:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls_id]

            print(f"Class: {name}, Confidence: {conf:.2f}")

    print("------------------------\n")
    print("Processed detections:", detections)


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
client_socket.close()