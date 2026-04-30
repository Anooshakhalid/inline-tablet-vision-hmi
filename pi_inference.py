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
MODEL_PATH = "models/bestt.pt"
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

# Use faster backend (important)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows (use CAP_V4L2 on Linux)
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
# NON-BLOCKING SEND
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
    # YOLO
    # =========================
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=0.25,
        device=DEVICE,
        verbose=False
    )

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
    # PROCESS
    # =========================
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
    # DRAW
    # =========================
    for d, box in zip(detections, r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{d['class']} {d['confidence']:.2f}"

        color = (0, 255, 0) if d["class"] in ["normal", "tablet"] else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # =========================
    # ✅ DISPLAY FIRST (NO LAG)
    # =========================
    display_frame = cv2.resize(frame, (480, 480))  # faster rendering
    cv2.imshow("QC", display_frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # =========================
    # SEND FRAME (ASYNC)
    # =========================
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        data = buffer.tobytes()
        message = struct.pack("Q", len(data)) + data

        threading.Thread(
            target=send_frame,
            args=(client_socket, message),
            daemon=True
        ).start()

    except Exception as e:
        print("[ERROR] Send failed:", e)
        break

# =========================
# CLEANUP --
# =========================
cap.release()
cv2.destroyAllWindows()
client_socket.close()