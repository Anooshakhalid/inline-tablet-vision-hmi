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
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640
DEVICE = "cpu"

PC_IP = "10.52.20.113"
PORT = 9999

FRAME_LIMIT = 30

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =========================
# SOCKET (RECONNECT SAFE)
# =========================
def connect_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((PC_IP, PORT))
            print("[INFO] Connected to PC")
            return s
        except:
            print("[INFO] Waiting for PC...")
            time.sleep(1)

client_socket = connect_socket()

# =========================
# BATCH
# =========================
batch_manager = BatchManager()
batch_id = batch_manager.new_batch()
frame_count = 0

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
    results = model(frame, imgsz=IMG_SIZE, conf=0.25, device=DEVICE, verbose=False)
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

    print("QC RESULT:", result)

    # =========================
    # BATCH CONTROL
    # =========================
    frame_count += 1
    if frame_count >= FRAME_LIMIT:
        frame_count = 0
        batch_id = batch_manager.new_batch()
        print(f"[NEW BATCH] {batch_id}")

    # =========================
    # VISUALIZATION FRAME
    # =========================
    annotated_frame = r.plot()
    annotated_frame = cv2.resize(annotated_frame, (640, 480))

    # IMPORTANT FIX: contiguous memory
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    annotated_frame = np.ascontiguousarray(annotated_frame)

    # =========================
    # SEND TO PC (STABLE PACKING)
    # =========================
    try:
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buffer.tobytes()

        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)

    except Exception as e:
        print("[ERROR] Send failed:", e)
        client_socket = connect_socket()

    print("Processed detections:", detections)

# =========================
# CLEANUP
# =========================
cap.release()
client_socket.close()