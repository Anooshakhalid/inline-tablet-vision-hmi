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
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640
DEVICE = "cpu"

PC_IP = "192.168.100.175"
PORT = 9999

FRAME_LIMIT = 30
frame_count = 0

# =========================
# SHARED STREAM FRAME
# =========================
latest_frame = None
frame_lock = threading.Lock()

# =========================
# INIT
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =========================
# SOCKET
# =========================
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

print("[INFO] Connected. Streaming...")

# =========================
# STREAM THREAD
# =========================
def stream_frames():
    global latest_frame

    while True:

        with frame_lock:
            if latest_frame is None:
                continue

            frame_to_send = latest_frame.copy()

        try:
            _, buffer = cv2.imencode(
                ".jpg",
                frame_to_send,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            data = buffer.tobytes()
            message = struct.pack("Q", len(data)) + data

            client_socket.sendall(message)

        except Exception as e:
            print("[STREAM ERROR]", e)
            break

# Start stream thread
threading.Thread(
    target=stream_frames,
    daemon=True
).start()

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
    # DETECTIONS
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
    # QC LOGIC
    # =========================
    result = process(detections, batch_id)

#    try:
#        save_to_influx(result)
#    except Exception as e:
#       print("[WARN] DB error:", e)

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
    # VISUALIZATION
    # =========================
    annotated_frame = r.plot()

    # =========================
    # UPDATE LATEST FRAME
    # =========================
    with frame_lock:
        latest_frame = annotated_frame.copy()

    print("Processed detections:", detections)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
client_socket.close()