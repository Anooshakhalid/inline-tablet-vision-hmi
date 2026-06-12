import cv2
import socket
import struct
from ultralytics import YOLO

# =========================
MODEL_PATH = "models/model.pt"
IMG_SIZE = 640
DEVICE = "cpu"

PC_IP = "192.168.100.175"
PORT = 9999

# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =========================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)

while True:
    try:
        client_socket.connect((PC_IP, PORT))
        break
    except:
        print("[INFO] Waiting for PC...")

print("[INFO] Connected. Streaming...")

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
    annotated_frame = r.plot()

    # =========================
    # ENCODE (NO QUALITY LOSS ADJUSTMENT)
    # =========================
    _, buffer = cv2.imencode(
        '.jpg',
        annotated_frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    data = buffer.tobytes()
    message = struct.pack("Q", len(data)) + data

    # =========================
    # NON-BLOCKING SEND (IMPORTANT)
    # =========================
    try:
        client_socket.send(message)
    except:
        pass