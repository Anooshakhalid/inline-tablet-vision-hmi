import cv2
import socket
import struct
from ultralytics import YOLO

PC_IP = "10.52.20.113"
PORT = 9999

model = YOLO("models/model.pt")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("Connecting...")

while True:
    try:
        client_socket.connect((PC_IP, PORT))
        break
    except:
        pass

print("Connected!")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    annotated = results[0].plot()

    ok, buffer = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not ok:
        continue

    data = buffer.tobytes()
    message = struct.pack("Q", len(data)) + data

    try:
        client_socket.sendall(message)
        frame_id += 1

        if frame_id % 30 == 0:
            print("Sent frames:", frame_id)

    except Exception as e:
        print("Send error:", e)
        break