import socket
import cv2
import struct
import numpy as np

HOST = "0.0.0.0"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print("Connected:", addr)

# reduce latency
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

data = b""
payload_size = struct.calcsize("Q")

while True:
    # get message size
    while len(data) < payload_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < payload_size:
        break

    packed_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_size)[0]

    # get frame data
    while len(data) < msg_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # DROP backlog (important!)
    if len(data) > msg_size:
        data = data[-msg_size:]

    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    # faster display
    frame = cv2.resize(frame, (480, 480))

    cv2.imshow("YOLO LIVE STREAM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

conn.close()
cv2.destroyAllWindows()