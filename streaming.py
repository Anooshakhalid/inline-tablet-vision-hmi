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

conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

data = b""
payload_size = struct.calcsize("Q")

def recv_all(size):
    buffer = b""
    while len(buffer) < size:
        packet = conn.recv(size - len(buffer))
        if not packet:
            return None
        buffer += packet
    return buffer

while True:

    # =========================
    # GET SIZE
    # =========================
    packed_size = recv_all(payload_size)
    if not packed_size:
        break

    msg_size = struct.unpack("Q", packed_size)[0]

    # =========================
    # GET FRAME
    # =========================
    frame_data = recv_all(msg_size)
    if frame_data is None:
        break

    # =========================
    # DECODE
    # =========================
    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    # =========================
    # FORCE LATEST FRAME (KEY FIX)
    # =========================
    conn.setblocking(False)
    try:
        while True:
            extra = conn.recv(4096)
            if not extra:
                break
            # discard extra buffered frames
    except:
        pass
    conn.setblocking(True)

    # =========================
    # DISPLAY
    # =========================
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow("YOLO LIVE STREAM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

conn.close()
cv2.destroyAllWindows()