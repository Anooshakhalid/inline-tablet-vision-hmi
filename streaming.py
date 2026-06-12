import socket
import cv2
import struct
import numpy as np

HOST = "0.0.0.0"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print("Connected:", addr)

payload_size = struct.calcsize("Q")

def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

frame_count = 0

while True:
    try:
        packed_size = recv_exact(conn, payload_size)
        if not packed_size:
            print("Disconnected or no size received")
            break

        msg_size = struct.unpack("Q", packed_size)[0]

        # safety check (VERY IMPORTANT)
        if msg_size > 10_000_000:
            print("Frame too large, skipping")
            continue

        frame_data = recv_exact(conn, msg_size)
        if frame_data is None:
            print("Frame data missing")
            continue

        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            print("Decode failed")
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            print("Frames received:", frame_count)

        cv2.imshow("YOLO LIVE STREAM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("Receiver error:", e)
        break

conn.close()
cv2.destroyAllWindows()