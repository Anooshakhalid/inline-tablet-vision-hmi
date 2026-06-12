import socket
import cv2
import struct
import numpy as np

HOST = "0.0.0.0"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)

server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print("Connected:", addr)

conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
conn.settimeout(0.001)   

payload_size = struct.calcsize("Q")

def recv_all(size):
    data = b""
    while len(data) < size:
        try:
            packet = conn.recv(size - len(data))
            if not packet:
                return None
            data += packet
        except:
            return None
    return data

while True:
    try:
        packed_size = recv_all(payload_size)
        if not packed_size:
            continue

        msg_size = struct.unpack("Q", packed_size)[0]

        frame_data = recv_all(msg_size)
        if frame_data is None:
            continue

        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        # show FULL quality (no resize unless needed)
        cv2.imshow("YOLO LIVE STREAM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except:
        continue

conn.close()
cv2.destroyAllWindows()