import socket
import cv2
import pickle
import struct

# =========================
# SERVER CONFIG (PC SIDE)
# =========================
HOST = "0.0.0.0"   # listen on all interfaces
PORT = 9999

# =========================
# SOCKET SETUP
# =========================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("[INFO] Waiting for connection...")

conn, addr = server_socket.accept()
print("[INFO] Connected from:", addr)

# =========================
# DATA HANDLING
# =========================
data = b""
payload_size = struct.calcsize("Q")

# =========================
# STREAM LOOP
# =========================
while True:
    try:
        # Get message size
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]

        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Get full frame
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Decode frame
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # Show window
        cv2.imshow("YOLO Stream (Live from Pi)", frame)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("[ERROR]", e)
        break

# =========================
# CLEANUP
# =========================
conn.close()
server_socket.close()
cv2.destroyAllWindows()