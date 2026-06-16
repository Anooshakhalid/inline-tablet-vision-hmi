import socket

HOST = "0.0.0.0"
PORT = 9999

print("[LOG SERVER] Starting...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)

    conn, addr = s.accept()
    print("[CONNECTED]", addr)

    buffer = ""

    with conn, open("qc_logs.txt", "a", buffering=1) as f:

        while True:
            data = conn.recv(4096)

            if not data:
                print("Client disconnected")
                break

            buffer += data.decode()

            # logs are newline separated JSON
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)

                if not line.strip():
                    continue

                print(line)
                f.write(line + "\n")