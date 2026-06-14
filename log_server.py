import socket

HOST = "0.0.0.0"
PORT = 9999

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print(f"[LOG SERVER] Listening on {PORT}")

        conn, addr = s.accept()
        print("[CONNECTED]", addr)

        with conn, open("qc_logs.txt", "a") as f:
            while True:
                data = conn.recv(4096)
                if not data:
                    break

                log = data.decode()
                print(log, end="")
                f.write(log)
                f.flush()