import cv2
import subprocess
import threading
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 15

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-loglevel", "info",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",

        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",

        "-pix_fmt", "yuv420p",

        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://127.0.0.1:8554/live"
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

def ffmpeg_logs():
    while True:
        line = ffmpeg.stderr.readline()
        if not line:
            break
        print("[FFMPEG]", line.decode(errors="ignore").strip())

threading.Thread(target=ffmpeg_logs, daemon=True).start()

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    try:
        ffmpeg.stdin.write(frame.tobytes())
    except Exception as e:
        print("WRITE ERROR:", e)
        break