import cv2
import time
import threading
import numpy as np
import subprocess

from pi_inference import latest_frame, lock

# =====================
# CONFIG
# =====================
WIDTH = 640
HEIGHT = 480
FPS = 15

RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# FFmpeg PIPE → RTSP
# =====================
ffmpeg = subprocess.Popen([
    "ffmpeg",

    "-loglevel", "error",

    "-f", "image2pipe",
    "-vcodec", "mjpeg",
    "-i", "-",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-g", "1",
    "-bf", "0",

    "-pix_fmt", "yuv420p",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL
], stdin=subprocess.PIPE)


# =====================
# STREAM LOOP
# =====================
def stream_loop():

    global latest_frame

    last = None

    while True:

        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        if frame is last:
            continue

        last = frame

        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            ffmpeg.stdin.write(buffer.tobytes())

        except Exception as e:
            print("[STREAM ERROR]", e)
            break


# =====================
# START
# =====================
threading.Thread(target=stream_loop, daemon=True).start()

print("[INFO] RTSP streaming started")

while True:
    time.sleep(1)