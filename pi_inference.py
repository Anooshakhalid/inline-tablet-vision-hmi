import cv2
import subprocess
import threading
import numpy as np
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"

WIDTH = 640
HEIGHT = 480
FPS = 15

RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# LOAD MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Camera opened")

# =====================
# FFMPEG
# =====================
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
        RTSP_URL,
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print("[INFO] FFmpeg started")

# =====================
# FFMPEG LOGGER
# =====================
def ffmpeg_logger():
    while True:
        line = ffmpeg.stderr.readline()
        if not line:
            break

        print("[FFMPEG]", line.decode(errors="ignore").strip())

threading.Thread(target=ffmpeg_logger, daemon=True).start()

# =====================
# MAIN LOOP
# =====================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # =====================
    # YOLO
    # =====================
    results = model(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    annotated = results.plot()
    annotated = cv2.resize(annotated, (WIDTH, HEIGHT))
    annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

    try:
        ffmpeg.stdin.write(annotated.tobytes())

    except BrokenPipeError:
        print("[ERROR] FFmpeg pipe broken")
        break

    except Exception as e:
        print("[ERROR]", e)
        break