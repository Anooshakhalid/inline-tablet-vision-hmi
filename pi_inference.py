import cv2
import time
import threading
import numpy as np
import subprocess
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
WIDTH = 640
HEIGHT = 480
FPS = 15

RTSP_URL = "rtsp://127.0.0.1:8554/live"
MODEL_PATH = "models/model.pt"

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# SHARED STATE (IMPORTANT)
# =====================
latest_frame = None
latest_annotated = None
lock = threading.Lock()

# =====================
# CAMERA THREAD
# =====================
def camera_thread():
    global latest_frame

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    print("[INFO] Camera started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with lock:
            latest_frame = frame

# =====================
# YOLO THREAD
# =====================
def inference_thread():
    global latest_annotated

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        start = time.time()

        results = model(
            frame,
            imgsz=320,   # IMPORTANT: faster on Pi
            conf=0.25,
            verbose=False
        )[0]

        annotated = results.plot()

        fps = round(1 / max(time.time() - start, 0.001), 2)
        cv2.putText(
            annotated,
            f"YOLO FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        annotated = cv2.resize(annotated, (WIDTH, HEIGHT))
        annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

        with lock:
            latest_annotated = annotated

# =====================
# FFMPEG STREAMER
# =====================
def stream_thread():
    global latest_annotated

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

            "-g", "15",
            "-keyint_min", "15",

            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            RTSP_URL,
        ],
        stdin=subprocess.PIPE,
        bufsize=0
    )

    print("[INFO] FFmpeg started")

    while True:
        with lock:
            frame = latest_annotated

        if frame is None:
            time.sleep(0.01)
            continue

        try:
            ffmpeg.stdin.write(frame.tobytes())

        except BrokenPipeError:
            print("[ERROR] FFmpeg crashed")
            break

        except Exception as e:
            print("[ERROR]", e)
            break

# =====================
# START THREADS
# =====================
threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=inference_thread, daemon=True).start()
threading.Thread(target=stream_thread, daemon=True).start()

print("[INFO] SYSTEM RUNNING")

while True:
    time.sleep(1)