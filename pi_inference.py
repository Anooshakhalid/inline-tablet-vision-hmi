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
# SHARED STATE (LATEST ONLY)
# =====================
lock = threading.Lock()
latest_frame = None
latest_overlay = None

# =====================
# CAMERA THREAD (FAST)
# =====================
def camera_loop():
    global latest_frame

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    print("[CAM] started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with lock:
            latest_frame = frame

# =====================
# YOLO THREAD (SLOW - SKIP SAFE)
# =====================
def yolo_loop():
    global latest_overlay

    frame_id = 0

    while True:

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_id += 1

        # SKIP FRAMES (VERY IMPORTANT)
        if frame_id % 2 != 0:
            continue

        start = time.time()

        results = model(
            frame,
            imgsz=320,   # IMPORTANT for realtime
            conf=0.25,
            verbose=False
        )[0]

        overlay = results.plot()

        fps = round(1 / max(time.time() - start, 0.001), 2)

        cv2.putText(
            overlay,
            f"YOLO FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        overlay = np.ascontiguousarray(overlay, dtype=np.uint8)

        with lock:
            latest_overlay = overlay

# =====================
# STREAM THREAD (ZERO BLOCKING)
# =====================
def stream_loop():

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "error",

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

            # LOW LATENCY FLAGS
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-flush_packets", "1",

            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            RTSP_URL,
        ],
        stdin=subprocess.PIPE,
        bufsize=0
    )

    print("[STREAM] started")

    while True:

        with lock:
            frame = latest_overlay if latest_overlay is not None else latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        try:
            ffmpeg.stdin.write(frame.tobytes())

        except BrokenPipeError:
            print("[FFMPEG DEAD]")
            break

# =====================
# START ALL THREADS
# =====================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=yolo_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()

print("[SYSTEM] REALTIME PIPELINE RUNNING")

while True:
    time.sleep(1)