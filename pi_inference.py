import cv2
import subprocess
import threading
import numpy as np
import time

WIDTH = 640
HEIGHT = 480
FPS = 15

RTSP_URL = "rtsp://127.0.0.1:8554/live"

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

        "-g", "15",
        "-keyint_min", "15",

        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        RTSP_URL,
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0,
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
# STREAM LOOP
# =====================
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("[WARN] Camera read failed")
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"[INFO] Frames sent: {frame_count}")

    # ---------------------
    # TEST OVERLAY
    # ---------------------
    annotated = frame.copy()

    cv2.putText(
        annotated,
        f"STREAM TEST {frame_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        time.strftime("%H:%M:%S"),
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    annotated = np.ascontiguousarray(
        annotated,
        dtype=np.uint8
    )

    try:
        ffmpeg.stdin.write(annotated.tobytes())

    except BrokenPipeError:
        print("[ERROR] FFmpeg pipe broken")
        break

    except Exception as e:
        print("[ERROR]", e)
        break