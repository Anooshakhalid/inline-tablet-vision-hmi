import cv2
import time
import subprocess
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "models/model.pt"
CAMERA = "/dev/video0"
WIDTH, HEIGHT = 640, 480
FPS = 15

RTSP_URL = "rtsp://192.168.100.121:8554/live"  # IMPORTANT: PI IP

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[OK] Camera opened")

# =====================
# FFmpeg STARTER (with debug enabled)
# =====================
def start_ffmpeg():
    return subprocess.Popen([
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
        RTSP_URL
    ], stdin=subprocess.PIPE)

ffmpeg = start_ffmpeg()

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO inference
    results = model(frame, imgsz=640, verbose=False)[0]
    annotated = results.plot()

    annotated = cv2.resize(annotated, (WIDTH, HEIGHT))

    try:
        ffmpeg.stdin.write(annotated.tobytes())

    except Exception as e:
        print("[FFMPEG RESTART]", e)
        ffmpeg = start_ffmpeg()
        time.sleep(1)