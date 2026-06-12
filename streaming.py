import subprocess
import time

# =====================
# CONFIG
# =====================
CAMERA = "/dev/video1"
WIDTH = 640
HEIGHT = 480
FPS = 15

RTSP_URL = "rtsp://127.0.0.1:8554/live"

# =====================
# FFmpeg PIPELINE (DIRECT CAMERA → RTSP)
# =====================
cmd = [
    "ffmpeg",

    "-f", "v4l2",
    "-i", CAMERA,

    # encoding
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",

    # performance tuning
    "-r", str(FPS),
    "-g", "15",
    "-bf", "0",

    # RTSP output
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL
]

print("[INFO] Starting RTSP stream...")
print("[INFO] URL:", RTSP_URL)

process = subprocess.Popen(cmd)

try:
    process.wait()
except KeyboardInterrupt:
    print("\n[INFO] Stopping stream...")
    process.terminate()