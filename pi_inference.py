import cv2
import subprocess
from ultralytics import YOLO

model = YOLO("models/model.pt")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# GStreamer PIPE (IMPORTANT)
gst_str = (
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=1200 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=10.52.20.113 port=5000 sync=false"
)

out = cv2.VideoWriter(
    gst_str,
    cv2.CAP_GSTREAMER,
    0,
    30,
    (640, 480),
    True
)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO inference
    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    annotated = results[0].plot()

    # WRITE DIRECTLY TO GSTREAMER PIPELINE
    out.write(annotated)

cap.release()
out.release()