import cv2
import time
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240

# use small model for smooth FPS
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Camera started")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # =====================
    # YOLO INFERENCE
    # =====================
    start = time.time()

    results = model(
        frame,
        imgsz=256,
        conf=0.25,
        verbose=False
    )[0]

    frame = results.plot()

    # =====================
    # FPS CALC
    # =====================
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # =====================
    # SHOW WINDOW (VNC WILL DISPLAY THIS)
    # =====================
    cv2.imshow("YOLO VNC VIEW", frame)

    # IMPORTANT: required for imshow to work
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()