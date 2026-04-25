import cv2
import time
import requests
import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
PC_API_URL = "http://192.168.100.7:8086/api/v2/write"
MODEL_PATH = "models/best.pt"
FRAME_SIZE = 320
DEVICE = "cpu"

SHOW_WINDOW = False   # will auto-enable if display exists

# =========================
# AUTO DISPLAY CHECK
# =========================
if "DISPLAY" in os.environ:
    SHOW_WINDOW = True

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# CAMERA INIT
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Pi Inference Started")

# =========================
# LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    # YOLO inference
    results = model(frame, imgsz=320, device=DEVICE)

    r = results[0]
    defect_count = len(r.boxes)

    print(f"[INFO] Defects: {defect_count}")

    # =========================
    # DRAW OUTPUT
    # =========================
    annotated_frame = r.plot()

    # =========================
    # SAFE VISUALIZATION
    # =========================
    if SHOW_WINDOW:
        cv2.imshow("Tablet QC - Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # fallback: save preview frame (optional)
        cv2.imwrite("latest_frame.jpg", annotated_frame)

    # =========================
    # SEND TO PC
    # =========================
    payload = {
        "defects": defect_count,
        "timestamp": time.time()
    }

    try:
        requests.post(
            PC_API_URL,
            json=payload,
            timeout=0.5
        )
    except Exception as e:
        print("[WARN] Send failed:", e)

    time.sleep(0.01)

# =========================
# CLEANUP
# =========================
cap.release()
if SHOW_WINDOW:
    cv2.destroyAllWindows()