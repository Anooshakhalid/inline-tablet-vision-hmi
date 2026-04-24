import cv2
import time
import requests
from ultralytics import YOLO

# =========================
# CONFIG (EDIT THIS)
# =========================
PC_API_URL = "http://192.168.100.7:8086/api/v2/write"
MODEL_PATH = "models/best.pt"
FRAME_SIZE = 320
DEVICE = "cpu"

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
# INFERENCE LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    # resize for edge performance
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    # YOLO inference
    results = model(frame, imgsz=320, device=DEVICE)

    r = results[0]
    defect_count = len(r.boxes)

    print(f"[INFO] Defects: {defect_count}")

    # =========================
    # DRAW RESULTS (OPEN CV WINDOW)
    # =========================
    annotated_frame = r.plot()

    cv2.imshow("Tablet QC - Detection", annotated_frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # =========================
    # SEND DATA TO PC (InfluxDB API)
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
cv2.destroyAllWindows()