import cv2
import time
from ultralytics import YOLO

from processing.analyzer import process
from utils.batch_manager import BatchManager
from influx_worker import InfluxWorker

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240
MODEL_PATH = "models/model.pt"

TABLETS_PER_BATCH = 50

# =====================
# INIT
# =====================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Camera started")

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

influx = InfluxWorker()
influx.start()

tablet_counter = 0

# helps prevent duplicate counting
prev_tablet_present = False
stable_frames = 0
STABLE_THRESHOLD = 3   # important for stationary tablets

prev_time = 0

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
    results = model(frame, imgsz=256, conf=0.25, verbose=False)[0]

    # =====================
    # DETECTIONS
    # =====================
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": results.names[cls_id],
            "confidence": conf
        })

    # =====================
    # PROCESS LOGIC
    # =====================
    result = process(detections, batch_id)

    # =====================
    # SAFE TABLET COUNTING (IMPORTANT PART)
    # =====================
    tablet_present = result["tablet"] > 0

    if tablet_present:
        stable_frames += 1
    else:
        stable_frames = 0
        prev_tablet_present = False

    # count ONLY when:
    # - tablet is present
    # - stable for a few frames
    # - and it was previously NOT counted
    if stable_frames >= STABLE_THRESHOLD and not prev_tablet_present:
        tablet_counter += 1
        prev_tablet_present = True

    # =====================
    # BATCH LOGIC
    # =====================
    if tablet_counter >= TABLETS_PER_BATCH:
        batch_id = batch_manager.new_batch()
        print(f"[INFO] NEW BATCH: {batch_id}")
        tablet_counter = 0

    # =====================
    # ASYNC INFLUX
    # =====================
    influx.write(result)

    # =====================
    # DISPLAY
    # =====================
    annotated = results.plot()

    cv2.putText(annotated, f"Batch: {batch_id}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(annotated, f"Tablets: {tablet_counter}/{TABLETS_PER_BATCH}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(annotated, f"Status: {result['status']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if result["status"] == "PASS" else (0, 0, 255), 2)

    # =====================
    # FPS
    # =====================
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("QC PIPELINE", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
influx.stop()