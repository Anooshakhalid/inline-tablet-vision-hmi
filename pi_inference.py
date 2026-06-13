import cv2
import time
import logging
import os
from ultralytics import YOLO

from processing.analyzer import process
from utils.batch_manager import BatchManager
from influx_worker import InfluxWorker

# =====================
# LOGGING (FIXED PATH)
# =====================
LOG_FILE = os.path.join(os.path.dirname(__file__), "qc_logs.txt")

logger = logging.getLogger("qc")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240
MODEL_PATH = "models/model.pt"

LINE_X = WIDTH // 2
COOLDOWN_TIME = 1.0  # seconds

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
logger.info("Camera started")

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

influx = InfluxWorker()
influx.start()

# =====================
# STATE
# =====================
last_side = {}       # id -> side
counted = set()      # ids already counted in batch
cooldown = {}        # id -> last count time

count = 0
prev_time = 0

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # 🔥 YOLO TRACKING (IMPORTANT FIX)
    results = model.track(frame, persist=True, imgsz=256, conf=0.25, verbose=False)[0]

    detections = []

    # =====================
    # PROCESS DETECTIONS
    # =====================
    if results.boxes is not None:
        for box in results.boxes:
            if box.id is None:
                continue

            oid = int(box.id[0])

            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": results.names[cls_id],
                "confidence": conf
            })

            # =====================
            # SIDE DETECTION
            # =====================
            current_side = "LEFT" if cx < LINE_X else "RIGHT"
            prev_side = last_side.get(oid, current_side)

            # =====================
            # COOLDOWN CHECK (FIX DOUBLE COUNTING)
            # =====================
            now = time.time()
            if oid in cooldown and (now - cooldown[oid]) < COOLDOWN_TIME:
                continue

            # =====================
            # COUNT LOGIC
            # =====================
            if prev_side == "LEFT" and current_side == "RIGHT":
                if oid not in counted:
                    count += 1
                    counted.add(oid)
                    cooldown[oid] = now

                    logger.info(f"COUNTED ID:{oid} TOTAL:{count}")

            last_side[oid] = current_side

    # =====================
    # ANALYTICS
    # =====================
    result = process(detections, batch_id)

    influx.write(result)

    logger.info(
        f"Batch:{batch_id} | Total:{result['total']} | "
        f"Chip:{result['chip']} | Cap:{result['cap']} | Status:{result['status']}"
    )

    # =====================
    # BATCH RESET
    # =====================
    if count >= 50:
        batch_id = batch_manager.new_batch()
        logger.info(f"NEW BATCH: {batch_id}")

        print(f"[INFO] NEW BATCH: {batch_id}")

        count = 0
        counted.clear()
        last_side.clear()
        cooldown.clear()

    # =====================
    # VISUALIZATION
    # =====================
    annotated = results.plot()

    cv2.line(annotated, (LINE_X, 0), (LINE_X, HEIGHT), (0, 255, 255), 2)

    cv2.putText(annotated, f"COUNT: {count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(annotated, f"Batch: {batch_id}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(annotated, f"Status: {result['status']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if result["status"] == "PASS" else (0, 0, 255), 2)

    # FPS
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
logger.info("System stopped")