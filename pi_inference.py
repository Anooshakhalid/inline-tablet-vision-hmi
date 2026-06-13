import cv2
import time
import logging
from ultralytics import YOLO

from processing.analyzer import process
from utils.batch_manager import BatchManager
from influx_worker import InfluxWorker

# =====================
# LOGGING
# =====================
logging.basicConfig(
    filename="qc_logs.txt",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240
MODEL_PATH = "models/model.pt"

LINE_X = WIDTH // 2          # conveyor counting line
DIST_THRESHOLD = 40          # tracking tolerance

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
logging.info("Camera started")

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

influx = InfluxWorker()
influx.start()

# =====================
# STATE
# =====================
tracked_objects = {}   # id -> (cx, cy)
next_id = 0
count = 0

prev_time = 0


# =====================
# SIMPLE TRACKING MATCH
# =====================
def match_objects(prev, curr):
    global next_id

    matched = []

    for cx, cy in curr:
        assigned = False

        for oid, (px, py) in prev.items():
            if abs(cx - px) < DIST_THRESHOLD and abs(cy - py) < DIST_THRESHOLD:
                matched.append((oid, cx, cy))
                assigned = True
                break

        if not assigned:
            next_id += 1
            matched.append((next_id, cx, cy))

    return matched


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
    # GET CENTERS
    # =====================
    centers = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        centers.append((cx, cy))

    # =====================
    # PROCESS LOGIC
    # =====================
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": results.names[cls_id],
            "confidence": conf
        })

    result = process(detections, batch_id)

    # =====================
    # TRACK OBJECTS
    # =====================
    matched = match_objects(tracked_objects, centers)

    new_state = {}

    for oid, cx, cy in matched:

        prev_x = tracked_objects.get(oid, (cx, cy))[0]

        # =====================
        # CONVEYOR CROSSING RULE
        # =====================
        if prev_x < LINE_X and cx >= LINE_X:
            count += 1
            logging.info(f"COUNTED OBJECT ID:{oid} | TOTAL:{count}")

        new_state[oid] = (cx, cy)

    tracked_objects = new_state

    # =====================
    # BATCH LOGIC
    # =====================
    if count >= 50:
        batch_id = batch_manager.new_batch()
        logging.info(f"NEW BATCH: {batch_id}")
        print(f"[INFO] NEW BATCH: {batch_id}")
        count = 0

    # =====================
    # INFLUX (ASYNC)
    # =====================
    influx.write(result)

    # =====================
    # LOG FRAME DATA
    # =====================
    logging.info(
        f"Batch:{batch_id} | "
        f"Total:{result['total']} | "
        f"Chip:{result['chip']} | "
        f"Cap:{result['cap']} | "
        f"Status:{result['status']}"
    )

    # =====================
    # VISUALIZATION
    # =====================
    annotated = results.plot()

    # conveyor line
    cv2.line(annotated, (LINE_X, 0), (LINE_X, HEIGHT), (0, 255, 255), 2)

    cv2.putText(annotated, f"COUNT: {count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(annotated, f"Batch: {batch_id}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
logging.info("System stopped")