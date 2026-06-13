import cv2
import time
import logging
from ultralytics import YOLO

from processing.analyzer import process
from utils.batch_manager import BatchManager
from influx_worker import InfluxWorker

# =====================
# LOGGING (FORCE FIX)
# =====================
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("qc_logs.txt", mode="a")
formatter = logging.Formatter("%(asctime)s | %(message)s")
file_handler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(file_handler)

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240
MODEL_PATH = "models/model.pt"

LINE_X = WIDTH // 2

# cooldown zone (VERY IMPORTANT in real systems)
COOLDOWN_DISTANCE = 25

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
tracked = {}         # object_id -> (cx, cy)
last_side = {}       # object_id -> LEFT / RIGHT
counted = set()      # already counted objects
next_id = 0

count = 0
prev_time = 0


# =====================
# SIMPLE TRACKING
# =====================
def match(prev, curr):
    global next_id

    matched = []

    for cx, cy in curr:
        assigned = False

        for oid, (px, py) in prev.items():
            if abs(cx - px) < 40 and abs(cy - py) < 40:
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

    results = model(frame, imgsz=256, conf=0.25, verbose=False)[0]

    detections = []
    centers = []

    # =====================
    # EXTRACT DETECTIONS
    # =====================
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        centers.append((cx, cy))

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
    matched = match(tracked, centers)

    new_tracked = {}

    for oid, cx, cy in matched:

        prev_x = tracked.get(oid, (cx, cy))[0]

        # LEFT / RIGHT SIDE
        current_side = "LEFT" if cx < LINE_X else "RIGHT"

        # store side memory
        prev_side = last_side.get(oid, current_side)

        # =====================
        # REAL CONVEYOR LOGIC
        # =====================
        if prev_side == "LEFT" and current_side == "RIGHT":

            if oid not in counted:
                count += 1
                counted.add(oid)

                logger.info(f"COUNTED ID:{oid} TOTAL:{count}")

        # update state
        new_tracked[oid] = (cx, cy)
        last_side[oid] = current_side

    tracked = new_tracked

    # =====================
    # BATCH RESET
    # =====================
    if count >= 50:
        batch_id = batch_manager.new_batch()
        logger.info(f"NEW BATCH: {batch_id}")

        print(f"[INFO] NEW BATCH: {batch_id}")

        count = 0
        counted.clear()
        tracked.clear()
        last_side.clear()

    # =====================
    # INFLUX
    # =====================
    influx.write(result)

    logger.info(
        f"Batch:{batch_id} | Total:{result['total']} | "
        f"Chip:{result['chip']} | Cap:{result['cap']} | Status:{result['status']}"
    )

    # =====================
    # VISUALS
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