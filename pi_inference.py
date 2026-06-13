import cv2
import time
from ultralytics import YOLO

from processing.analyzer import process
from influx_worker import InfluxWorker

# =====================
# CONFIG
# =====================
WIDTH = 320
HEIGHT = 240
MODEL_PATH = "models/model.pt"

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

influx = InfluxWorker()
influx.start()

# =====================
# STATE (IMPORTANT)
# =====================
seen_ids = set()

total_count = 0
pass_count = 0
fail_count = 0

prev_time = 0

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # YOLO tracking ON
    results = model.track(frame, persist=True, imgsz=256, conf=0.25, verbose=False)[0]

    detections = []

    if results.boxes is not None:
        for box in results.boxes:

            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()

            # always send to analyzer
            detections.append({
                "class": label,
                "confidence": float(box.conf[0])
            })

            # =====================
            # UNIQUE OBJECT COUNTING
            # =====================
            if box.id is None:
                continue

            oid = int(box.id[0])

            if oid in seen_ids:
                continue

            seen_ids.add(oid)
            total_count += 1

            # =====================
            # PASS / FAIL (production)
            # =====================
            if label == "pass":
                pass_count += 1
            else:
                fail_count += 1

    # =====================
    # ANALYTICS (FRAME LEVEL)
    # =====================
    result = process(detections)
    influx.write(result)

    # =====================
    # DEBUG OUTPUT
    # =====================
    print(
        f"TOTAL:{total_count} | "
        f"PASS:{pass_count} | "
        f"FAIL:{fail_count} | "
        f"FRAME_STATUS:{result['status']}"
    )

    # =====================
    # VISUALIZATION
    # =====================
    annotated = results.plot()

    cv2.putText(annotated, f"TOTAL: {total_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(annotated, f"PASS: {pass_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(annotated, f"FAIL: {fail_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(annotated, f"STATUS: {result['status']}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if result["status"] == "PASS" else (0, 0, 255), 2)

    # =====================
    # FPS
    # =====================
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("QC PIPELINE", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
influx.stop()