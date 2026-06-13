import cv2
import time
import numpy as np
import socket
from ultralytics import YOLO

from processing.analyzer import process
from influx_worker import InfluxWorker

# =====================
# CONFIG
# =====================
WIDTH = 640
HEIGHT = 480

IMGSZ1 = 256
IMGSZ2 = 256

CONF1 = 0.5
CONF2 = 0.2

FRAME_SKIP = 2

# =====================
# MODELS
# =====================
stage1 = YOLO("models/new_m_1.pt")
stage2 = YOLO("models/new_m_2.pt")

# =====================
# LOG SOCKET
# =====================
PC_IP = "192.168.100.175"
PORT = 9999

log_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
log_socket.settimeout(0.5)

try:
    log_socket.connect((PC_IP, PORT))
    print("[LOG] Connected to PC")
except:
    print("[LOG] PC not reachable")
    log_socket = None


def send_log(msg):
    if log_socket:
        try:
            log_socket.send((msg + "\n").encode())
        except:
            pass


# =====================
# CAMERA
# =====================
# cap = cv2.VideoCapture("http://192.168.1.9:8080/video")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =====================
# INFLUX
# =====================
influx = InfluxWorker()
influx.start()

# =====================
# STATE
# =====================
frame_count = 0

tablet_results = {}   # oid -> result (RUN ONLY ONCE)
seen_ids = set()

batch_id = 1
batch_limit = 40

pass_count = 0
fail_count = 0

prev_time = time.time()

# =====================
# MAIN LOOP
# =====================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # =====================
    # STAGE 1 - DETECT TABLETS
    # =====================
    results1 = stage1.track(
        frame,
        persist=True,
        imgsz=IMGSZ1,
        conf=CONF1,
        verbose=False
    )[0]

    tablets = []

    if results1.boxes is not None:

        for box, cls in zip(results1.boxes.xyxy, results1.boxes.cls):

            cls_id = int(cls)

            # only tablet or normal
            if cls_id not in [0, 3]:
                continue

            if results1.boxes.id is None:
                continue

            oid = int(results1.boxes.id[0])

            x1, y1, x2, y2 = map(int, box.tolist())
            tablets.append((oid, x1, y1, x2, y2))

    detections = []

    # =====================
    # STAGE 2 - PROCESS EACH TABLET ONCE
    # =====================
    for (oid, x1, y1, x2, y2) in tablets:

        if oid in tablet_results:
            continue

        # crop with margin
        m = 0.05
        w, h = x2 - x1, y2 - y1

        x1e = max(0, int(x1 - m * w))
        y1e = max(0, int(y1 - m * h))
        x2e = min(frame.shape[1], int(x2 + m * w))
        y2e = min(frame.shape[0], int(y2 + m * h))

        crop = frame[y1e:y2e, x1e:x2e]

        tablet_status = "PASS"
        defect_type = None

        if crop.size != 0:

            results2 = stage2(
                crop,
                imgsz=IMGSZ2,
                conf=CONF2,
                verbose=False
            )

            for r2 in results2:

                if r2.boxes is None:
                    continue

                for cls_tensor in r2.boxes.cls:

                    cls_id = int(cls_tensor)

                    if cls_id == 0:
                        tablet_status = "FAIL"
                        defect_type = "chip"

                    elif cls_id == 1:
                        tablet_status = "FAIL"
                        defect_type = "cap"

        # store result once
        tablet_results[oid] = {
            "status": tablet_status,
            "defect": defect_type
        }

        # counting
        if oid not in seen_ids:

            seen_ids.add(oid)

            if tablet_status == "PASS":
                pass_count += 1
            else:
                fail_count += 1

            detections.append({
                "status": tablet_status,
                "defect": defect_type
            })

    # =====================
    # ANALYTICS
    # =====================
    result = process(detections)
    influx.write(result)

    # =====================
    # BATCH LOGIC
    # =====================
    total_count = pass_count + fail_count

    if total_count >= batch_limit:

        batch_id += 1
        pass_count = 0
        fail_count = 0
        seen_ids.clear()
        tablet_results.clear()

        send_log(f"NEW_BATCH:{batch_id}")
        print(f"[BATCH] STARTED {batch_id}")

    # =====================
    # FPS
    # =====================
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # =====================
    # CLEAN OVERLAY (NO IDS)
    # =====================
    status = "PASS" if fail_count == 0 else "FAIL"
    color = (0, 255, 0) if status == "PASS" else (0, 0, 255)

    cv2.putText(frame,
                f"B:{batch_id} P:{pass_count} F:{fail_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)

    cv2.putText(frame,
                f"{status} | {int(fps)} FPS",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1)

    cv2.imshow("QC PIPELINE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
influx.stop()
send_log("SYSTEM: STOPPED")