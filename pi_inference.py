import cv2
import time
import numpy as np
import socket
import json
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
# SOCKET LOGGING (SAFE)
# =====================
PC_IP = "192.168.100.175"
PORT = 9999

log_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
log_socket.settimeout(0.2)

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
IP_URL = "http://192.168.100.6:8080/video"

cap = cv2.VideoCapture(IP_URL, cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
tablet_results = {}
seen_ids = set()

batch_id = 1
batch_limit = 40

pass_count = 0
fail_count = 0

prev_time = time.time()

# =====================
# MAIN LOOP (ONLY ONE)
# =====================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    display_frame = frame.copy()
    infer_frame = cv2.resize(frame, (640, 480))

    # =====================
    # STAGE 1 - TABLET DETECTION
    # =====================
    results1 = stage1.track(
        infer_frame,
        persist=True,
        imgsz=IMGSZ1,
        conf=CONF1,
        verbose=False
    )[0]

    tablets = []

    if results1.boxes is not None:

        for box, cls, conf in zip(
            results1.boxes.xyxy,
            results1.boxes.cls,
            results1.boxes.conf
        ):

            cls_id = int(cls)

            if cls_id not in [0, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            confidence = float(conf)

            # scale back to display frame
            scale_x = display_frame.shape[1] / infer_frame.shape[1]
            scale_y = display_frame.shape[0] / infer_frame.shape[0]

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # =====================
            # STAGE 1 VISUAL
            # =====================
            label = f"{stage1.names[cls_id]} {confidence*100:.1f}%"

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

            if results1.boxes.id is not None:
                oid = int(results1.boxes.id[0])
                tablets.append((oid, x1, y1, x2, y2))

    detections = []

    # =====================
    # STAGE 2 - DEFECT DETECTION
    # =====================
    for (oid, x1, y1, x2, y2) in tablets:

        if oid in tablet_results:
            continue

        m = 0.05
        w, h = x2 - x1, y2 - y1

        x1e = max(0, int(x1 - m * w))
        y1e = max(0, int(y1 - m * h))
        x2e = min(display_frame.shape[1], int(x2 + m * w))
        y2e = min(display_frame.shape[0], int(y2 + m * h))

        crop = display_frame[y1e:y2e, x1e:x2e]

        tablet_status = "PASS"
        defect_type = None

        if crop.size != 0:

            results2 = stage2(
                crop,
                imgsz=IMGSZ2,
                conf=CONF2,
                verbose=False
            )

            results2 = stage2(crop, imgsz=IMGSZ2, conf=CONF2, verbose=False)

            for r2 in results2:

                if r2.boxes is None:
                    continue

                for box, cls, conf in zip(r2.boxes.xyxy, r2.boxes.cls, r2.boxes.conf):

                    cls_id = int(cls)
                    conf_pct = float(conf) * 100

                    defect_name = stage2.names[cls_id]

                    # IMPORTANT: PASS / FAIL LOGIC
                    if defect_name == "chip":
                        tablet_status = "FAIL"
                        color = (0, 255, 0)

                    elif defect_name == "cap":
                        tablet_status = "FAIL"
                        color = (0, 0, 255)

                    x1, y1, x2, y2 = map(int, box.tolist())

                    cv2.rectangle(crop, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        crop,
                        f"{defect_name} {conf_pct:.1f}%",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

        tablet_results[oid] = {
            "status": tablet_status,
            "defect": defect_type
        }

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

    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    status = "PASS" if fail_count == 0 else "FAIL"

    if total_count >= batch_limit:

        log_data = {
            "event": "BATCH_UPDATE",
            "batch": batch_id,
            "status": status,
            "pass": pass_count,
            "fail": fail_count,
            "fps": round(fps, 2)
        }

        send_log(json.dumps(log_data))

        batch_id += 1
        pass_count = 0
        fail_count = 0
        seen_ids.clear()
        tablet_results.clear()

    # =====================
    # UI TEXT
    # =====================
    cv2.putText(
        display_frame,
        f"B:{batch_id} P:{pass_count} F:{fail_count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display_frame,
        f"{status} | {int(fps)} FPS",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if status == "PASS" else (0, 0, 255),
        1
    )

    cv2.imshow("QC PIPELINE", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
influx.stop()
send_log("SYSTEM: STOPPED")