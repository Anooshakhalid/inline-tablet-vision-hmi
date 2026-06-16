import cv2
import time
import numpy as np
import socket
import json
from datetime import datetime
from ultralytics import YOLO
import os

from processing.analyzer import process
from influx_worker import InfluxWorker

# =====================
# CONFIG
# =====================
WIDTH, HEIGHT = 640, 480
IMGSZ1, IMGSZ2 = 320, 320
CONF1 = 0.5
CONF2 = 0.25
FRAME_SKIP = 2

# =====================
# MODELS
# =====================
stage1 = YOLO("models/new_m_1.pt")
stage2 = YOLO("models/new_m_2.pt")

# =====================
# SOCKET (SAFE + RECONNECT)
# =====================
PC_IP = "192.168.100.175"
PORT = 9999


def connect_socket():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((PC_IP, PORT))
        print("[LOG] Connected")
        return s
    except:
        print("[LOG] Server not reachable")
        return None


log_socket = connect_socket()


def send_log(event, data):
    global log_socket

    payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        **data
    }

    try:
        if log_socket:
            log_socket.sendall((json.dumps(payload) + "\n").encode())
    except:
        log_socket = connect_socket()

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
def get_last_batch_id_from_log(file_path="qc_logs.txt"):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return 1

        # get last non-empty line
        last_line = lines[-1].strip()

        if not last_line:
            return 1

        data = json.loads(last_line)

        return int(data.get("batch", 1))

    except Exception as e:
        print("[LOG] Failed to read batch from log:", e)
        return 1
    

frame_count = 0
tablet_results = {}

batch_id = get_last_batch_id_from_log() + 1
batch_limit = 40

pass_count = 0
fail_count = 0

prev_time = time.time()

# IMPORTANT FIX: real tracking memory
seen_ids = set()

# =====================
# TRANSPARENT MASK (FIXED PROPERLY)
# =====================
def apply_mask(frame, mask, color, alpha=0.25):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# =====================
# LOOP
# =====================
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    display = frame.copy()

    # FPS
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now

    # =====================
    # STAGE 1 (TABLET DETECTION + TRACKING)
    # =====================
    results1 = stage1.track(
        frame,
        persist=True,
        imgsz=320,
        conf=0.5,
        device="cpu",
        verbose=False
    )[0]

    tablets = []

    if results1.boxes is not None and results1.boxes.id is not None:

        for box, cls, tid in zip(
            results1.boxes.xyxy,
            results1.boxes.cls,
            results1.boxes.id
        ):

            if int(cls) not in [0, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            tid = int(tid)

            tablets.append((tid, x1, y1, x2, y2))

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # =====================
    # STAGE 2 (SEGMENTATION - ALL TABLETS)
    # =====================
    detections = []

    for (tid, x1, y1, x2, y2) in tablets:

        m = 0.05
        w, h = x2 - x1, y2 - y1

        x1e = max(0, int(x1 - m*w))
        y1e = max(0, int(y1 - m*h))
        x2e = min(frame.shape[1], int(x2 + m*w))
        y2e = min(frame.shape[0], int(y2 + m*h))

        crop = frame[y1e:y2e, x1e:x2e]

        status = "PASS"
        defect = None

        if crop.size != 0:

            results2 = stage2(crop, imgsz=IMGSZ2, conf=CONF2, verbose=False)

            for r2 in results2:

                if r2.masks is None:
                    continue

                for mask_t, cls_t in zip(r2.masks.data, r2.boxes.cls):

                    name = stage2.names[int(cls_t)]

                    mask = mask_t.cpu().numpy()
                    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)

                    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    full_mask[y1e:y2e, x1e:x2e] = mask

                    if name == "chip":
                        status = "FAIL"
                        defect = "chip"
                        color = (0, 255, 0)

                    elif name == "cap":
                        status = "FAIL"
                        defect = "cap"
                        color = (0, 0, 255)

                    else:
                        continue

                    display = apply_mask(display, full_mask, color, alpha=0.22)

        # =====================
        # FIX: COUNT ONLY ONCE PER TRACK ID
        # =====================
        if tid not in tablet_results:
            tablet_results[tid] = status
        else:
            # upgrade status if failure appears later
            if status == "FAIL":
                tablet_results[tid] = "FAIL"

        detections.append({
            "track_id": tid,
            "status": status,
            "defect": defect
        })

        # =====================
        # LIVE LOG (UNCHANGED FORMAT)
        # =====================
        send_log("TABLET_RESULT", {
            "batch": batch_id,
            "track_id": tid,
            "status": status,
            "defect": defect,
            "pass": pass_count,
            "fail": fail_count
        })

    # =====================
    # INFLUX
    # =====================
    pass_count = sum(1 for v in tablet_results.values() if v == "PASS")
    fail_count = sum(1 for v in tablet_results.values() if v == "FAIL")
    result = {
        "batch_id": batch_id,
        "total": pass_count + fail_count,
        "pass": pass_count,
        "fail": fail_count,
        "chip": sum(1 for d in detections if d.get("defect") == "chip"),
        "cap": sum(1 for d in detections if d.get("defect") == "cap"),
        "status": "FAIL" if fail_count > 0 else "PASS"
    }
    result["batch_id"] = batch_id
    influx.write(result)

    # =====================
    # BATCH LOGIC
    # =====================
    total = len(tablet_results)

    if total >= batch_limit:

        send_log("BATCH_UPDATE", {
            "batch": batch_id,
            "pass": pass_count,
            "fail": fail_count,
            "fps": fps
        })

        batch_id += 1
        pass_count = 0
        fail_count = 0
        seen_ids.clear()

    # =====================
    # UI (FULL FIXED)
    # =====================
    cv2.putText(display,
                f"BATCH: {batch_id}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(display,
                f"PASS: {pass_count}  FAIL: {fail_count}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if fail_count == 0 else (0, 0, 255),
                2)

    cv2.putText(display,
                f"FPS: {int(fps)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

    cv2.imshow("QC PIPELINE", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
influx.stop()