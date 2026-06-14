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
# SOCKET
# =====================
PC_IP = "192.168.100.175"
PORT = 9999

log_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
log_socket.settimeout(1)

try:
    log_socket.connect((PC_IP, PORT))
    print("[LOG] Connected")
except:
    print("[LOG] Not connected")
    log_socket = None


def send_log(event, data):
    if not log_socket:
        return
    try:
        payload = {"event": event, **data, "ts": time.time()}
        log_socket.sendall((json.dumps(payload) + "\n").encode())
    except:
        pass


# =====================
# CAMERA
# =====================
IP_URL = "http://192.168.100.6:8080/video"
cap = cv2.VideoCapture(IP_URL,cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)

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
seen_ids = set()
tablet_results = {}

batch_id = 1
batch_limit = 40

pass_count = 0
fail_count = 0

prev_time = time.time()

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
    # STAGE 1
    # =====================
    results1 = stage1(frame, imgsz=IMGSZ1, conf=CONF1, verbose=False)

    tablets = []

    for r in results1:
        if r.boxes is None:
            continue

        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):

            if int(cls) not in [0, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            tablets.append((x1, y1, x2, y2))

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)

    detections = []

    # =====================
    # STAGE 2 (IMPORTANT FIXED)
    # =====================
    for (x1, y1, x2, y2) in tablets:

        m = 0.05
        w, h = x2 - x1, y2 - y1

        x1e = max(0, int(x1 - m * w))
        y1e = max(0, int(y1 - m * h))
        x2e = min(frame.shape[1], int(x2 + m * w))
        y2e = min(frame.shape[0], int(y2 + m * h))

        crop = frame[y1e:y2e, x1e:x2e]

        if crop.size == 0:
            continue

        results2 = stage2(crop, imgsz=IMGSZ2, conf=CONF2, verbose=False)

        status = "PASS"
        defect = None

        for r2 in results2:
            if r2.masks is None:
                continue

            for mask_tensor, cls_tensor, conf_tensor in zip(
                r2.masks.data,
                r2.boxes.cls,
                r2.boxes.conf
            ):

                name = stage2.names[int(cls_tensor)]

                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

                full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                full_mask[y1e:y2e, x1e:x2e] = (mask > 0.5).astype(np.uint8)

                if name in ["chip", "cap"]:
                    status = "FAIL"
                    defect = name

                    color = (0, 255, 0) if name == "chip" else (0, 0, 255)

                    # IMPORTANT: SAME AS YOUR WORKING FILE
                    display[full_mask > 0] = color

        # store result
        detections.append({"status": status, "defect": defect})

    # =====================
    # ANALYTICS
    # =====================
    result = process(detections)
    result["batch_id"] = batch_id
    influx.write(result)

    # =====================
    # BATCH + LIVE LOG
    # =====================
    total = pass_count + fail_count
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

    # =====================
    # DISPLAY
    # =====================
    cv2.putText(display, f"FPS:{int(fps)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("QC PIPELINE", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
influx.stop()