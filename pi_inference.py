import cv2
import time
import numpy as np
import socket
import json
from datetime import datetime
from ultralytics import YOLO

from processing.analyzer import process
from influx_worker import InfluxWorker

# =====================
# CONFIG
# =====================
WIDTH, HEIGHT = 640, 480
IMGSZ1, IMGSZ2 = 256, 256
CONF1, CONF2 = 0.5, 0.2
FRAME_SKIP = 2

# =====================
# MODELS
# =====================
stage1 = YOLO("models/new_m_1.pt")
stage2 = YOLO("models/new_m_2.pt")

print("Stage2 mapping:", stage2.names)

# =====================
# SOCKET LOGGING
# =====================
PC_IP = "192.168.100.175"
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)

try:
    sock.connect((PC_IP, PORT))
    print("[LOG] Connected to PC")
except:
    print("[LOG] PC not reachable")
    sock = None


def send_log(payload):
    if not sock:
        return
    try:
        sock.sendall((json.dumps(payload) + "\n").encode())
    except:
        pass

# =====================
# CAMERA (IP CAM SAFE)
# =====================
# IP_URL = "http://192.168.100.6:8080/video"
# cap = cv2.VideoCapture(IP_URL,cv2.CAP_FFMPEG)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Force highest available resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# IMPORTANT: reduce auto adjustment issues
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual-ish mode (depends on camera)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)        # adjust manually if supported

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)        # disable autofocus (VERY IMPORTANT)

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
seen = set()
tablet_cache = {}

batch_id = 1
batch_limit = 40

pass_count = 0
fail_count = 0

prev = time.time()

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
    infer = cv2.resize(frame, (WIDTH, HEIGHT))

    # FPS
    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now

    # =====================
    # STAGE 1
    # =====================
    r1 = stage1.track(infer, persist=True, imgsz=IMGSZ1, conf=CONF1, verbose=False)[0]

    tablets = []

    if r1.boxes is not None and r1.boxes.id is not None:

        for box, cls, conf, tid in zip(
            r1.boxes.xyxy,
            r1.boxes.cls,
            r1.boxes.conf,
            r1.boxes.id
        ):

            cls = int(cls)
            if cls not in [0, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            sx = display.shape[1] / infer.shape[1]
            sy = display.shape[0] / infer.shape[0]

            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

            cv2.rectangle(display, (x1, y1), (x2, y2), (255,255,0), 2)

            tablets.append((int(tid), x1, y1, x2, y2))

    detections = []

    # =====================
    # STAGE 2 (SEGMENTATION)
    # =====================
    for tid, x1, y1, x2, y2 in tablets:

        if tid in tablet_cache:
            continue

        m = 0.05
        w, h = x2-x1, y2-y1

        x1e = max(0, int(x1 - m*w))
        y1e = max(0, int(y1 - m*h))
        x2e = min(display.shape[1], int(x2 + m*w))
        y2e = min(display.shape[0], int(y2 + m*h))

        crop = frame[y1e:y2e, x1e:x2e]   # IMPORTANT: RAW FRAME

        status = "PASS"
        defect = None

        if crop.size:

            r2 = stage2(crop, imgsz=IMGSZ2, conf=CONF2, verbose=False)

            for res in r2:

                if res.masks is None:
                    continue

                for mask, cls_id, conf in zip(
                    res.masks.data,
                    res.boxes.cls,
                    res.boxes.conf
                ):

                    name = stage2.names[int(cls_id)]

                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

                    full = np.zeros(display.shape[:2], dtype=np.uint8)
                    full[y1e:y2e, x1e:x2e] = (mask > 0.5).astype(np.uint8)

                    if name == "chip":
                        color = (0,255,0)
                        status = "FAIL"
                        defect = "chip"

                    elif name == "cap":
                        color = (0,0,255)
                        status = "FAIL"
                        defect = "cap"
                    else:
                        continue

                    overlay = display.copy()
                    overlay[full > 0] = color
                    display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

        tablet_cache[tid] = {"status": status, "defect": defect}

        if tid not in seen:
            seen.add(tid)

            if status == "PASS":
                pass_count += 1
            else:
                fail_count += 1

            # =====================
            # LIVE TABLET LOG 🔥
            # =====================
            send_log({
                "event": "TABLET_RESULT",
                "timestamp": datetime.now().isoformat(),
                "batch": batch_id,
                "track_id": tid,
                "status": status,
                "defect": defect,
                "pass": pass_count,
                "fail": fail_count
            })

            detections.append({"status": status, "defect": defect})

    # =====================
    # INFLUX (SAFE)
    # =====================
    if detections:
        result = process(detections)
        result["batch_id"] = batch_id
        influx.write(result)

    # =====================
    # BATCH LOG
    # =====================
    total = pass_count + fail_count
    status = "PASS" if fail_count == 0 else "FAIL"

    if total >= batch_limit:

        send_log({
            "event": "BATCH_COMPLETE",
            "batch": batch_id,
            "status": status,
            "pass": pass_count,
            "fail": fail_count,
            "fps": round(fps, 2)
        })

        batch_id += 1
        pass_count = 0
        fail_count = 0
        seen.clear()
        tablet_cache.clear()

    # =====================
    # UI
    # =====================
    cv2.putText(display, f"B:{batch_id} P:{pass_count} F:{fail_count}",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(display, f"{status} | {int(fps)} FPS",
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0) if status=="PASS" else (0,0,255), 1)

    cv2.imshow("QC PIPELINE", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
influx.stop()
send_log({"event": "SYSTEM_STOP"})