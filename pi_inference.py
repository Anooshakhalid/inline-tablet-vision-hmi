import cv2
import time
import socket
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
# PC LOGGING SETUP
# =====================
PC_IP = "192.168.100.175"   
PORT = 9999

log_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
log_socket.settimeout(0.5)

try:
    log_socket.connect((PC_IP, PORT))
    print("[LOG] Connected to PC")
except:
    print("[LOG] PC not available")
    log_socket = None


def send_log(msg):
    if log_socket:
        try:
            log_socket.send((msg + "\n").encode())
        except:
            pass


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
send_log("SYSTEM: Camera started")

influx = InfluxWorker()
influx.start()

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
    # YOLO DETECTION ONLY
    # =====================
    results = model(frame, imgsz=256, conf=0.25, verbose=False)[0]

    detections = []

    total = 0
    pass_count = 0
    fail_count = 0

    if results.boxes is not None:
        total = len(results.boxes)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]

            detections.append({
                "class": label
            })

            if label.upper() == "PASS":
                pass_count += 1
            else:
                fail_count += 1

    # =====================
    # ANALYTICS
    # =====================
    result = process(detections)
    influx.write(result)

    # =====================
    # SEND LOGS TO PC
    # =====================
    send_log(
        f"TOTAL:{total} | PASS:{pass_count} | FAIL:{fail_count} | STATUS:{result['status']}"
    )

    # =====================
    # FPS
    # =====================
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # =====================
    # DISPLAY
    # =====================
    annotated = results.plot()

    cv2.putText(annotated, f"TOTAL: {total}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(annotated, f"PASS: {pass_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(annotated, f"FAIL: {fail_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 95),
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
send_log("SYSTEM: Stopped")