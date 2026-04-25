import cv2
import time
import threading
from flask import Flask, Response
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/best.pt"
FRAME_SIZE = 320
DEVICE = "cpu"

app = Flask(__name__)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# 🔥 FIX CAMERA BUFFER (important for smooth video)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] System Started")

latest_frame = None
lock = threading.Lock()

# =========================
# INFERENCE THREAD
# =========================
def inference_loop():
    global latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # FIX: stable resize
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)

        results = model(frame, imgsz=320, device=DEVICE)
        r = results[0]

        annotated = frame.copy()

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # FIX: safe frame update
        with lock:
            latest_frame = annotated

        # FIX: stable FPS (~30)
        time.sleep(0.03)

# =========================
# STREAM GENERATOR
# =========================
def generate():
    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # FIX: better encoding quality (reduces flicker)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.03)

# =========================
# FLASK ROUTE
# =========================
@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# START SYSTEM
# =========================
if __name__ == "__main__":

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)