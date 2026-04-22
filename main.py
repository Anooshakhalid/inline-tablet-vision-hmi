import time

from config.setting import FRAME_LIMIT
from detection.model import get_detections
from processing.analyzer import process
from display.hmi import show
from database.db import save_to_influx
from utils.batch_manager import BatchManager
from camera.capture import get_frame


batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

frame_count = 0

while True:

    # STEP 1: get REAL camera frame 
    frame = get_frame()

    if frame is None:
        print("Camera error")
        continue

    # STEP 2: run YOLO model 
    detections = get_detections(frame)

    # STEP 3: process logic
    result = process(detections, batch_id)

    # STEP 4: save to InfluxDB
    save_to_influx(result)

    # STEP 5: show HMI
    show(result)

    print(result)

    # STEP 6: batch control
    frame_count += 1

    if frame_count >= FRAME_LIMIT:
        frame_count = 0
        batch_id = batch_manager.new_batch()
        print(f"\nNEW BATCH: {batch_id}\n")

    time.sleep(0.5)