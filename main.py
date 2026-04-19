import time

from config.setting import FRAME_LIMIT
from config.setting import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET
from detection.model import get_detections
from processing.analyzer import process
from display.hmi import show
from database.db import save_to_influx
from utils.batch_manager import BatchManager

batch_manager = BatchManager()
batch_id = batch_manager.new_batch()

frame_count = 0

while True:

    # STEP 1: fake detections
    detections = get_detections()

    # STEP 2: process logic
    result = process(detections, batch_id)

    # STEP 3: save to InfluxDB (for Grafana)
    save_to_influx(result)

    # STEP 4: show HMI
    show(result)

    print(result)

    # STEP 5: batch control (CLEAN + CONFIG-BASED)
    frame_count += 1

    if frame_count >= FRAME_LIMIT:
        frame_count = 0
        batch_id = batch_manager.new_batch()
        print(f"\nNEW BATCH: {batch_id}\n")

    time.sleep(0.5)