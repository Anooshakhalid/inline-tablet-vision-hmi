from influxdb_client_3 import InfluxDBClient3, Point
import time
import os

# Load from environment variables (.env)
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

client = InfluxDBClient3(
    host=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

def save_to_influx(batch_id, tablet_count, defect_count):

    point = Point("tablet_detection") \
        .tag("batch_id", batch_id) \
        .field("tablet_count", tablet_count) \
        .field("defect_count", defect_count) \
        .time(time.time_ns())

    client.write(database=INFLUX_BUCKET, record=point)