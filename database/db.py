from influxdb_client_3 import InfluxDBClient3, Point
from config.setting import INFLUX_URL, INFLUX_TOKEN, INFLUX_BUCKET
from datetime import datetime

client = InfluxDBClient3(
    host=INFLUX_URL,
    token=INFLUX_TOKEN,
)

def save_to_influx(result):

    point = Point("tablet_detection") \
        .tag("batch_id", result["batch_id"]) \
        .field("tablet_count", result["total"]) \
        .field("normal", result["normal"]) \
        .field("chipped", result["chipped"]) \
        .field("capped", result["capped"]) \
        .field("status", result["status"]) \
        .time(datetime.fromisoformat(result["time"]))

    client.write(database=INFLUX_BUCKET, record=point)