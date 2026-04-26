from influxdb_client import InfluxDBClient, Point, WritePrecision
from config.setting import INFLUX_URL, INFLUX_TOKEN, INFLUX_BUCKET, INFLUX_ORG
from datetime import datetime

# =========================
# CLIENT (InfluxDB v2)
# =========================
client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api()

# =========================
# SAVE FUNCTION
# =========================
def save_to_influx(result):

    point = (
        Point("tablet_detection")
        .tag("batch_id", result["batch_id"])
        .field("tablet_count", int(result["total"]))
        .field("normal", int(result["normal"]))
        .field("chipped", int(result["chipped"]))
        .field("capped", int(result["capped"]))
        .field("status", result["status"])
        .time(datetime.fromisoformat(result["time"]), WritePrecision.NS)
    )

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=point
    )