from influxdb_client import InfluxDBClient, Point, WritePrecision
from config.setting import INFLUX_URL, INFLUX_TOKEN, INFLUX_BUCKET, INFLUX_ORG
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS

# =========================
# CLIENT (InfluxDB v2)
# =========================
client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api(write_options=SYNCHRONOUS)

# =========================
# SAVE FUNCTION
# =========================
def save_to_influx(result):
    try:
        point = (
            Point("tablet_detection")
            .tag("batch_id", str(result["batch_id"]))

            .field("status", str(result["status"]))
            .field("total", int(result["total"]))
            .field("normal", int(result["normal"]))
            .field("chip", int(result["chip"]))
            .field("cap", int(result["cap"]))

            .time(datetime.utcnow(), WritePrecision.NS)
        )

        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=point
        )

        print("Data successfully written to InfluxDB")

    except Exception as e:
        print("Error writing to InfluxDB:", e)