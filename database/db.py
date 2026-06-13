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
            .tag("batch_id", str(result.get("batch_id", "0")))

            .field("status", str(result.get("status", "UNKNOWN")))
            .field("total", int(result.get("total", 0)))

            .field("pass", int(result.get("pass", 0)))
            .field("fail", int(result.get("fail", 0)))

            .field("chip", int(result.get("chip", 0)))
            .field("cap", int(result.get("cap", 0)))

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