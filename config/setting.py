FRAME_LIMIT = 20   # how many samples per batch
BATCH_PREFIX = "B"

from dotenv import load_dotenv
import os

load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")