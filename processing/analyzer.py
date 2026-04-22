from datetime import datetime

def process(detections, batch_id):

    if not detections:
        return {
            "time": datetime.now().isoformat(),
            "batch_id": batch_id,
            "total": 0,
            "normal": 0,
            "chipped": 0,
            "capped": 0,
            "status": "NO_DETECTION"
        }

    detections = [d for d in detections if d["confidence"] > 0.5]

    normal = sum(1 for d in detections if d["class"] == "normal")
    chipped = sum(1 for d in detections if d["class"] == "chipped")
    capped = sum(1 for d in detections if d["class"] == "capped")

    total = len(detections)

    status = "FAIL" if (chipped + capped) > 0 else "PASS"

    return {
        "time": datetime.now().isoformat(),
        "batch_id": batch_id,
        "total": total,
        "normal": normal,
        "chipped": chipped,
        "capped": capped,
        "status": status
    }