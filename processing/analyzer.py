from datetime import datetime

def process(detections, batch_id):

    normal = sum(1 for d in detections if d["class"] == "normal")
    chipped = sum(1 for d in detections if d["class"] == "chipped")
    capped = sum(1 for d in detections if d["class"] == "capped")

    total = len(detections)

    status = "FAIL" if (chipped + capped) > 1 else "PASS"

    return {
        "time": datetime.now().isoformat(),
        "batch_id": batch_id,
        "total": total,
        "normal": normal,
        "chipped": chipped,
        "capped": capped,
        "status": status
    }