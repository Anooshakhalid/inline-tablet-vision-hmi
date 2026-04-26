from datetime import datetime

def process(detections, batch_id):

    if not detections:
        return {
            "time": datetime.now().isoformat(),
            "batch_id": batch_id,
            "total": 0,
            "normal": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DETECTION"
        }

    # normalize class names
    detections = [
        {**d, "class": d["class"].lower().strip()}
        for d in detections
        if d["confidence"] > 0.5
    ]

    normal = sum(1 for d in detections if d["class"] == "normal")
    chip = sum(1 for d in detections if d["class"] == "chip")
    cap = sum(1 for d in detections if d["class"] == "cap")

    total = len(detections)

    status = "FAIL" if (chip + cap) > 0 else "PASS"

    return {
        "time": datetime.now().isoformat(),
        "batch_id": batch_id,
        "total": total,
        "normal": normal,
        "chip": chip,
        "cap": cap,
        "status": status
    }