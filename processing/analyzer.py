from datetime import datetime

def process(detections, batch_id):

    current_time = datetime.now().isoformat()

    # =========================
    # NORMALIZE + FILTER
    # =========================
    detections = [
        {**d, "class": d["class"].lower().strip()}
        for d in detections
        if d["confidence"] > 0.25
    ]

    # =========================
    # NO DETECTION CASE
    # =========================
    if not detections:
        return {
            "time": current_time,
            "batch_id": batch_id,
            "total": 0,
            "normal": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DATA"
        }

    # =========================
    # COUNT CLASSES
    # =========================
    normal = sum(1 for d in detections if d["class"] == "normal")
    chip = sum(1 for d in detections if d["class"] == "chip")
    cap = sum(1 for d in detections if d["class"] == "cap")

    total = len(detections)

    # =========================
    # QC LOGIC
    # =========================
    has_defect = (chip + cap) > 0

    if has_defect:
        status = "FAIL"
    else:
        status = "PASS"

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "time": current_time,
        "batch_id": batch_id,
        "total": total,
        "normal": normal,
        "chip": chip,
        "cap": cap,
        "status": status
    }