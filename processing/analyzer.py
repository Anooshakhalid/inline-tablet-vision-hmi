from datetime import datetime

def process(detections, batch_id):

    # =========================
    # NORMALIZE + FILTER
    # =========================
    detections = [
        {**d, "class": d["class"].lower().strip()}
        for d in detections
        if d["confidence"] > 0.25
    ]

    if not detections:
        return {
            "time": datetime.now().isoformat(),
            "batch_id": batch_id,
            "total": 0,
            "tablet": 0,
            "normal": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DETECTION"
        }

    # =========================
    # COUNT CLASSES
    # =========================
    tablet = sum(1 for d in detections if d["class"] == "tablet")
    normal = sum(1 for d in detections if d["class"] == "normal")
    chip = sum(1 for d in detections if d["class"] == "chip")
    cap = sum(1 for d in detections if d["class"] == "cap")

    # =========================
    # QC LOGIC
    # =========================
    has_tablet = tablet > 0
    has_defect = (chip + cap) > 0

    if not has_tablet:
        status = "NO_TABLET"
    elif has_defect:
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "time": datetime.now().isoformat(),
        "batch_id": batch_id,
        "total": len(detections),
        "tablet": tablet,
        "normal": normal,
        "chip": chip,
        "cap": cap,
        "status": status
    }