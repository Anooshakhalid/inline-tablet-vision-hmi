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
            "tablet": 0,
            "normal": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DATA",          # cleaner than NO_DETECTION
            "detection": "NO_OBJECT"
        }

    # =========================
    # COUNT CLASSES
    # =========================
    tablet = sum(1 for d in detections if d["class"] == "tablet")
    normal = sum(1 for d in detections if d["class"] == "normal")
    chip = sum(1 for d in detections if d["class"] == "chip")
    cap = sum(1 for d in detections if d["class"] == "cap")

    # =========================
    # DETECTION STATUS
    # =========================
    has_tablet = tablet > 0
    detection_status = "TABLET_PRESENT" if has_tablet else "NO_TABLET"

    # =========================
    # QC LOGIC (STRICT QUALITY ONLY)
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
        "total": len(detections),
        "tablet": tablet,
        "normal": normal,
        "chip": chip,
        "cap": cap,
        "status": status,                 # PASS / FAIL only
        "detection": detection_status     # separate tracking
    }