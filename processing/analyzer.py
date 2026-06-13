from datetime import datetime

def process(detections):
    """
    Frame-level QC analyzer (stateless)
    """

    total = len(detections)

    if total == 0:
        return {
            "total": 0,
            "normal": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DATA"
        }

    normal = sum(1 for d in detections if d["class"] == "normal")
    chip = sum(1 for d in detections if d["class"] == "chip")
    cap = sum(1 for d in detections if d["class"] == "cap")

    status = "FAIL" if (chip + cap) > 0 else "PASS"

    return {
        "total": total,
        "normal": normal,
        "chip": chip,
        "cap": cap,
        "status": status
    }