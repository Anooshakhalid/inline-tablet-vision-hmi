def process(detections):
    """
    Tablet-level QC analyzer (production-safe)
    """

    total = len(detections)

    if total == 0:
        return {
            "total": 0,
            "pass": 0,
            "fail": 0,
            "chip": 0,
            "cap": 0,
            "status": "NO_DATA"
        }

    pass_count = 0
    fail_count = 0
    chip_count = 0
    cap_count = 0

    for d in detections:

        defect = d.get("defect")

        if defect is None:
            pass_count += 1
        else:
            fail_count += 1

        if defect == "chip":
            chip_count += 1
        elif defect == "cap":
            cap_count += 1

    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "chip": chip_count,
        "cap": cap_count,
        "status": "FAIL" if fail_count > 0 else "PASS"
    }