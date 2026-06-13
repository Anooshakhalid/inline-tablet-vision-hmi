def process(detections):
    """
    Two-stage QC analyzer

    detections example:

    [
        {"status": "PASS", "defect": None},
        {"status": "FAIL", "defect": "chip"},
        {"status": "FAIL", "defect": "cap"}
    ]
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

        if d["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        if d.get("defect") == "chip":
            chip_count += 1

        if d.get("defect") == "cap":
            cap_count += 1

    status = "FAIL" if fail_count > 0 else "PASS"

    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "chip": chip_count,
        "cap": cap_count,
        "status": status
    }