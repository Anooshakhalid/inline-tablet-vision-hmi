import random

CLASSES = ["normal", "chipped", "capped"]

def get_detections():
    detections = []

    for _ in range(random.randint(1, 5)):
        detections.append({
            "class": random.choice(CLASSES),
            "confidence": round(random.uniform(0.7, 0.99), 2)
        })

    return detections