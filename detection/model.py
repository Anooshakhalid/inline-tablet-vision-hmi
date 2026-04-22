from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")

def get_detections(image_path):
    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": model.names[cls_id],
                "confidence": round(conf, 2)
            })

    return detections