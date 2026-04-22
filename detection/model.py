from ultralytics import YOLO

# Load model once
model = YOLO("models/best.pt")

def get_detections(frame):
    # Run inference
    results = model(frame)
    
    # Return raw YOLO results (NOT custom dict)
    return results