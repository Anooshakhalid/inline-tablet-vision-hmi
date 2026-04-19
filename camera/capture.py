import cv2

cap = cv2.VideoCapture(0)  # external webcam

def get_frame():
    ret, frame = cap.read()

    if not ret:
        return None

    return frame