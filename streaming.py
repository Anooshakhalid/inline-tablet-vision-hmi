import cv2

cap = cv2.VideoCapture("udp://@0.0.0.0:5000", cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("LIVE STREAM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()