import cv2

def show(result):

    frame = 255 * (np.ones((300, 500, 3), dtype="uint8"))

    cv2.putText(frame, f"Batch: {result['batch_id']}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.putText(frame, f"Normal: {result['normal']}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Chipped: {result['chipped']}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Capped: {result['capped']}", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Status: {result['status']}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Tablet HMI", frame)
    cv2.waitKey(1)