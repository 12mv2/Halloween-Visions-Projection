import cv2

cap = cv2.VideoCapture(0)  # or try cv2.CAP_V4L2 as second arg
if not cap.isOpened():
    print("Camera not opened")
else:
    print("Camera opened OK")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break
        cv2.imshow("cam test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

