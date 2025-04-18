import cv2

cap = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

if not cap.isOpened():
    print("Error: Cannot access webcam")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
