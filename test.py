import cv2

# Try to open the default webcam (ID 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access the webcam.")
    exit()

print("✅ Webcam access successful. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
