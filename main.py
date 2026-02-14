import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Ошибка: Не удается открыть камеру")
    exit()


while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Ошибка: Не удается получить кадр")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors= 5,
        minSize=(1, 1)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if len(faces) >= 2:
        print("Внимание! Несколько зрителей")

    cv2.imshow('Камера', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()