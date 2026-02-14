import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.rectangle import create_rectangle
from utils.search_eyes import *

cap = cv2.VideoCapture(1)

base_options = python.BaseOptions(
    model_asset_path="./cascades/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=5
)

detector = vision.FaceLandmarker.create_from_options(options)

if not cap.isOpened():
    print("Ошибка: Не удается открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Ошибка: Не удается получить кадр")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    
    result = detector.detect(mp_image)
    h, w, _ = frame.shape
    
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            rectangle = create_rectangle(face_landmarks, w, h)
            left_eye_box = eye_region(face_landmarks, LEFT_EYE, w, h)
            right_eye_box = eye_region(face_landmarks, RIGHT_EYE, w, h)
            
            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]),
               (left_eye_box[2], left_eye_box[3]), (0,255,0), 2)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]),
                    (right_eye_box[2], right_eye_box[3]), (0,255,0), 2)
            print("Лицо найдено, количество точек:", len(face_landmarks))


    cv2.imshow('Камера', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()