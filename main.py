import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.rectangle import create_rectangle
from utils.search_eyes import LEFT_EYE, RIGHT_EYE, eye_region


DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Локальное обнаружение лиц и областей глаз с веб-камеры."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Индекс камеры OpenCV (по умолчанию: 0).",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="Максимальное число лиц в кадре (по умолчанию: 5).",
    )
    parser.add_argument(
        "--alert-threshold",
        type=int,
        default=2,
        help="Количество лиц для предупреждения (по умолчанию: 2).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Путь к модели MediaPipe Face Landmarker.",
    )
    return parser.parse_args()


def draw_detection(frame, face_landmarks, width, height):
    x1, y1, x2, y2 = create_rectangle(face_landmarks, width, height)
    left_eye_box = eye_region(face_landmarks, LEFT_EYE, width, height)
    right_eye_box = eye_region(face_landmarks, RIGHT_EYE, width, height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 2)
    cv2.rectangle(
        frame,
        (left_eye_box[0], left_eye_box[1]),
        (left_eye_box[2], left_eye_box[3]),
        (0, 255, 0),
        2,
    )
    cv2.rectangle(
        frame,
        (right_eye_box[0], right_eye_box[1]),
        (right_eye_box[2], right_eye_box[3]),
        (0, 255, 0),
        2,
    )


def main():
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Модель не найдена: {args.model}")

    base_options = python.BaseOptions(model_asset_path=str(args.model))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=args.max_faces,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError(
            f"Не удалось открыть камеру с индексом {args.camera}. "
            "Попробуйте передать другой индекс через --camera."
        )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = detector.detect(mp_image)

            height, width, _ = frame.shape
            faces = result.face_landmarks or []

            for face_landmarks in faces:
                draw_detection(frame, face_landmarks, width, height)

            multiple_viewers = len(faces) >= args.alert_threshold
            status_color = (0, 0, 255) if multiple_viewers else (0, 255, 0)
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            if multiple_viewers:
                cv2.putText(
                    frame,
                    "WARNING: multiple viewers",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    status_color,
                    2,
                )

            cv2.imshow("Machine Eyes Check", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
