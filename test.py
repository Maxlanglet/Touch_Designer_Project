import time

import cv2
import mediapipe as mp

model_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    connect_lines(frame, hand_landmarks)


# Connection of the lines require a specific order
con_lines = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


def connect_lines(frame, hand_landmarks):
    h, w, _ = frame.shape
    for line in con_lines:
        for i in range(len(line) - 1):
            cv2.line(
                frame,
                (int(hand_landmarks[line[i]].x * w), int(hand_landmarks[line[i]].y * h)),
                (int(hand_landmarks[line[i + 1]].x * w), int(hand_landmarks[line[i + 1]].y * h)),
                (0, 255, 0),
                3,
            )


def draw(frame, result: GestureRecognizerResult):
    for hand_landmarks in result.hand_landmarks:
        draw_landmarks(frame, hand_landmarks)


def on_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    start = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # OpenCV is BGR; MediaPipe wants RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # timestamp MUST be monotonically increasing (in ms)
        timestamp_ms = int((time.time() - start) * 1000)
        recognizer.recognize_async(mp_image, timestamp_ms)

        if latest_result and latest_result.gestures:
            top = latest_result.gestures[0][0]
            cv2.putText(
                frame,
                f"{top.category_name} ({top.score:.2f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            draw(frame, latest_result)

        cv2.imshow("Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# The detector is initialized. Use it here.
# ...
