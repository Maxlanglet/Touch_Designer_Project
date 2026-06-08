import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
from loguru import logger

from src.gestures.gesture_manager import DetectedGesture, GestureManager
from src.gestures.gesture_state_tracker import GestureEvent, GestureStateTracker
from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark

model_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class DetectionResult:
    hands: list[HandData]
    gestures: list[DetectedGesture]
    events: list[GestureEvent]


class Detector:
    def __init__(self, gesture_manager: GestureManager):
        self.latest_result = None

        self.start = time.time()
        self.frame_idx = 0
        self.gesture_manager = gesture_manager
        self.gesture_state_tracker = GestureStateTracker()
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.on_result,
        )
        self.recognizer = GestureRecognizer.create_from_options(self.options)

    def on_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def detect(self, frame) -> DetectionResult:
        self.frame_idx += 1
        mp_image = self.process_frame(frame)

        self.recognizer.recognize_async(mp_image, self.frame_idx)

        if not (self.latest_result and self.latest_result.gestures):
            return DetectionResult(
                hands=[],
                gestures=[],
                events=[],
            )

        if self.latest_result and self.latest_result.gestures:
            # frame = self.renderer.render_frame(frame, self.latest_result)
            hand_data = self.get_hand_data(latest_result=self.latest_result)
            gestures = self.gesture_manager.detect(hand_data)
            events = self.gesture_state_tracker.update(gestures)

        return DetectionResult(
            hands=hand_data,
            gestures=gestures,
            events=events,
        )

    def get_hand_data(self, latest_result: GestureRecognizerResult) -> list[HandData]:
        hand_data = []
        for i in range(len(latest_result.gestures)):
            hand_data.append(
                HandData(
                    handedness=latest_result.handedness[i],
                    gesture=latest_result.gestures[i][0].category_name,
                    score=latest_result.gestures[i][0].score,
                    hand_landmarks=[
                        SimNormalizedLandmark(lm.x, lm.y, lm.z)
                        for lm in latest_result.hand_landmarks[i]
                    ],
                    hand_world_landmarks=latest_result.hand_world_landmarks[i],
                )
            )

        return hand_data

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        return mp_image

    def close(self):
        try:
            self.recognizer.close()
        except Exception as e:
            logger.error(f"Error closing recognizer: {e}")
