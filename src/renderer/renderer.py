import cv2
import mediapipe as mp
import numpy as np

from src.models.sim_normalized_landmarks import SimNormalizedLandmark

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

RESERVED_HAND_ROWS = 2


class Renderer:
    def __init__(self, row_text: int = 0):
        self.next_row = RESERVED_HAND_ROWS
        self.text_offset = 20
        self.row_height = 40

    def circle(
        self,
        frame: np.ndarray,
        landmark: SimNormalizedLandmark,
        color: tuple[int, int, int],
        radius: int = 2,
        thickness: int = -1,
    ):
        h, w, _ = frame.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), radius, color, thickness)
        return frame

    def line(
        self,
        frame: np.ndarray,
        landmark1: SimNormalizedLandmark,
        landmark2: SimNormalizedLandmark,
        color: tuple[int, int, int],
        thickness: int = 3,
    ):
        h, w, _ = frame.shape
        cv2.line(
            frame,
            (int(landmark1.x * w), int(landmark1.y * h)),
            (int(landmark2.x * w), int(landmark2.y * h)),
            color,
            thickness,
        )
        return frame

    def text(
        self,
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1,
        thickness: int = 2,
        row: int | None = None,
    ):
        target = row if row is not None else self.next_row
        if row is None:
            self.next_row += 1
        y = self.text_offset + target * self.row_height
        x = 10

        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        return frame

    def render_polygon(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int = 3,
    ):
        h, w, _ = frame.shape
        px_points = [(int(point[0] * w), int(point[1] * h)) for point in points]
        cv2.polylines(frame, [np.array(px_points)], True, color, thickness)
        return frame

    def _to_px(self, landmark: SimNormalizedLandmark, w: int, h: int):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        return cx, cy

    def _get_text_position(self):
        return (10, self.text_offset + self.next_row * self.row_height)

    def _increment_row(self):
        self.next_row += 1

    def reset(self):
        self.next_row = RESERVED_HAND_ROWS

    def close(self):
        pass
