import numpy as np

from src.models.hand_data import HandData
from src.models.sim_normalized_landmarks import SimNormalizedLandmark
from src.renderer.colors import BLUE, GREEN, RED
from src.renderer.renderable import Renderable
from src.renderer.renderer import Renderer

con_lines = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


class HandRenderer(Renderable):
    def __init__(self, hand: HandData):
        self.hand = hand

    def render(self, frame: np.ndarray, canvas: Renderer) -> np.ndarray:
        frame = self._connect_lines(frame, self.hand.hand_landmarks, canvas)
        frame = self._draw_landmarks(frame, self.hand.hand_landmarks, canvas)
        frame = self._put_text(frame, self.hand.get_hand_stats(), canvas)
        return frame

    def _connect_lines(
        self, frame: np.ndarray, hand_landmarks: list[SimNormalizedLandmark], canvas: Renderer
    ):
        for line in con_lines:
            for i in range(len(line) - 1):
                frame = canvas.line(
                    frame, hand_landmarks[line[i]], hand_landmarks[line[i + 1]], GREEN
                )
        return frame

    def _draw_landmarks(
        self, frame: np.ndarray, hand_landmarks: list[SimNormalizedLandmark], canvas: Renderer
    ):
        for landmark in hand_landmarks:
            frame = canvas.circle(frame, landmark, RED)
        return frame

    def _put_text(self, frame: np.ndarray, hand_stats: dict[str, any], canvas: Renderer):
        hand_category = "Right" if hand_stats["handedness"] == "Right" else "Left"
        frame = canvas.text(
            frame,
            f"Hand {hand_category} detected: {hand_stats['gesture']} ({hand_stats['score']:.2f})",
            BLUE,
        )
        return frame

    def close(self):
        pass
