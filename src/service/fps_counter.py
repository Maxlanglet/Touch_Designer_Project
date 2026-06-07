import cv2


class FpsCounter:
    def __init__(self, smoothing: float = 0.9):
        self.prev = cv2.getTickCount()
        self.fps = 0.0
        self.smoothing = smoothing

    def update(self) -> float:
        now = cv2.getTickCount()
        dt = (now - self.prev) / cv2.getTickFrequency()
        self.prev = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = (
                (self.smoothing * self.fps + (1 - self.smoothing) * inst) if self.fps else inst
            )
        return self.fps

    def draw(self, frame):
        _, w = frame.shape[:2]
        text = f"{self.fps:4.1f} FPS"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame, text, (w - tw - 10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        return frame
