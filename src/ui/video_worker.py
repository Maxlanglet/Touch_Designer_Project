import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.service.fps_counter import FpsCounter
from src.service.frame_pipeline import FramePipeline


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, cap: cv2.VideoCapture, pipeline: FramePipeline, fps_counter: FpsCounter):
        super().__init__()
        self.cap = cap
        self.pipeline = pipeline
        self.fps_counter = fps_counter
        self.running = True

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.pipeline.process(frame)
            self.fps_counter.update()
            frame = self.fps_counter.draw(frame)
            self.frame_ready.emit(frame)

    def stop(self):
        self.running = False
        self.wait()
