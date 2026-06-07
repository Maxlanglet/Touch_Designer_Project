import cv2
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QScrollArea, QWidget

from src.effects.effect_manager import EffectManager
from src.gestures.gesture_manager import GestureManager
from src.service.fps_counter import FpsCounter
from src.service.frame_pipeline import FramePipeline
from src.ui.control_panel import ControlPanel
from src.ui.video_widget import VideoWidget
from src.ui.video_worker import VideoWorker


class MainWindow(QMainWindow):
    def __init__(
        self,
        cap: cv2.VideoCapture,
        pipeline: FramePipeline,
        gesture_manager: GestureManager,
        effect_manager: EffectManager,
    ):
        super().__init__()
        self.setWindowTitle("Hands")

        self.video = VideoWidget()
        fps_counter = FpsCounter()
        self.panel = ControlPanel(gesture_manager, effect_manager, fps_counter)

        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.video, stretch=1)
        layout.addWidget(scroll)
        self.setCentralWidget(central)

        self.worker = VideoWorker(cap, pipeline, fps_counter)
        self.worker.frame_ready.connect(self.video.on_frame)
        self.worker.start()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()
