import sys

import cv2
from PySide6.QtWidgets import QApplication

from src.config.config import AppConfig
from src.config.factory import build_from_config
from src.renderer.renderer import Renderer
from src.service.detector import Detector
from src.service.frame_pipeline import FramePipeline
from src.ui.main_window import MainWindow


def main():
    cap = cv2.VideoCapture(0)
    app_config = AppConfig.from_json("src/config/app.json")
    gesture_manager, effect_manager = build_from_config(app_config)

    detector = Detector(gesture_manager=gesture_manager)  # injection fix from earlier
    renderer = Renderer()
    pipeline = FramePipeline(detector, renderer, effect_manager)

    app = QApplication(sys.argv)
    window = MainWindow(cap, pipeline, gesture_manager, effect_manager)
    window.show()
    code = app.exec()

    detector.close()
    cap.release()
    sys.exit(code)


if __name__ == "__main__":
    main()
