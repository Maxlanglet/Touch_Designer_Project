from abc import ABC, abstractmethod

import numpy as np

from src.renderer.renderer import Renderer


class Renderable(ABC):
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    @abstractmethod
    def render(self, frame: np.ndarray, canva: Renderer) -> np.ndarray:
        """Render the object on the frame"""
