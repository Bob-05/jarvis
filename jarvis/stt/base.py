"""Абстрактный интерфейс STT-движка."""
from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, audio_data: np.ndarray, fs: int) -> str:
        ...