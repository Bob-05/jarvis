"""Абстрактный интерфейс TTS-движка."""
from abc import ABC, abstractmethod
from typing import Callable


AmplitudeCallback = Callable[[list[float], float], None]
"""Callback для передачи массива амплитуд.

Аргументы:
  amplitudes: list[float] — RMS-значения, нормализованные в [0..1]
  chunk_sec:  float       — длительность одного чанка в секундах
"""


class TTSEngine(ABC):
    name: str = "base"
    speaker: str = ""

    @abstractmethod
    def speak(self, text: str,
              on_amplitude: AmplitudeCallback | None = None) -> None:
        ...