"""STT через Vosk."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from jarvis import constants as C
from jarvis.stt.base import STTEngine
from jarvis.ui.console import ui_error, ui_step

try:
    from vosk import KaldiRecognizer, Model
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


def validate_vosk_model(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    required = ("am", "conf", "graph")
    return all(os.path.isdir(os.path.join(path, sub)) for sub in required)


class VoskSTT(STTEngine):
    def __init__(self, model_path: str):
        if not VOSK_AVAILABLE:
            ui_error("Библиотека 'vosk' не установлена.")
            sys.exit(1)
        if not validate_vosk_model(model_path):
            ui_error(f"Модель Vosk повреждена: {model_path}")
            sys.exit(1)
        self.path = model_path
        self.name = f"Vosk · {Path(model_path).name}"
        ui_step("⏳", f"Загрузка '{Path(model_path).name}'...", "yellow")
        try:
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, C.VOSK_RATE)
        except Exception as e:
            ui_error(f"Не удалось создать модель Vosk: {e}")
            sys.exit(1)

    def recognize(self, audio_data: np.ndarray, fs: int) -> str:
        if audio_data is None or audio_data.size == 0:
            return ""
        try:
            pcm_bytes = audio_data.tobytes()
            text_parts = []
            for i in range(0, len(pcm_bytes), 4000):
                chunk = pcm_bytes[i:i + 4000]
                if self.recognizer.AcceptWaveform(chunk):
                    r = json.loads(self.recognizer.Result())
                    if r.get("text"):
                        text_parts.append(r["text"])
            final = json.loads(self.recognizer.FinalResult())
            if final.get("text"):
                text_parts.append(final["text"])
            try:
                self.recognizer.Reset()
            except Exception:
                self.recognizer = KaldiRecognizer(self.model, C.VOSK_RATE)
            return " ".join(text_parts).strip().lower()
        except Exception as e:
            ui_error(f"Ошибка Vosk: {e}")
            return ""