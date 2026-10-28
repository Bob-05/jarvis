"""TTS через Silero v5."""
import sys
import threading

import numpy as np
import sounddevice as sd

from jarvis import constants as C
from jarvis.config import AppConfig
from jarvis.tts.amplitude import AMPLITUDE_CHUNK_SEC, compute_amplitudes
from jarvis.tts.base import AmplitudeCallback, TTSEngine
from jarvis.ui.console import ui_error, ui_step, ui_success, ui_warning

try:
    from silero import silero_tts
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False


class SileroTTS(TTSEngine):
    def __init__(self, config: AppConfig):
        if not SILERO_AVAILABLE:
            ui_error("Библиотека 'silero' не установлена.")
            sys.exit(1)
        self.name = "Silero TTS v5"
        self.speaker = config.silero_speaker
        self._lock = threading.Lock()
        ui_step("🎙️", "Загрузка Silero TTS v5 (может занять до минуты)...", "yellow")
        try:
            self.model, _ = silero_tts(language="ru", speaker="v5_ru")
            ui_success(f"Silero TTS v5 загружен · голос: {self.speaker}")
        except Exception as e:
            ui_error(f"Ошибка Silero TTS: {e}")
            sys.exit(1)

    def speak(self, text: str,
              on_amplitude: AmplitudeCallback | None = None) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            try:
                audio = self.model.apply_tts(
                    text=text,
                    speaker=self.speaker,
                    sample_rate=C.TTS_SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True,
                )
                audio_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)

                # Считаем амплитуды заранее и отдаём в UI
                if on_amplitude is not None:
                    try:
                        amps = compute_amplitudes(audio_np, C.TTS_SAMPLE_RATE)
                        if amps:
                            on_amplitude(amps, AMPLITUDE_CHUNK_SEC)
                    except Exception as e:
                        ui_warning(f"Ошибка расчёта амплитуд: {e}")

                sd.play(audio_np, C.TTS_SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                ui_warning(f"Ошибка озвучки Silero: {e}")