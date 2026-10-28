"""TTS через Piper."""
import io
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from jarvis.config import AppConfig
from jarvis.tts.amplitude import AMPLITUDE_CHUNK_SEC, compute_amplitudes
from jarvis.tts.base import AmplitudeCallback, TTSEngine
from jarvis.ui.console import ui_error, ui_success, ui_warning

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


class PiperTTS(TTSEngine):
    def __init__(self, config: AppConfig):
        if not PIPER_AVAILABLE:
            ui_error("Библиотека 'piper-tts' не установлена.")
            sys.exit(1)
        self.name = "Piper TTS"
        self.model_path = config.piper_model_path
        self.speaker = Path(self.model_path).stem
        self._lock = threading.Lock()
        if not Path(self.model_path).exists():
            ui_error(f"Файл модели Piper не найден: {self.model_path}")
            sys.exit(1)
        try:
            self.voice = PiperVoice.load(self.model_path)
            ui_success(f"Piper TTS загружен · {Path(self.model_path).name}")
        except Exception as e:
            ui_error(f"Не удалось загрузить Piper: {e}")
            sys.exit(1)

    def speak(self, text: str,
              on_amplitude: AmplitudeCallback | None = None) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
                buf.seek(0)
                with wave.open(buf, "rb") as wav_file:
                    sr = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16)

                if on_amplitude is not None:
                    try:
                        amps = compute_amplitudes(audio_np, sr)
                        if amps:
                            on_amplitude(amps, AMPLITUDE_CHUNK_SEC)
                    except Exception as e:
                        ui_warning(f"Ошибка расчёта амплитуд: {e}")

                sd.play(audio_np, sr)
                sd.wait()
            except Exception as e:
                ui_warning(f"Ошибка озвучки Piper: {e}")