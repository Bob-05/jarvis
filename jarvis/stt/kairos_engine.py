"""STT через Kairos ASR (GigaAM)."""
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from jarvis.stt.base import STTEngine
from jarvis.ui.console import ui_error, ui_step, ui_success, ui_warning

try:
    from kairos_asr import KairosASR
    KAIROS_AVAILABLE = True
except ImportError:
    KAIROS_AVAILABLE = False


class KairosSTT(STTEngine):
    def __init__(self):
        if not KAIROS_AVAILABLE:
            ui_error("Библиотека 'kairos-asr' не установлена.")
            sys.exit(1)
        self.name = "Kairos ASR"
        ui_step("⏳", "Инициализация Kairos ASR (первый запуск скачает веса)...", "yellow")
        try:
            self.asr = KairosASR(device="auto")
            ui_success("Kairos ASR готов")
        except Exception as e:
            ui_error(f"Ошибка инициализации Kairos ASR: {e}")
            ui_warning("Проверьте, что ffmpeg установлен и доступен в PATH.")
            sys.exit(1)

    def recognize(self, audio_data: np.ndarray, fs: int) -> str:
        if audio_data is None or audio_data.size == 0:
            return ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(audio_data.tobytes())

            result = self.asr.transcribe(wav_file=tmp_path)
            if hasattr(result, "full_text"):
                text = result.full_text
            elif hasattr(result, "text"):
                text = result.text
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
            return text.strip().lower()
        except Exception as e:
            ui_error(f"Ошибка распознавания Kairos: {e}")
            return ""
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass