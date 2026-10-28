"""Запись с микрофона с VAD до наступления тишины."""
import numpy as np
import sounddevice as sd

from jarvis import constants as C
from jarvis.audio.resample import resample_audio
from jarvis.ui.console import console, ui_error, ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


class VoiceRecorder:
    def __init__(self, native_rate: int, vad_threshold: int):
        self.native_rate = native_rate
        self.vad_threshold = vad_threshold

    def record_until_silence(self):
        """Запись с VAD. Не пишет в UI — события эмитит ядро.

        Возвращает (audio_int16 @ VOSK_RATE, sample_rate) или (None, None).
        """
        native_rate = self.native_rate
        chunk = int(native_rate * C.CHUNK_SEC)
        silence_chunks_needed = int(C.VAD_SILENCE_SEC / C.CHUNK_SEC)
        max_chunks = int(C.MAX_RECORD_SEC / C.CHUNK_SEC)
        min_chunks = int(C.MIN_RECORD_SEC / C.CHUNK_SEC)

        buffer = []
        silence_count = 0
        speech_started = False
        current_threshold = self.vad_threshold
        noise_floor = current_threshold / C.VAD_NOISE_FLOOR_RATIO

        try:
            with sd.RawInputStream(
                samplerate=native_rate, channels=1,
                dtype="int16", blocksize=chunk,
            ) as stream:
                for _ in range(max_chunks):
                    data, _ = stream.read(chunk)
                    audio_np = np.frombuffer(data, dtype=np.int16).copy()
                    buffer.append(audio_np)
                    energy = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))

                    if not speech_started:
                        if energy < current_threshold:
                            noise_floor = (1 - C.VAD_UPDATE_RATE) * noise_floor \
                                          + C.VAD_UPDATE_RATE * energy
                            current_threshold = max(
                                C.VAD_INITIAL_THRESHOLD,
                                noise_floor * C.VAD_NOISE_FLOOR_RATIO,
                            )

                    if energy > current_threshold:
                        if not speech_started:
                            LOGGER.write("VAD", "Обнаружена речь")
                        speech_started = True
                        silence_count = 0
                    elif speech_started:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            LOGGER.write("VAD", "Конец речи")
                            break
        except Exception as e:
            LOGGER.write("ERROR", f"Ошибка записи с микрофона: {e}")
            return None, None

        if not speech_started or len(buffer) < min_chunks:
            return None, None

        audio_native = np.concatenate(buffer, axis=0)
        audio = resample_audio(audio_native, native_rate, C.VOSK_RATE)
        if audio.size == 0:
            return None, None
        return audio, C.VOSK_RATE


def check_microphone() -> None:
    """Диагностика микрофона. Вызывается из main.py до старта ядра."""
    ui_step("🎙️", "Проверка аудиоустройств...", "yellow")
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind="input")
        native_rate = int(default_input["default_samplerate"])
        ui_success(f"Устройство ввода по умолчанию: {default_input['name']} ({native_rate} Гц)")

        ui_step("📋", "Доступные устройства ввода:", "cyan")
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " (по умолчанию)" if i == sd.default.device[0] else ""
                console.print(f"    [{i}] {dev['name']}{marker}")

        try:
            with sd.RawInputStream(
                samplerate=native_rate,
                channels=1,
                dtype="int16",
                blocksize=int(native_rate * C.CHUNK_SEC),
            ):
                ui_success(f"RawInputStream работает на {native_rate} Гц.")
        except Exception as e:
            ui_warning(f"RawInputStream не открылся: {e}")
    except Exception as e:
        ui_error(f"Ошибка при проверке аудиоустройств: {e}")