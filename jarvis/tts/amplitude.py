"""Расчёт амплитуд из TTS-аудио для визуализации орба.

Содержит простое сглаживание (moving average), чтобы орб
двигался плавно, без резких скачков между чанками.
"""
import numpy as np

AMPLITUDE_CHUNK_SEC = 0.04       # 25 значений в секунду
SMOOTHING_WINDOW = 5             # окно moving average (нечётное)


def _moving_average(values: list[float], window: int) -> list[float]:
    """Простой сглаживающий фильтр — скользящее среднее.

    На краях окно сужается, чтобы не терять амплитуду в начале и конце.
    """
    if not values or window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(values)
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def compute_amplitudes(audio_np: np.ndarray, sample_rate: int,
                       chunk_sec: float = AMPLITUDE_CHUNK_SEC,
                       smoothing: int = SMOOTHING_WINDOW) -> list[float]:
    """Разбивает аудио на чанки и возвращает RMS-значения, нормализованные в 0..1.

    Дополнительно применяет скользящее среднее для плавности.
    """
    if audio_np is None or len(audio_np) == 0:
        return []

    chunk_size = max(1, int(sample_rate * chunk_sec))
    amps: list[float] = []
    audio_f = audio_np.astype(np.float32)

    for i in range(0, len(audio_f), chunk_size):
        chunk = audio_f[i:i + chunk_size]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        amps.append(rms)

    if not amps:
        return []

    # Нормализация по пику
    peak = max(amps)
    if peak > 0:
        amps = [a / peak for a in amps]

    # Сглаживание
    amps = _moving_average(amps, smoothing)
    # Ограничиваем [0, 1] на случай, если moving average дал больше
    amps = [min(1.0, max(0.0, a)) for a in amps]
    return amps