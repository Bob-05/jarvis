"""Ресемплинг int16-аудио."""
import numpy as np


def resample_audio(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio_np.size == 0:
        return audio_np
    if orig_sr % target_sr == 0:
        ratio = orig_sr // target_sr
        n = (len(audio_np) // ratio) * ratio
        if n == 0:
            return np.array([], dtype=np.int16)
        trimmed = audio_np[:n].astype(np.float32).reshape(-1, ratio)
        averaged = trimmed.mean(axis=1)
        return np.clip(averaged, -32768, 32767).astype(np.int16)

    audio_float = audio_np.astype(np.float32)
    target_len = int(len(audio_float) * target_sr / orig_sr)
    if target_len == 0:
        return np.array([], dtype=np.int16)
    orig_indices = np.arange(len(audio_float))
    target_indices = np.linspace(0, len(audio_float) - 1, target_len)
    resampled = np.interp(target_indices, orig_indices, audio_float)
    return np.clip(resampled, -32768, 32767).astype(np.int16)