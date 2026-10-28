"""Движки синтеза речи."""
from jarvis.tts.amplitude import AMPLITUDE_CHUNK_SEC, compute_amplitudes
from jarvis.tts.base import AmplitudeCallback, TTSEngine
from jarvis.tts.piper_engine import PIPER_AVAILABLE, PiperTTS
from jarvis.tts.silero_engine import SILERO_AVAILABLE, SileroTTS

__all__ = [
    "TTSEngine", "AmplitudeCallback",
    "SileroTTS", "PiperTTS",
    "SILERO_AVAILABLE", "PIPER_AVAILABLE",
    "compute_amplitudes", "AMPLITUDE_CHUNK_SEC",
]