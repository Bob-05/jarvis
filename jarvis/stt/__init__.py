"""Движки распознавания речи."""
from jarvis.stt.base import STTEngine
from jarvis.stt.kairos_engine import KairosSTT
from jarvis.stt.vosk_engine import VoskSTT, validate_vosk_model

__all__ = ["STTEngine", "VoskSTT", "KairosSTT", "validate_vosk_model"]