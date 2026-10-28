"""События, генерируемые ядром Jarvis.

Каждое событие — dataclass Event с полем type (строка из EventType)
и data (произвольный dict с полезной нагрузкой).

UI-слои (ConsoleUI, WebUI) читают эти события из очереди ядра
и реагируют в соответствии с типом.
"""
from dataclasses import dataclass, field


class EventType:
    """Все типы событий, которые может генерировать ядро."""

    # --- Жизненный цикл ---
    STARTED = "started"
    SHUTDOWN = "shutdown"

    # --- Голосовое состояние (для орба / индикатора) ---
    # data: {"state": "idle" | "listening" | "thinking" | "speaking"}
    VOICE_STATE = "voice_state"

    # --- Амплитуды TTS (для визуализации голоса Джарвиса) ---
    # data: {"amplitudes": [float, ...], "chunk_sec": float}
    TTS_AMPLITUDE = "tts_amplitude"

    # --- Диалог ---
    # data: {"text": str}
    USER_SPOKE = "user_spoke"
    # data: {"text": str, "spoken": bool}
    ASSISTANT_REPLIED = "assistant_replied"

    # --- Статусы / шаги ---
    # data: {"text": str, "color": str}
    STATUS = "status"
    # data: {"icon": str, "text": str, "color": str}
    STEP = "step"
    # data: {"text": str}
    SUCCESS = "success"
    # data: {"text": str}
    WARNING = "warning"
    # data: {"text": str}
    ERROR = "error"

    # --- Команды ---
    # data: {"text": str, "dangerous": bool}
    COMMAND_REQUESTED = "command_requested"
    # data: {"text": str, "success": bool}
    COMMAND_EXECUTED = "command_executed"
    # data: {"text": str}
    COMMAND_CANCELLED = "command_cancelled"

    # --- Инструменты (поиск / чтение) ---
    # data: {"kind": "search"|"fetch", "arg": str}
    TOOL_CALL = "tool_call"
    # data: {"kind": str, "arg": str, "chars": int}
    TOOL_RESULT = "tool_result"

    # --- Логи ---
    # data: {"level": str, "text": str}
    LOG = "log"


@dataclass
class Event:
    type: str
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.type!r}, {self.data!r})"