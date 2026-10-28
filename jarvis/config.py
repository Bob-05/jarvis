"""Dataclass AppConfig + загрузка/сохранение config.json."""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis import constants as C


@dataclass
class ResourceLimits:
    """Ограничения нагрузки на LLM (передаются в LM Studio).

    Применяются при `lms load` (через --gpu / --context-length) и через
    PowerShell PriorityClass для llama-server.exe и LM Studio.exe.
    Пользователь выбирает профиль при старте Jarvis или в панели ⚙.

    Поля:
        enabled: если False — все остальные поля игнорируются.
        gpu_ratio: доля слоёв на GPU, 0.0 = CPU only, 1.0 = все слои.
        context_length: размер контекста, 0 = значение по умолчанию LM Studio.
        process_priority: normal | below_normal | low.
    """
    enabled: bool = False
    gpu_ratio: float = 1.0
    context_length: int = 0
    process_priority: str = "normal"

    def describe(self) -> str:
        if not self.enabled:
            return "без ограничений"
        gpu = "CPU only" if self.gpu_ratio <= 0.0 else f"GPU {int(self.gpu_ratio * 100)}%"
        ctx = f"ctx {self.context_length}" if self.context_length > 0 else "ctx default"
        prio_map = {
            "normal": "priority normal",
            "below_normal": "priority below normal",
            "low": "priority idle",
        }
        prio = prio_map.get(self.process_priority, self.process_priority)
        return f"{gpu} · {ctx} · {prio}"

    def effective_priority(self) -> str:
        """Приоритет, реально применяемый к процессам LLM.

        Согласно контракту класса (enabled=False ⇒ остальные поля
        игнорируются, см. describe() → «без ограничений») при выключенных
        ограничениях приоритет не применяется — возвращаем 'normal'.
        """
        if not self.enabled:
            return "normal"
        return self.process_priority


@dataclass
class AppConfig:
    # --- Активация ---
    wake_word: str = "джарвис"
    require_wake_word: bool = True

    # --- LM Studio ---
    lm_studio_url: str = "http://localhost:1234/v1"
    default_model_key: str = ""
    lm_studio_ttl_seconds: int = 180
    lm_studio_auto_start: bool = True
    lm_studio_cli_path: str = ""
    lm_studio_auto_stop: bool = True
    lm_studio_idle_stop_seconds: int = 600

    # --- Ресурсы LLM ---
    llm_limits_enabled: bool = False
    llm_gpu_ratio: float = 1.0
    llm_context_length: int = 0
    llm_process_priority: str = "normal"

    # --- История / память ---
    history_file: str = C.HISTORY_FILE_DEFAULT
    embeddings_file: str = C.EMBEDDINGS_FILE_DEFAULT
    semantic_search_enabled: bool = True
    semantic_top_k: int = C.SEMANTIC_TOP_K_DEFAULT
    embedding_model_key: str = ""
    embedding_instruction: str = C.EMBEDDING_DEFAULT_INSTRUCTION
    rebuild_embeddings_on_start: bool = False

    # --- Флаги стартовых меню ---
    ask_stt_on_start: bool = True
    ask_tts_on_start: bool = True
    ask_wake_mode_on_start: bool = True
    ask_model_on_start: bool = True
    ask_docker_stop_mode_on_start: bool = True
    ask_logging_mode_on_start: bool = True
    ask_search_engine_on_start: bool = True
    ask_semantic_on_start: bool = True
    ask_clear_history_on_start: bool = True
    ask_embedding_model_on_start: bool = True
    ask_resource_limits_on_start: bool = True

    # --- STT ---
    stt_engine: str = "kairos-asr"
    vosk_paths: dict = field(default_factory=lambda: {
        "vosk-large": "vosk-model-ru-0.42",
        "vosk-small": "vosk-model-small-ru-0.22",
    })

    # --- TTS ---
    tts_engine: str = "silero"
    piper_model_path: str = "ru_RU-ruslan-medium.onnx"
    silero_speaker: str = "eugene"

    # --- VAD / запись ---
    vad_threshold: int = C.VAD_INITIAL_THRESHOLD

    # --- Инструменты ---
    web_search_max_results: int = 5
    web_fetch_max_chars: int = 4000
    max_tool_iterations: int = 4

    # --- SearXNG ---
    searxng_enabled: bool = True
    searxng_auto_start: bool = True
    searxng_auto_stop: bool = True
    searxng_stop_mode: str = "container"
    searxng_url: str = "http://127.0.0.1:8080"
    searxng_container_name: str = "searxng"
    searxng_compose_path: str = "./searxng"
    searxng_startup_timeout: int = 45
    searxng_fallback_to_ddg: bool = True
    search_engine_priority: str = "searxng-first"

    # --- Логи ---
    verbose_logging: bool = False
    log_file: str = C.LOG_FILE_DEFAULT

    # ---------- Методы ----------
    def get_resource_limits(self) -> ResourceLimits:
        """Собирает ResourceLimits из плоских полей конфига.

        Значения нормализуются — на случай ручной правки config.json.
        """
        try:
            gpu = float(self.llm_gpu_ratio)
        except (TypeError, ValueError):
            gpu = 1.0
        gpu = max(0.0, min(1.0, gpu))

        try:
            ctx = int(self.llm_context_length)
        except (TypeError, ValueError):
            ctx = 0
        if ctx < 0:
            ctx = 0

        prio = str(self.llm_process_priority or "normal").strip().lower()
        if prio not in ("normal", "below_normal", "low"):
            prio = "normal"

        return ResourceLimits(
            enabled=bool(self.llm_limits_enabled),
            gpu_ratio=gpu,
            context_length=ctx,
            process_priority=prio,
        )

    @classmethod
    def load(cls, path: Path | str = C.CONFIG_FILE) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)

    def save(self, path: Path | str = C.CONFIG_FILE) -> None:
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)