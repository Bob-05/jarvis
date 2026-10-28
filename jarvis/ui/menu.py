"""Интерактивные меню (стрелки + Enter, Windows msvcrt)."""
import sys
from pathlib import Path

import msvcrt
from rich.align import Align
from rich.box import DOUBLE
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from jarvis import constants as C
from jarvis.config import AppConfig
from jarvis.ui.console import console, ui_error
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


# ---------- Базовое меню ----------
def _build_menu_render(title, options, selected, extra_hint=""):
    elements = [Text()]
    elements.append(Panel(
        Align.center(Text(title, style="bold bright_cyan")),
        border_style="bright_cyan", box=DOUBLE, padding=(0, 4),
    ))
    elements.append(Text())
    for i, (label, desc) in enumerate(options):
        if i == selected:
            elements.append(Text(f"  ▶  {label}", style="bold bright_cyan"))
            if desc:
                elements.append(Text(f"       {desc}", style="dim cyan"))
        else:
            elements.append(Text(f"     {label}", style="white"))
            if desc:
                elements.append(Text(f"       {desc}", style="dim"))
    elements.append(Text())
    hint = "  ↑/↓ — навигация · Enter — выбрать"
    if extra_hint:
        hint += " · " + extra_hint
    elements.append(Text(hint, style="dim"))
    return Group(*elements)


def _menu(title, options, initial=0, extra_hint=""):
    selected = initial
    console.clear()
    with Live(_build_menu_render(title, options, selected, extra_hint),
              console=console, refresh_per_second=8) as live:
        while True:
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"H":
                    selected = (selected - 1) % len(options)
                elif ch2 == b"P":
                    selected = (selected + 1) % len(options)
                live.update(_build_menu_render(title, options, selected, extra_hint))
            elif ch == b"\r":
                console.clear()
                return selected
            elif ch == b"\x1b" and extra_hint:
                console.clear()
                return -1


# ---------- Меню: активация ----------
def menu_select_wake_mode(config: AppConfig) -> bool:
    wake = config.wake_word
    options = [
        ("С активационным словом", f"Отвечает только после слова «{wake}»"),
        ("Без активационного слова", "Отвечает на любую речь (может срабатывать на шум)"),
    ]
    initial = 0 if config.require_wake_word else 1
    return _menu("РЕЖИМ АКТИВАЦИИ", options, initial) == 0


# ---------- Меню: логирование ----------
def menu_select_logging_mode(config: AppConfig) -> bool:
    options = [
        ("Краткий", "Только вывод в консоль, без файла"),
        ("Подробный (консоль + файл логов)", "Все события пишутся в файл"),
    ]
    initial = 1 if config.verbose_logging else 0
    return _menu("РЕЖИМ ЛОГИРОВАНИЯ", options, initial) == 1


def menu_ask_clear_logs() -> bool:
    size = LOGGER.file_size_kb()
    options = [
        ("Нет — оставить прежние логи", f"Текущий размер: {size} КБ"),
        ("Да — очистить файл логов", "Начать с чистого листа"),
    ]
    return _menu("ОЧИСТИТЬ ФАЙЛ ЛОГОВ?", options, 0) == 1


# ---------- Меню: поисковик ----------
def menu_select_search_engine(config: AppConfig) -> str:
    options = [
        ("SearXNG → DuckDuckGo",
         "Сначала SearXNG, при сбое — DuckDuckGo (нужен Docker)"),
        ("DuckDuckGo → SearXNG",
         "Сначала DuckDuckGo, при сбое — SearXNG"),
        ("Только DuckDuckGo",
         "Без Docker, простой поиск"),
    ]
    current = config.search_engine_priority
    initial = {"searxng-first": 0, "ddg-first": 1, "ddg-only": 2}.get(current, 0)
    return ["searxng-first", "ddg-first", "ddg-only"][
        _menu("ПОИСКОВИК ПО УМОЛЧАНИЮ", options, initial)
    ]


# ---------- Меню: остановка Docker ----------
def menu_select_docker_stop_mode(config: AppConfig) -> str:
    options = [
        ("Только контейнер SearXNG",
         "Остановлю контейнер, Docker Desktop продолжит работу"),
        ("Весь Docker Desktop",
         "Освобожу ~1 ГБ RAM, но закрою все запущенные контейнеры"),
    ]
    initial = 0 if config.searxng_stop_mode == "container" else 1
    return "container" if _menu("ЧТО ДЕЛАТЬ С DOCKER ПРИ ВЫХОДЕ", options, initial) == 0 else "docker"


# ---------- Меню: STT ----------
def menu_select_stt(config: AppConfig, kairos_available: bool, vosk_available: bool,
                    validate_fn) -> str:
    options = []
    if kairos_available:
        options.append(("kairos-asr", "Kairos ASR (GigaAM) · современное, точное распознавание"))
    if vosk_available:
        for key, path in config.vosk_paths.items():
            if validate_fn(path):
                options.append((key, f"Vosk · {path}"))

    if not options:
        ui_error("Ни один движок распознавания речи не доступен.")
        sys.exit(1)

    current = config.stt_engine
    initial = 0
    for i, (key, _) in enumerate(options):
        if key == current:
            initial = i
            break
    return options[_menu("ВЫБОР ДВИЖКА РАСПОЗНАВАНИЯ РЕЧИ", options, initial)][0]


# ---------- Меню: TTS ----------
def menu_select_tts(config: AppConfig, silero_available: bool, piper_available: bool) -> str:
    options = []
    if silero_available:
        options.append(("silero", "Silero TTS v5 · естественный голос с интонациями"))
    if piper_available:
        options.append(("piper", "Piper TTS · быстрый и лёгкий"))

    if not options:
        ui_error("Ни один движок TTS не установлен.")
        sys.exit(1)

    current = config.tts_engine
    initial = 0
    for i, (key, _) in enumerate(options):
        if key == current:
            initial = i
            break
    return options[_menu("ВЫБОР ДВИЖКА ОЗВУЧКИ", options, initial)][0]


# ---------- Меню: LLM ----------
def menu_select_model(models, current_key="", default_key=""):
    if not models:
        return None
    options = []
    for m in models:
        key = m["key"]
        display = m.get("display_name") or key
        params = m.get("params_string") or ""
        marker = ""
        if key == current_key:
            marker += " (активна)"
        if key == default_key:
            marker += " (по умолчанию)"
        options.append((f"{display}  {params}{marker}", ""))
    initial = 0
    for i, m in enumerate(models):
        if m["key"] == current_key:
            initial = i
            break
    result = _menu("ВЫБОР МОДЕЛИ LM STUDIO", options, initial,
                   extra_hint="Esc — оставить по умолчанию")
    if result < 0:
        return None
    return models[result]["key"]


# ---------- Меню: модель эмбеддингов ----------
def menu_select_embedding_model(models, current_key="", default_key=""):
    """Меню выбора модели эмбеддингов.

    Возвращает:
      str  — ключ выбранной модели;
      None — оставить по умолчанию (Esc).
    """
    options: list[tuple[str, str]] = []
    keys: list[str | None] = []

    for m in models:
        key = m["key"]
        display = m.get("display_name") or key
        marker = ""
        if key == current_key:
            marker += " (активна)"
        if key == default_key and key != current_key:
            marker += " (по умолчанию)"
        options.append((f"{display}{marker}", ""))
        keys.append(key)

    options.append(("← Ввести имя вручную",
                    "Использовать любую модель LM Studio по её ID"))
    keys.append(None)

    initial = 0
    for i, k in enumerate(keys):
        if k == current_key:
            initial = i
            break

    result = _menu("МОДЕЛЬ ЭМБЕДДИНГОВ (для семантического поиска)",
                   options, initial, extra_hint="Esc — оставить по умолчанию")
    if result < 0:
        return None

    if keys[result] is None:
        console.clear()
        console.print("[bold bright_cyan]Введите ID модели эмбеддингов из LM Studio[/bold bright_cyan]")
        console.print("[dim]Например: ai-babai/giga-embeddings-0826-480m-gguf[/dim]")
        try:
            manual = console.input("[bold yellow]➤ [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return manual or None

    return keys[result]


# ---------- Меню: семантический поиск ----------
def menu_select_semantic_top_k(config: AppConfig) -> int:
    current = config.semantic_top_k
    options = []
    for k in C.SEMANTIC_TOP_K_CHOICES:
        if k == 0:
            options.append(("Отключить семантический поиск",
                            "В LM Studio уйдёт только текущий запрос, без истории"))
        else:
            options.append((f"{k} релевантных фрагмент(ов) истории",
                            "Будут подмешаны к текущему запросу как контекст"))
    try:
        initial = C.SEMANTIC_TOP_K_CHOICES.index(current)
    except ValueError:
        initial = C.SEMANTIC_TOP_K_CHOICES.index(C.SEMANTIC_TOP_K_DEFAULT)
    return C.SEMANTIC_TOP_K_CHOICES[
        _menu("СЕМАНТИЧЕСКИЙ ПОИСК ПО ИСТОРИИ", options, initial)
    ]


# ---------- Меню: очистка истории ----------
def menu_clear_history() -> str:
    """Возвращает: 'cancel' | 'dialogs' | 'all' | 'rebuild'."""
    options = [
        ("Отмена", "Ничего не менять"),
        ("Очистить только диалоги", "Файл истории опустеет, эмбеддинги оставим"),
        ("Очистить диалоги и эмбеддинги", "Полный сброс памяти"),
        ("Пересчитать эмбеддинги заново", "Диалоги сохранить, векторы пересчитать"),
    ]
    keys = ["cancel", "dialogs", "all", "rebuild"]
    return keys[_menu("УПРАВЛЕНИЕ ИСТОРИЕЙ ДИАЛОГОВ", options, 0)]


# ---------- Меню: ресурсы LLM ----------
def menu_select_resource_limits(config: AppConfig) -> dict:
    """Меню выбора профиля ресурсов для LLM.

    Возвращает dict:
      {"enabled": bool, "gpu_ratio": float,
       "context_length": int, "priority": str}
    """
    profiles = [
        ("Без ограничений",
         "LM Studio сам решит, сколько слоёв на GPU",
         {"enabled": False, "gpu_ratio": 1.0, "context_length": 0,
          "priority": "normal"}),
        ("Сбалансированный",
         "GPU 60% · ctx 4096 · приоритет ниже обычного",
         {"enabled": True, "gpu_ratio": 0.6, "context_length": 4096,
          "priority": "below_normal"}),
        ("Экономия VRAM",
         "GPU 30% · ctx 2048 · приоритет ниже обычного",
         {"enabled": True, "gpu_ratio": 0.3, "context_length": 2048,
          "priority": "below_normal"}),
        ("Только CPU (не занимает VRAM)",
         "GPU off · ctx 2048 · приоритет ниже обычного",
         {"enabled": True, "gpu_ratio": 0.0, "context_length": 2048,
          "priority": "below_normal"}),
        ("Настроить вручную...",
         "Ввести параметры с клавиатуры",
         None),
    ]

    # Стартовая позиция — по текущему профилю
    initial = 0
    if config.llm_limits_enabled:
        ratio = float(config.llm_gpu_ratio or 0.0)
        for i, (_, _, data) in enumerate(profiles):
            if data and data["enabled"] and abs(data["gpu_ratio"] - ratio) < 0.01:
                initial = i
                break
        else:
            initial = 4  # manual

    options = [(p[0], p[1]) for p in profiles]
    idx = _menu("РЕСУРСЫ ДЛЯ МОДЕЛИ (GPU / контекст / приоритет)",
                options, initial, extra_hint="Esc — оставить как есть")

    if idx < 0:
        # Esc — возвращаем текущее состояние без изменений
        return {
            "enabled": config.llm_limits_enabled,
            "gpu_ratio": config.llm_gpu_ratio,
            "context_length": config.llm_context_length,
            "priority": config.llm_process_priority,
        }

    _, _, data = profiles[idx]

    if data is not None:
        return data

    # --- Ручной ввод ---
    console.clear()
    console.print("[bold bright_cyan]Настройка ресурсов вручную[/bold bright_cyan]")
    console.print("[dim]Enter — оставить значение по умолчанию[/dim]\n")

    try:
        raw = console.input(
            "[bold yellow]Доля слоёв на GPU (0.0–1.0; 0=CPU only) [1.0]: "
            "[/bold yellow]"
        ).strip()
        gpu_ratio = float(raw) if raw else 1.0
        gpu_ratio = max(0.0, min(1.0, gpu_ratio))
    except (ValueError, EOFError, KeyboardInterrupt):
        gpu_ratio = 1.0

    try:
        raw = console.input(
            "[bold yellow]Длина контекста (0=по умолчанию LM Studio) [4096]: "
            "[/bold yellow]"
        ).strip()
        context_length = int(raw) if raw else 4096
        context_length = max(0, context_length)
    except (ValueError, EOFError, KeyboardInterrupt):
        context_length = 4096

    prio_opts = [
        ("Обычный", "Приоритет по умолчанию Windows"),
        ("Ниже обычного (рекомендуется)",
         "Система отдаст CPU другим приложениям первым делом"),
        ("Низкий (Idle)",
         "Максимально уступать системе, но медленнее"),
    ]
    prio_keys = ["normal", "below_normal", "low"]
    prio_idx = _menu("ПРИОРИТЕТ ПРОЦЕССА", prio_opts, 1)
    priority = prio_keys[prio_idx]

    return {
        "enabled": True,
        "gpu_ratio": gpu_ratio,
        "context_length": context_length,
        "priority": priority,
    }