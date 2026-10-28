"""Консольный UI — подписчик событий ядра.

Запускает JarvisCore в фоновом потоке, читает события из очереди
и печатает их через ui_* функции Rich.

Основной цикл завершается:
  * когда ядро эмитит EventType.SHUTDOWN
  * или по Ctrl+C
"""
import queue
import threading

from jarvis.core.events import Event, EventType
from jarvis.core.jarvis_core import JarvisCore
from jarvis.ui.console import (
    console, ui_command, ui_error, ui_jarvis, ui_status, ui_step,
    ui_success, ui_user, ui_warning,
)
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


class ConsoleUI:
    def __init__(self, core: JarvisCore):
        self.core = core
        self._stop = threading.Event()
        self._core_thread: threading.Thread | None = None

    # ============================================================
    # Обработка событий
    # ============================================================
    def handle(self, event: Event) -> None:
        t = event.type
        d = event.data or {}

        if t == EventType.USER_SPOKE:
            ui_user(d.get("text", ""))

        elif t == EventType.ASSISTANT_REPLIED:
            # Озвучка идёт параллельно; печатаем только один раз
            if d.get("spoken", True):
                ui_jarvis(d.get("text", ""))

        elif t == EventType.STATUS:
            ui_status(d.get("text", ""), d.get("color", "green"))

        elif t == EventType.STEP:
            ui_step(d.get("icon", "•"), d.get("text", ""),
                    d.get("color", "cyan"))

        elif t == EventType.SUCCESS:
            ui_success(d.get("text", ""))

        elif t == EventType.WARNING:
            ui_warning(d.get("text", ""))

        elif t == EventType.ERROR:
            ui_error(d.get("text", ""))

        elif t == EventType.COMMAND_REQUESTED:
            ui_command(d.get("text", ""), dangerous=d.get("dangerous", False))

        elif t == EventType.COMMAND_EXECUTED:
            if not d.get("success", True):
                ui_warning(f"Команда не выполнена: {d.get('text', '')}")

        elif t == EventType.COMMAND_CANCELLED:
            ui_warning(f"Команда отменена: {d.get('text', '')}")

        elif t == EventType.TOOL_CALL:
            icon = "🌐" if d.get("kind") == "search" else "📄"
            ui_step(icon, f"Инструмент: {d.get('kind')} → {d.get('arg')}", "cyan")

        elif t == EventType.TOOL_RESULT:
            ui_step("📊", f"Результат: {d.get('chars', 0)} символов", "dim")

        elif t == EventType.VOICE_STATE:
            # В консоли состояние не отображаем — для этого есть WebUI
            pass

        elif t == EventType.SHUTDOWN:
            self._stop.set()

        # прочие типы игнорируем

    # ============================================================
    # Запуск
    # ============================================================
    def run(self) -> None:
        """Запускает ядро в фоне, читает события в главном потоке."""
        self._core_thread = threading.Thread(
            target=self.core.run, name="JarvisCore", daemon=True,
        )
        self._core_thread.start()

        try:
            while not self._stop.is_set():
                try:
                    event = self.core.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.handle(event)
                if event.type == EventType.SHUTDOWN:
                    break
        except KeyboardInterrupt:
            console.print()
            console.print("[dim]Получен Ctrl+C, завершаю работу...[/dim]")

        # Просим ядро остановиться и ждём его
        self.core.request_shutdown()
        if self._core_thread is not None:
            self._core_thread.join(timeout=10)
        LOGGER.write("UI", "ConsoleUI завершён")