"""Выполнение shell-команд с проверками безопасности."""
import subprocess

from jarvis import constants as C
from jarvis.ui.console import console, ui_error, ui_success

from rich.box import DOUBLE
from rich.panel import Panel
from rich.text import Text


class CommandRunner:
    """Инкапсулирует классификацию, подтверждение и выполнение команд."""

    @staticmethod
    def is_dangerous(cmd: str) -> bool:
        low = cmd.lower()
        if low.startswith(C.URL_PREFIXES):
            return False
        if any(ch in cmd for ch in C.INJECTION_CHARS):
            return True
        for kw in C.SAFE_KEYWORDS:
            if low.startswith(kw):
                return False
        return any(kw in low for kw in C.DANGEROUS_KEYWORDS)

    @staticmethod
    def execute(command: str) -> bool:
        if command.lower().startswith("appopener open "):
            try:
                from AppOpener import open as app_open
                name = command[len("appopener open "):].strip().strip("\"'")
                app_open(name)
                ui_success(f"Приложение запущено: {name}")
                return True
            except ImportError:
                ui_error("Библиотека AppOpener не установлена")
                return False
            except Exception as e:
                ui_error(f"Ошибка AppOpener: {e}")
                return False
        try:
            subprocess.Popen(command, shell=True)
            ui_success("Команда выполнена")
            return True
        except Exception as e:
            ui_error(f"Ошибка: {e}")
            return False

    @staticmethod
    def text_confirm() -> bool:
        console.print(Panel(
            Text("Введите 'ДА' для подтверждения или 'НЕТ' для отмены",
                 style="bold yellow", justify="center"),
            title="[bold red]⚠  ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ  ⚠[/bold red]",
            border_style="red", box=DOUBLE,
        ))
        try:
            ans = console.input("[bold yellow]➤ [/bold yellow]").strip().lower()
            return ans in ("да", "yes", "y", "д")
        except (EOFError, KeyboardInterrupt):
            return False