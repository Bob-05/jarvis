"""Файловый логгер-синглтон."""
import re
import threading
import time
from pathlib import Path

from jarvis import constants as C


class FileLogger:
    """Пишет события в файл. Консоль — через ui_* функции."""

    def __init__(self):
        self.enabled = False
        self.path = Path(C.LOG_FILE_DEFAULT)
        self._lock = threading.Lock()

    def configure(self, enabled: bool, path: Path | None = None) -> None:
        self.enabled = enabled
        if path is not None:
            self.path = path

    def write(self, level: str, text: str) -> None:
        if not self.enabled:
            return
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            # Убираем только rich-теги: латиница/подчёркивания/пробел внутри [..]
            clean = re.sub(r"\[/?[a-z_\s]+\]", "", str(text))
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] [{level}] {clean}\n")
        except Exception:
            pass

    def clear(self) -> bool:
        try:
            if self.path.exists():
                self.path.write_text("", encoding="utf-8")
            return True
        except Exception:
            return False

    def file_size_kb(self) -> int:
        try:
            if self.path.exists():
                return self.path.stat().st_size // 1024
        except Exception:
            pass
        return 0


_logger = FileLogger()


def get_logger() -> FileLogger:
    return _logger