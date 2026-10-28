"""История диалогов без ограничения по длине."""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from jarvis import constants as C
from jarvis.ui.console import ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


def _new_id() -> str:
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class HistoryStore:
    """Хранит список записей вида {id, ts, role, text}.

    Защищён threading.Lock — можно безопасно писать из нескольких потоков.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.entries: list[dict] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.entries = []
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                ui_warning(f"Не удалось прочитать историю: {e}")
                self.entries = []
                return

            if not isinstance(raw, list):
                self.entries = []
                return

            migrated: list[dict] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if "text" in item:
                    entry = {
                        "id": item.get("id") or _new_id(),
                        "ts": item.get("ts") or _now_iso(),
                        "role": item.get("role", "user"),
                        "text": item.get("text", ""),
                    }
                elif "content" in item:
                    entry = {
                        "id": _new_id(),
                        "ts": _now_iso(),
                        "role": item.get("role", "user"),
                        "text": item.get("content", ""),
                    }
                else:
                    continue
                if entry["text"]:
                    migrated.append(entry)

            self.entries = migrated
            ui_success(f"История загружена · {len(self.entries)} сообщений")

    def save(self) -> None:
        with self._lock:
            try:
                self.path.write_text(
                    json.dumps(self.entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                ui_warning(f"Не удалось сохранить историю: {e}")

    def append(self, role: str, text: str) -> dict:
        with self._lock:
            entry = {
                "id": _new_id(),
                "ts": _now_iso(),
                "role": role,
                "text": text,
            }
            self.entries.append(entry)
            try:
                self.path.write_text(
                    json.dumps(self.entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                ui_warning(f"Не удалось сохранить историю: {e}")
            return entry

    def all(self) -> list[dict]:
        with self._lock:
            # Возвращаем shallow copy списка
            return list(self.entries)

    def clear(self) -> None:
        with self._lock:
            self.entries = []
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception:
                pass
        LOGGER.write("MEMORY", "История диалогов очищена")