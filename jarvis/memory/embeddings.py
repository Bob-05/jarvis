"""Хранилище векторов и косинусная близость.

Формат файла (v2):
{
  "_meta": {"model": "text-embedding-...", "dim": 1024},
  "vectors": {"<entry_id>": [float, ...], ...}
}

Поддерживается миграция старого формата (плоский dict без _meta) —
при загрузке он подхватывается, но _meta остаётся пустой,
что заставит систему пересчитать эмбеддинги под текущую модель.
"""
import json
from pathlib import Path

import numpy as np

from jarvis.ui.console import ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()

_META_KEY = "_meta"


def cosine_similarity(a, b) -> float:
    """Косинусная близость двух векторов. 0.0, если размерности разные."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingStore:
    """Хранит вектора {entry_id: [float,...]} + метаданные о модели и dim."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.vectors: dict[str, list[float]] = {}
        self.meta_model: str = ""
        self.meta_dim: int = 0

    # ---------- Загрузка/сохранение ----------
    def load(self) -> None:
        if not self.path.exists():
            self.vectors = {}
            self.meta_model = ""
            self.meta_dim = 0
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            ui_warning(f"Не удалось прочитать эмбеддинги: {e}")
            self.vectors = {}
            return

        if not isinstance(raw, dict):
            self.vectors = {}
            return

        # Новый формат
        if _META_KEY in raw and isinstance(raw.get("vectors"), dict):
            meta = raw.get(_META_KEY) or {}
            self.meta_model = str(meta.get("model", ""))
            try:
                self.meta_dim = int(meta.get("dim", 0))
            except (TypeError, ValueError):
                self.meta_dim = 0
            self.vectors = {
                str(k): list(v) for k, v in raw["vectors"].items()
                if isinstance(v, list) and v
            }
        else:
            # Старый формат (плоский dict) — мигрируем, метаданные неизвестны
            self.vectors = {
                str(k): list(v) for k, v in raw.items()
                if isinstance(v, list) and v
            }
            self.meta_model = ""
            self.meta_dim = 0

    def save(self) -> None:
        try:
            payload = {
                _META_KEY: {"model": self.meta_model, "dim": self.meta_dim},
                "vectors": self.vectors,
            }
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            ui_warning(f"Не удалось сохранить эмбеддинги: {e}")

    # ---------- Метаданные ----------
    def set_meta(self, model: str, dim: int) -> None:
        self.meta_model = model or ""
        self.meta_dim = int(dim or 0)

    def meta_matches(self, model: str, dim: int) -> bool:
        """True, если модель и размерность совпадают с сохранёнными."""
        if not self.meta_model and self.meta_dim == 0 and not self.vectors:
            return True  # пустой файл — совпадает с чем угодно
        return self.meta_model == (model or "") and self.meta_dim == int(dim or 0)

    # ---------- Работа с векторами ----------
    def has(self, entry_id: str) -> bool:
        return entry_id in self.vectors

    def get(self, entry_id: str) -> list[float] | None:
        return self.vectors.get(entry_id)

    def set(self, entry_id: str, vector: list[float]) -> None:
        self.vectors[entry_id] = list(vector)

    def clear(self) -> None:
        self.vectors = {}
        self.meta_model = ""
        self.meta_dim = 0
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass
        LOGGER.write("MEMORY", "Эмбеддинги очищены")

    def prune(self, valid_ids: set[str]) -> int:
        """Удаляет векторы, для которых нет записей в истории."""
        removed = [k for k in self.vectors if k not in valid_ids]
        for k in removed:
            del self.vectors[k]
        if removed:
            self.save()
        return len(removed)

    def dim(self) -> int:
        """Возвращает фактическую размерность первого вектора (0 — если пусто)."""
        for v in self.vectors.values():
            return len(v)
        return 0