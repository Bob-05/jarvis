"""Семантическая память: объединяет историю и эмбеддинги."""
from jarvis import constants as C
from jarvis.llm.lm_studio import LMStudioManager
from jarvis.memory.embeddings import EmbeddingStore, cosine_similarity
from jarvis.memory.history import HistoryStore
from jarvis.ui.console import ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


# Fallback на случай, если константы нет в constants.py
_DEFAULT_RECENT = 10


def _recent_limit_default() -> int:
    return int(getattr(C, "RECENT_HISTORY_COUNT", _DEFAULT_RECENT))


class SemanticMemory:
    def __init__(self, history: HistoryStore, embeddings: EmbeddingStore,
                 llm: LMStudioManager, embedding_model: str,
                 instruction: str = C.EMBEDDING_DEFAULT_INSTRUCTION):
        self.history = history
        self.embeddings = embeddings
        self.llm = llm
        self.embedding_model = embedding_model or ""
        # Пустая строка => не оборачивать вручную (полагаемся на автошаблон LM Studio)
        self.instruction = (instruction or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.embedding_model)

    # ---------- Служебное ----------
    def _format_query(self, text: str) -> str:
        if not self.instruction:
            return text
        return f"Instruct: {self.instruction}\nQuery: {text}"

    def _embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        if not self.enabled or not texts:
            return []
        result: list[list[float] | None] = [None] * len(texts)
        for i in range(0, len(texts), C.EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + C.EMBEDDING_BATCH_SIZE]
            try:
                vecs = self.llm.embed(batch, model=self.embedding_model)
            except Exception as e:
                LOGGER.write("MEMORY", f"embed batch ошибка: {e}")
                continue
            for j, v in enumerate(vecs):
                if i + j < len(result):
                    result[i + j] = v
        return result

    def _embed_query(self, text: str) -> list[float] | None:
        if not self.enabled or not text.strip():
            return None
        vecs = self._embed_texts([self._format_query(text)])
        return vecs[0] if vecs else None

    def _embed_documents(self, texts: list[str]) -> list[list[float] | None]:
        return self._embed_texts(texts)

    # ---------- Проверка смены модели ----------
    def _probe_dim(self) -> int:
        """Запрашивает у эмбеддера размерность вектора на коротком тексте."""
        try:
            vecs = self.llm.embed(["ping"], model=self.embedding_model)
        except Exception as e:
            LOGGER.write("MEMORY", f"probe_dim ошибка: {e}")
            return 0
        if vecs and vecs[0]:
            return len(vecs[0])
        return 0

    # ---------- Публичное API ----------
    def ensure_embeddings(self, silent: bool = False) -> int:
        """Гарантирует, что все записи истории имеют вектор.

        Если модель/размерность эмбеддера изменилась с прошлого запуска —
        автоматически сбрасывает старые вектора и считает всё заново.
        """
        if not self.enabled:
            return 0

        # --- Проверка смены модели ---
        current_dim = self._probe_dim()
        if current_dim == 0:
            ui_warning("Эмбеддер не отвечает, семантический поиск пропущен.")
            return 0

        if not self.embeddings.meta_matches(self.embedding_model, current_dim):
            old_model = self.embeddings.meta_model or "(неизвестна)"
            old_dim = self.embeddings.meta_dim or self.embeddings.dim() or "?"
            ui_warning(
                f"Модель эмбеддингов изменилась: "
                f"'{old_model}' ({old_dim}) → "
                f"'{self.embedding_model}' ({current_dim}). "
                f"Пересчитываю все вектора..."
            )
            self.embeddings.clear()
            self.embeddings.set_meta(self.embedding_model, current_dim)

        # --- Досчёт отсутствующих ---
        entries = self.history.all()
        missing = [e for e in entries if not self.embeddings.has(e["id"])]
        if not missing:
            return 0

        if not silent:
            ui_step("🧠", f"Считаю эмбеддинги для {len(missing)} записей...", "yellow")
        vecs = self._embed_documents([e["text"] for e in missing])
        added = 0
        for e, v in zip(missing, vecs):
            if v is not None:
                self.embeddings.set(e["id"], v)
                added += 1
        if added:
            self.embeddings.set_meta(self.embedding_model, current_dim)
            self.embeddings.save()
        if not silent:
            ui_success(f"Готово: {added} новых векторов ({current_dim} dims)")
        LOGGER.write("MEMORY", f"ensure_embeddings: +{added}, dim={current_dim}")
        return added

    def rebuild(self) -> int:
        """Полный пересчёт всех векторов под текущую модель."""
        if not self.enabled:
            return 0
        self.embeddings.clear()
        entries = self.history.all()
        if not entries:
            return 0

        current_dim = self._probe_dim()
        if current_dim == 0:
            ui_warning("Эмбеддер не отвечает, пересчёт отменён.")
            return 0

        ui_step("♻️", f"Пересчёт эмбеддингов для {len(entries)} записей "
                      f"(dim={current_dim})...", "yellow")
        vecs = self._embed_documents([e["text"] for e in entries])
        added = 0
        for e, v in zip(entries, vecs):
            if v is not None:
                self.embeddings.set(e["id"], v)
                added += 1
        self.embeddings.set_meta(self.embedding_model, current_dim)
        self.embeddings.save()
        ui_success(f"Пересчитано: {added}/{len(entries)}")
        return added

    # ---------- Последние записи истории ----------
    def recent(self, limit: int | None = None) -> list[dict]:
        """Возвращает последние N записей истории (от старых к новым).

        Используется для гарантированного подмешивания контекста,
        когда семантический поиск не даёт всех нужных совпадений.

        Args:
            limit: сколько записей вернуть. Если None — берётся из
                constants.RECENT_HISTORY_COUNT (или 10 по умолчанию).
        """
        if limit is None:
            limit = _recent_limit_default()
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = _recent_limit_default()

        entries = self.history.all()
        if not entries or limit <= 0:
            return []
        return entries[-limit:] if len(entries) > limit else list(entries)

    def add_turn(self, role: str, text: str) -> dict:
        """Добавляет запись в историю и сразу считает её вектор."""
        entry = self.history.append(role, text)
        if not self.enabled:
            return entry

        # Актуализируем метаданные, если их ещё нет
        if not self.embeddings.meta_model or self.embeddings.meta_dim == 0:
            current_dim = self._probe_dim()
            if current_dim:
                self.embeddings.set_meta(self.embedding_model, current_dim)

        vecs = self._embed_documents([text])
        if vecs and vecs[0] is not None:
            v = vecs[0]
            if self.embeddings.meta_dim and len(v) != self.embeddings.meta_dim:
                LOGGER.write("MEMORY",
                             f"add_turn: dim mismatch {len(v)} != "
                             f"{self.embeddings.meta_dim}, пропуск")
                return entry
            self.embeddings.set(entry["id"], v)
            self.embeddings.save()
        return entry

    @staticmethod
    def _should_skip_recall(prompt: str) -> bool:
        p = prompt.strip().lower()
        if len(p) < C.SEMANTIC_MIN_PROMPT_LEN:
            return True
        if p in C.SEMANTIC_SKIP_UTTERANCES:
            return True
        return False

    def recall(self, query: str, top_k: int = C.SEMANTIC_TOP_K_DEFAULT) -> list[dict]:
        """Возвращает top_k наиболее релевантных записей (или [])."""
        if not self.enabled or top_k <= 0:
            return []
        if self._should_skip_recall(query):
            return []
        if not self.embeddings.vectors:
            return []

        qvec = self._embed_query(query)
        if qvec is None:
            return []

        qdim = len(qvec)
        scored: list[tuple[float, dict]] = []
        skipped_dim_mismatch = 0

        for entry in self.history.all():
            v = self.embeddings.get(entry["id"])
            if v is None:
                continue
            if len(v) != qdim:
                skipped_dim_mismatch += 1
                continue
            scored.append((cosine_similarity(qvec, v), entry))

        if skipped_dim_mismatch:
            LOGGER.write("MEMORY",
                         f"recall: пропущено {skipped_dim_mismatch} векторов "
                         f"чужой размерности (нужно {qdim})")

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[:top_k]]
        LOGGER.write("MEMORY", f"recall '{query[:40]}…' → top={len(top)}")
        return top

    def clear(self, clear_embeddings_too: bool = True) -> None:
        self.history.clear()
        if clear_embeddings_too:
            self.embeddings.clear()
        LOGGER.write("MEMORY",
                     f"Сброс памяти: history=clean, embeddings={clear_embeddings_too}")