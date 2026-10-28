"""JarvisCore — бизнес-логика ассистента без привязки к UI.

Класс JarvisCore инкапсулирует:
  * инициализацию STT/TTS/LLM/Web/SearXNG
  * основной цикл: запись → распознавание → запрос к LLM → озвучка
  * обработку текстовых запросов из UI (через очередь)
  * выполнение команд с проверками безопасности
  * работу с семантической памятью
  * применение настроек «на лету» (apply_settings)

Поток работы:
  core.run() — блокирующий цикл (микрофон). Запускать в отдельном потоке.
  core.submit_text(text) — неблокирующее добавление текстового запроса.
  core.event_queue — queue.Queue событий для UI-слоя.
  core.request_shutdown() — сигнал остановки.

Гарантия: handle_user_request() всегда вызывается под _request_lock,
поэтому в один момент времени обрабатывается ровно один запрос.

Дополнительно: флаг _busy предотвращает idle-остановку LM Studio,
пока идёт обработка запроса (иначе возникает гонка «stop → start → stop»).

Очисткой LM Studio при завершении занимается main.py / jarvis_exe.py
через единый метод lm.shutdown_cleanup(). В finally run() НЕ вызывается
unload, чтобы не держать _lock во время subprocess.
"""
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable

import sounddevice as sd

from jarvis import constants as C
from jarvis.audio.recorder import VoiceRecorder
from jarvis.config import AppConfig
from jarvis.core.events import Event, EventType
from jarvis.llm.lm_studio import LMStudioManager
from jarvis.memory.semantic import SemanticMemory
from jarvis.stt import KairosSTT, VoskSTT
from jarvis.tools.commands import CommandRunner
from jarvis.tools.searxng import SearxngManager
from jarvis.tools.web import WebTools
from jarvis.tts import PiperTTS, SileroTTS
from jarvis.ui.console import ui_error, ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger
from jarvis.utils import build_datetime_context, detect_language, strip_cjk

LOGGER = get_logger()

TOOL_TAG_RE = {
    "search": re.compile(r"\[ПОИСК\](.*?)\[/ПОИСК\]", re.DOTALL | re.IGNORECASE),
    "fetch":  re.compile(r"\[ЧТЕНИЕ\](.*?)\[/ЧТЕНИЕ\]", re.DOTALL | re.IGNORECASE),
}


class JarvisCore:
    def __init__(self, config: AppConfig, system_prompt: str,
                 stt_engine: str, tts_engine: str,
                 searxng: SearxngManager | None = None,
                 memory: SemanticMemory | None = None):
        self.config = config
        self.system_prompt = system_prompt
        self.wake_word = config.wake_word
        self.require_wake_word = config.require_wake_word
        self.vad_threshold = config.vad_threshold
        self.memory = memory

        # --- Событийная шина ---
        self.event_queue: queue.Queue[Event] = queue.Queue()
        self._shutdown = threading.Event()

        # --- Сериализация запросов ---
        self._request_lock = threading.Lock()
        self._text_queue: queue.Queue[str] = queue.Queue(maxsize=32)

        # --- Флаг «идёт обработка запроса» ---
        self._busy = threading.Event()

        # --- Callback подтверждения ---
        self._confirmation_callback: Callable[[], bool] | None = None

        # --- TTS ---
        if tts_engine == "silero":
            self.tts = SileroTTS(config)
        elif tts_engine == "piper":
            self.tts = PiperTTS(config)
        else:
            ui_error(f"Неизвестный TTS: {tts_engine}")
            raise SystemExit(1)

        # --- STT ---
        ui_step("👂", "Инициализация распознавания речи...", "yellow")
        if stt_engine == "kairos-asr":
            self.stt = KairosSTT()
        elif stt_engine in config.vosk_paths:
            self.stt = VoskSTT(config.vosk_paths[stt_engine])
        else:
            ui_error(f"Неизвестный STT: {stt_engine}")
            raise SystemExit(1)
        ui_success(f"STT готов · {self.stt.name}")

        # --- LLM ---
        # cli_path прокидываем, чтобы stop_server()/shutdown_cleanup()
        # могли найти lms.exe, если он лежит в нестандартном месте.
        # resource_limits — пользовательские ограничения нагрузки
        # (GPU/контекст/приоритет), применяются при `lms load`.
        self.lm = LMStudioManager(
            base_url=config.lm_studio_url,
            default_model_key=config.default_model_key,
            ttl_seconds=config.lm_studio_ttl_seconds,
            embedding_model_key=config.embedding_model_key,
            cli_path=config.lm_studio_cli_path,
            resource_limits=config.get_resource_limits(),
        )

        # --- Веб ---
        self.web = WebTools(
            max_results=config.web_search_max_results,
            max_chars=config.web_fetch_max_chars,
            searxng=searxng,
            priority=config.search_engine_priority,
        )

        # --- Микрофон ---
        try:
            dev_info = sd.query_devices(sd.default.device[0])
            self.native_rate = int(dev_info["default_samplerate"])
            ui_success(f"Микрофон: {dev_info['name']} · {self.native_rate} Гц")
        except Exception as e:
            self.native_rate = 48000
            ui_warning(f"Не удалось определить частоту микрофона: {e}. Использую 48000 Гц.")

        self.recorder = VoiceRecorder(self.native_rate, self.vad_threshold)

        # --- Запуск текстового воркера ---
        self._text_worker = threading.Thread(
            target=self._text_worker_loop,
            name="JarvisTextWorker",
            daemon=True,
        )
        self._text_worker.start()

    # ============================================================
    # События
    # ============================================================
    def emit(self, event_type: str, **data) -> None:
        self.event_queue.put(Event(type=event_type, data=data))

    def request_shutdown(self) -> None:
        self._shutdown.set()
        try:
            self._text_queue.put_nowait("")
        except queue.Full:
            pass

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown.is_set()

    @property
    def is_busy(self) -> bool:
        """True, если сейчас обрабатывается запрос (голос/текст)."""
        return self._busy.is_set()

    # ============================================================
    # Callback подтверждения
    # ============================================================
    def set_confirmation_callback(self, cb: Callable[[], bool] | None) -> None:
        self._confirmation_callback = cb

    def _request_text_confirmation(self) -> bool:
        if self._confirmation_callback is None:
            return CommandRunner.text_confirm()
        try:
            return bool(self._confirmation_callback())
        except Exception as e:
            LOGGER.write("ERROR", f"Ошибка callback подтверждения: {e}")
            return False

    # ============================================================
    # Применение настроек
    # ============================================================
    def apply_settings(self, data: dict) -> dict:
        applied: list[str] = []
        restart_required: list[str] = []

        # ----- Активация -----
        if "wake_word" in data:
            new_wake = str(data["wake_word"]).strip() or "джарвис"
            if new_wake != self.wake_word:
                self.wake_word = new_wake
                self.config.wake_word = new_wake
                applied.append("wake_word")

        if "require_wake_word" in data:
            new_req = bool(data["require_wake_word"])
            if new_req != self.require_wake_word:
                self.require_wake_word = new_req
                self.config.require_wake_word = new_req
                applied.append("require_wake_word")

        # ----- Семантический поиск -----
        if "semantic_top_k" in data:
            try:
                new_k = int(data["semantic_top_k"])
            except (TypeError, ValueError):
                new_k = self.config.semantic_top_k
            if new_k != self.config.semantic_top_k:
                self.config.semantic_top_k = new_k
                self.config.semantic_search_enabled = new_k > 0
                applied.append("semantic_top_k")

        if "embedding_instruction" in data:
            new_instr = str(data["embedding_instruction"]).strip()
            if new_instr != self.config.embedding_instruction:
                self.config.embedding_instruction = new_instr
                if self.memory is not None:
                    self.memory.instruction = new_instr
                applied.append("embedding_instruction")

        # ----- Поиск -----
        if "search_engine_priority" in data:
            new_prio = str(data["search_engine_priority"])
            if new_prio != self.config.search_engine_priority:
                self.config.search_engine_priority = new_prio
                if hasattr(self.web, "priority"):
                    self.web.priority = new_prio
                applied.append("search_engine_priority")

        if "web_search_max_results" in data:
            try:
                new_max = max(1, int(data["web_search_max_results"]))
            except (TypeError, ValueError):
                new_max = self.config.web_search_max_results
            if new_max != self.config.web_search_max_results:
                self.config.web_search_max_results = new_max
                if hasattr(self.web, "max_results"):
                    self.web.max_results = new_max
                applied.append("web_search_max_results")

        if "web_fetch_max_chars" in data:
            try:
                new_chars = max(200, int(data["web_fetch_max_chars"]))
            except (TypeError, ValueError):
                new_chars = self.config.web_fetch_max_chars
            if new_chars != self.config.web_fetch_max_chars:
                self.config.web_fetch_max_chars = new_chars
                if hasattr(self.web, "max_chars"):
                    self.web.max_chars = new_chars
                applied.append("web_fetch_max_chars")

        if "max_tool_iterations" in data:
            try:
                new_iter = max(1, int(data["max_tool_iterations"]))
            except (TypeError, ValueError):
                new_iter = self.config.max_tool_iterations
            if new_iter != self.config.max_tool_iterations:
                self.config.max_tool_iterations = new_iter
                applied.append("max_tool_iterations")

        # ----- VAD -----
        if "vad_threshold" in data:
            try:
                new_vad = int(data["vad_threshold"])
            except (TypeError, ValueError):
                new_vad = self.config.vad_threshold
            if new_vad != self.config.vad_threshold:
                self.config.vad_threshold = new_vad
                self.vad_threshold = new_vad
                if hasattr(self, "recorder") and self.recorder is not None:
                    self.recorder.vad_threshold = new_vad
                applied.append("vad_threshold")

        # ----- Логи -----
        if "verbose_logging" in data:
            new_verbose = bool(data["verbose_logging"])
            if new_verbose != self.config.verbose_logging:
                self.config.verbose_logging = new_verbose
                LOGGER.configure(enabled=new_verbose,
                                 path=Path(self.config.log_file))
                applied.append("verbose_logging")

        # ----- TTS: голос -----
        if "silero_speaker" in data:
            new_voice = str(data["silero_speaker"]).strip()
            if new_voice and new_voice != self.config.silero_speaker:
                self.config.silero_speaker = new_voice
                if hasattr(self.tts, "speaker"):
                    self.tts.speaker = new_voice
                applied.append("silero_speaker")

        # ----- LM Studio: авто-стоп и idle -----
        if "lm_studio_auto_stop" in data:
            new_auto_stop = bool(data["lm_studio_auto_stop"])
            if new_auto_stop != self.config.lm_studio_auto_stop:
                self.config.lm_studio_auto_stop = new_auto_stop
                if new_auto_stop:
                    self.lm.idle_stop_seconds = int(
                        self.config.lm_studio_idle_stop_seconds
                    )
                else:
                    self.lm.idle_stop_seconds = 0
                applied.append("lm_studio_auto_stop")

        if "lm_studio_idle_stop_seconds" in data:
            try:
                new_idle = max(0, int(data["lm_studio_idle_stop_seconds"]))
            except (TypeError, ValueError):
                new_idle = self.config.lm_studio_idle_stop_seconds
            if new_idle != self.config.lm_studio_idle_stop_seconds:
                self.config.lm_studio_idle_stop_seconds = new_idle
                if self.config.lm_studio_auto_stop:
                    self.lm.idle_stop_seconds = new_idle
                applied.append("lm_studio_idle_stop_seconds")

        if "lm_studio_ttl_seconds" in data:
            try:
                new_ttl = max(10, int(data["lm_studio_ttl_seconds"]))
            except (TypeError, ValueError):
                new_ttl = self.config.lm_studio_ttl_seconds
            if new_ttl != self.config.lm_studio_ttl_seconds:
                self.config.lm_studio_ttl_seconds = new_ttl
                if hasattr(self.lm, "ttl"):
                    self.lm.ttl = new_ttl
                applied.append("lm_studio_ttl_seconds")

        # ----- cli_path: применяем «на лету» -----
        if "lm_studio_cli_path" in data:
            new_cli = str(data["lm_studio_cli_path"]).strip()
            if new_cli != self.config.lm_studio_cli_path:
                self.config.lm_studio_cli_path = new_cli
                if hasattr(self.lm, "cli_path"):
                    self.lm.cli_path = new_cli
                applied.append("lm_studio_cli_path")

        # ----- Ресурсы LLM (GPU / контекст / приоритет) -----
        if any(k in data for k in (
            "llm_limits_enabled", "llm_gpu_ratio",
            "llm_context_length", "llm_process_priority"
        )):
            new_limits = self.config.get_resource_limits()
            cur = self.lm.resource_limits

            limits_changed = (
                new_limits.enabled != cur.enabled
                or abs(new_limits.gpu_ratio - cur.gpu_ratio) > 1e-6
                or new_limits.context_length != cur.context_length
            )
            # Сравниваем эффективный приоритет (при выключенных лимитах —
            # normal), чтобы ловить и смену самого режима, и выключение
            # ограничений, которое должно вернуть процессы в Normal.
            prio_changed = (
                new_limits.effective_priority() != cur.effective_priority()
            )

            if limits_changed or prio_changed:
                self.lm.set_resource_limits(new_limits)
            if limits_changed:
                applied.append("llm_resource_limits")
            if prio_changed:
                # Приоритет применяем к уже работающим процессам немедленно,
                # в т.ч. возврат в Normal при смене/выключении настроек.
                self.lm._apply_process_priority(force=True)
                applied.append("llm_process_priority")

        # ----- Требуют перезапуска -----
        if "tts_engine" in data and str(data["tts_engine"]) != self.config.tts_engine:
            restart_required.append("tts_engine")
        if "stt_engine" in data and str(data["stt_engine"]) != self.config.stt_engine:
            restart_required.append("stt_engine")
        if "embedding_model_key" in data \
                and str(data["embedding_model_key"]).strip() != self.config.embedding_model_key:
            restart_required.append("embedding_model_key")
        if "lm_studio_url" in data and str(data["lm_studio_url"]) != self.config.lm_studio_url:
            restart_required.append("lm_studio_url")
        if "lm_studio_auto_start" in data \
                and bool(data["lm_studio_auto_start"]) != self.config.lm_studio_auto_start:
            restart_required.append("lm_studio_auto_start")
        if "searxng_enabled" in data \
                and bool(data["searxng_enabled"]) != self.config.searxng_enabled:
            restart_required.append("searxng_enabled")
        if "searxng_stop_mode" in data \
                and str(data["searxng_stop_mode"]) != self.config.searxng_stop_mode:
            restart_required.append("searxng_stop_mode")
        if "history_file" in data \
                and str(data["history_file"]) != self.config.history_file:
            restart_required.append("history_file")
        if "embeddings_file" in data \
                and str(data["embeddings_file"]) != self.config.embeddings_file:
            restart_required.append("embeddings_file")
        if "log_file" in data and str(data["log_file"]) != self.config.log_file:
            restart_required.append("log_file")

        return {"applied": applied, "restart_required": restart_required}

    # ============================================================
    # Аудио / речь
    # ============================================================
    def record_until_silence(self):
        self.emit(EventType.VOICE_STATE, state="listening")
        result = self.recorder.record_until_silence()
        self.emit(EventType.VOICE_STATE, state="idle")
        return result

    def recognize(self, audio_data, fs):
        return self.stt.recognize(audio_data, fs)

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        self.emit(EventType.ASSISTANT_REPLIED, text=text, spoken=True)
        self.emit(EventType.VOICE_STATE, state="speaking")

        def on_amplitude(amps: list[float], chunk_sec: float) -> None:
            if amps:
                self.emit(EventType.TTS_AMPLITUDE,
                          amplitudes=amps, chunk_sec=chunk_sec)

        try:
            self.tts.speak(text, on_amplitude=on_amplitude)
        finally:
            self.emit(EventType.VOICE_STATE, state="idle")

    # ============================================================
    # Парсинг ответов LLM
    # ============================================================
    @staticmethod
    def parse_tool_calls(text: str):
        tools = []
        for kind, pattern in TOOL_TAG_RE.items():
            for m in pattern.finditer(text or ""):
                arg = m.group(1).strip()
                if arg:
                    tools.append((kind, arg))
        cleaned = TOOL_TAG_RE["search"].sub("", text or "")
        cleaned = TOOL_TAG_RE["fetch"].sub("", cleaned).strip()
        return tools, cleaned

    @staticmethod
    def parse(text: str):
        if "[ОТВЕТ]" not in text:
            return "", ""

        rest = text.split("[ОТВЕТ]", 1)[1]
        rest = rest.replace("[/ОТВЕТ]", "")

        if "[КОМАНДА]" not in rest:
            return rest.strip(), ""

        answer, command = rest.split("[КОМАНДА]", 1)
        answer = answer.strip()
        command = command.replace("[/КОМАНДА]", "").strip()

        if len(command) >= 2 and command[0] in ("\"", "'", "`") and command[-1] == command[0]:
            command = command[1:-1].strip()

        if command.lower() in C.NO_COMMAND_VALUES:
            command = ""

        return answer, command

    # ============================================================
    # Подтверждение опасных команд
    # ============================================================
    def voice_confirm(self) -> bool:
        self.speak("Подтвердите голосом: скажите 'да' или 'нет'.")
        audio, fs = self.record_until_silence()
        if audio is None:
            self.emit(EventType.STEP, icon="🔇",
                      text="Речь не распознана", color="dim")
            return False
        text = self.recognize(audio, fs)
        self.emit(EventType.STEP, icon="🔊", text=f"Распознано: {text}", color="dim")
        return any(w in text for w in ("да", "yes", "подтверждаю", "выполняй"))

    def confirm_dangerous(self, cmd: str) -> bool:
        self.emit(EventType.COMMAND_REQUESTED, text=cmd, dangerous=True)
        self.speak("Внимание, сэр! Эта команда может привести к необратимым последствиям.")
        self.speak("Подтвердите дважды: голосом и текстом.")
        if not self.voice_confirm():
            self.speak("Голосовое подтверждение не получено. Отмена.")
            return False
        if not self._request_text_confirmation():
            self.speak("Текстовое подтверждение не получено. Отмена.")
            return False
        self.speak("Подтверждение получено. Выполняю.")
        return True

    # ============================================================
    # Построение сообщений для LLM
    # ============================================================
    def _build_messages(self, prompt: str) -> list:
        lang = detect_language(prompt)
        if lang == "ru":
            lang_rule = ("Отвечай ТОЛЬКО на русском языке. "
                         "НИКОГДА не используй китайские иероглифы, "
                         "арабскую вязь и другие не-кириллические символы.")
        else:
            lang_rule = ("Answer ONLY in English. "
                         "NEVER use Chinese characters, Cyrillic letters, "
                         "or any non-Latin symbols.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": build_datetime_context()},
            {"role": "system", "content": lang_rule},
        ]

        seen_ids: set[str] = set()

        if self.memory is not None and self.memory.enabled:
            top_k = self.config.semantic_top_k
            if top_k > 0:
                relevant = self.memory.recall(prompt, top_k=top_k)
                if relevant:
                    ctx_lines = []
                    for r in relevant:
                        rid = r.get("id")
                        if rid:
                            seen_ids.add(rid)
                        ctx_lines.append(f"[{r['ts']}] {r['role']}: {r['text']}")
                    messages.append({
                        "role": "system",
                        "content": (
                            "РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ПРЕДЫДУЩИХ ДИАЛОГОВ "
                            "(от самого похожего к менее похожему):\n"
                            + "\n".join(ctx_lines)
                        ),
                    })

        if self.memory is not None:
            recent = self.memory.recent(limit=C.RECENT_HISTORY_COUNT)
            recent = [r for r in recent if r.get("id") not in seen_ids]
            if recent:
                recent_lines = [
                    f"[{r['ts']}] {r['role']}: {r['text']}"
                    for r in recent
                ]
                messages.append({
                    "role": "system",
                    "content": (
                        f"ПОСЛЕДНИЕ {len(recent)} СООБЩЕНИЙ ИЗ ИСТОРИИ "
                        f"ДИАЛОГА (от старого к новому) — используй их "
                        f"для понимания контекста разговора:\n"
                        + "\n".join(recent_lines)
                    ),
                })

        messages.append({"role": "user", "content": prompt})
        return messages

    def ask(self, prompt: str) -> str:
        messages = self._build_messages(prompt)
        LOGGER.write("LLM", f"Запрос к модели: {prompt!r}")

        max_iter = int(self.config.max_tool_iterations)
        final = ""

        for it in range(max_iter):
            try:
                raw = self.lm.chat(messages, temperature=0.3, max_tokens=400)
            except Exception as e:
                final = f"[ОТВЕТ]Ошибка обращения к модели: {e}"
                break

            LOGGER.write("LLM_RAW", f"Итерация {it + 1}: {raw!r}")

            tools, _ = self.parse_tool_calls(raw)
            if not tools:
                final = strip_cjk(raw)
                break

            messages.append({"role": "assistant", "content": raw})
            tool_reports = []
            for kind, arg in tools:
                self.emit(EventType.TOOL_CALL, kind=kind, arg=arg)
                if kind == "search":
                    result = self.web.search(arg)
                else:
                    result = self.web.fetch(arg)
                tool_reports.append(f"[РЕЗУЛЬТАТ {kind.upper()} «{arg}»]\n{result}")
                self.emit(EventType.TOOL_RESULT, kind=kind, arg=arg,
                          chars=len(result))
                LOGGER.write("TOOL", f"{kind}: {arg} -> {len(result)} символов")

            messages.append({
                "role": "user",
                "content": (
                    "\n\n".join(tool_reports)
                    + "\n\nТеперь ответь пользователю строго в формате "
                      "[ОТВЕТ]твой ответ[/ОТВЕТ] (и при необходимости [КОМАНДА]…). "
                      "Не вызывай инструменты повторно, если данных достаточно. "
                      "Не выдумывай факты — используй только результаты выше."
                ),
            })
        else:
            final = "[ОТВЕТ]Извините, сэр, не удалось уложиться в лимит шагов обработки."

        if final and "[ОТВЕТ]" not in final:
            final = f"[ОТВЕТ]{final.strip()}"

        final = final.replace("[/ОТВЕТ]", "").replace("[/КОМАНДА]", "")

        answer, _ = self.parse(final)
        if answer and self.memory is not None:
            try:
                self.memory.add_turn("user", prompt)
                self.memory.add_turn("assistant", answer)
            except Exception as e:
                LOGGER.write("MEMORY", f"add_turn ошибка: {e}")

        return final

    # ============================================================
    # Обработка запроса (голос/текст — под общим локом)
    # ============================================================
    def handle_user_request(self, clean_text: str) -> None:
        with self._request_lock:
            self._busy.set()
            try:
                self._handle_user_request_inner(clean_text)
            finally:
                self._busy.clear()

    def _handle_user_request_inner(self, clean_text: str) -> None:
        if not clean_text or len(clean_text) < 2:
            self.speak("Слушаю, сэр.")
            time.sleep(C.POST_TTS_COOLDOWN_SEC)
            return

        self.emit(EventType.STATUS, text="Обрабатываю...", color="yellow")
        self.emit(EventType.VOICE_STATE, state="thinking")

        try:
            raw = self.ask(clean_text)
            answer, command = self.parse(raw)

            if answer:
                self.speak(answer)
                time.sleep(C.POST_TTS_COOLDOWN_SEC)

            if command:
                if CommandRunner.is_dangerous(command):
                    if self.confirm_dangerous(command):
                        ok = CommandRunner.execute(command)
                        self.emit(EventType.COMMAND_EXECUTED,
                                  text=command, success=ok)
                    else:
                        self.emit(EventType.COMMAND_CANCELLED, text=command)
                        self.speak("Команда отменена.")
                        time.sleep(C.POST_TTS_COOLDOWN_SEC)
                else:
                    self.emit(EventType.COMMAND_REQUESTED,
                              text=command, dangerous=False)
                    ok = CommandRunner.execute(command)
                    self.emit(EventType.COMMAND_EXECUTED,
                              text=command, success=ok)
                    if ok:
                        self.speak("Готово, сэр.")
                    else:
                        self.speak("Не удалось выполнить.")
                    time.sleep(C.POST_TTS_COOLDOWN_SEC)
        finally:
            # Гарантированный сброс «thinking», если ответа/речи не было
            # (например, ошибка LLM) — иначе индикатор/орб «залипнет».
            self.emit(EventType.VOICE_STATE, state="idle")

    # ============================================================
    # Текстовый ввод (очередь + воркер)
    # ============================================================
    def submit_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            self._text_queue.put_nowait(text)
        except queue.Full:
            LOGGER.write("ERROR", "Очередь текстовых запросов переполнена")
            self.emit(EventType.ERROR,
                      text="Слишком много запросов — подождите немного.")

    def ask_text(self, text: str) -> None:
        self.submit_text(text)

    def _text_worker_loop(self) -> None:
        LOGGER.write("CORE", "Text worker запущен")
        while not self._shutdown.is_set():
            try:
                text = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text:
                continue
            self.emit(EventType.USER_SPOKE, text=text)
            try:
                self.handle_user_request(text)
            except Exception as e:
                LOGGER.write("ERROR", f"Text worker: {e}")
                self.emit(EventType.ERROR, text=f"Ошибка обработки: {e}")
        LOGGER.write("CORE", "Text worker остановлен")

    # ============================================================
    # Обслуживание простоя
    # ============================================================
    def _idle_maintenance(self) -> None:
        if self._busy.is_set():
            return
        try:
            self.lm.unload_if_idle()
            if self.config.lm_studio_auto_stop:
                self.lm.maybe_stop_server_by_idle()
        except Exception as e:
            LOGGER.write("ERROR", f"idle_maintenance: {e}")

    # ============================================================
    # Основной цикл (голосовой)
    # ============================================================
    def run(self):
        self.emit(EventType.STARTED)

        if self.require_wake_word:
            self.speak("Системы инициализированы, сэр. Обращайтесь по имени. Чем можем помочь?")
        else:
            self.speak("Системы инициализированы, сэр. Я слушаю вас постоянно.")

        self.emit(EventType.STEP, icon="⌨️",
                  text="Ctrl+C или сигнал shutdown — выход", color="dim")

        try:
            while not self._shutdown.is_set():
                audio, fs = self.record_until_silence()

                if audio is None:
                    self._idle_maintenance()
                    continue

                text = self.recognize(audio, fs)
                if not text:
                    self._idle_maintenance()
                    continue

                if self.require_wake_word:
                    if self.wake_word not in text:
                        self._idle_maintenance()
                        continue
                    clean = text.replace(self.wake_word, "").strip()
                else:
                    clean = text.strip()
                    if len(clean) < 3:
                        self._idle_maintenance()
                        continue

                self.emit(EventType.USER_SPOKE, text=text)
                self.handle_user_request(clean)

                self._idle_maintenance()
                time.sleep(0.2)

        except Exception as e:
            self.emit(EventType.ERROR, text=f"Сбой в основном цикле: {e}")
            LOGGER.write("ERROR", f"Сбой в основном цикле: {e}")
        finally:
            # НЕ вызываем здесь self.lm.unload(): поток ядра держал бы
            # self._lock во время subprocess, и stop_server в главном
            # потоке залипал бы в ожидании lock. Полной очисткой занимается
            # main.py / jarvis_exe.py через lm.shutdown_cleanup().
            self.emit(EventType.SHUTDOWN)