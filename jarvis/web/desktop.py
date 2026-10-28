"""Десктопный запуск через pywebview + системный трей + панель настроек.

ВАЖНО: JsApi хранит config и core в ПРИВАТНЫХ атрибутах (с "_"),
потому что pywebview пытается интроспектировать публичные атрибуты,
уходить вглубь объектного графа и падать на методах типа
`core.memory.embeddings.prune` (unsupported callable).
"""
import threading

from jarvis import constants as C
from jarvis.config import AppConfig
from jarvis.core.jarvis_core import JarvisCore
from jarvis.ui.console import ui_error, ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


class JsApi:
    """API для вызова Python-функций из JS.

    Все внутренние объекты хранятся под приватными именами (_config, _core),
    чтобы pywebview не пытался их интроспектировать.
    """

    def __init__(self, config: AppConfig, core: JarvisCore,
                 exiting: "threading.Event | None" = None):
        self._config = config
        self._core = core
        # Общий флаг «идёт настоящий выход». Его проверяет хук closing в
        # run_desktop: пока флаг снят, закрытие окна = «свернуть в трей».
        self._exiting = exiting

    # ============================================================
    # Настройки: чтение
    # ============================================================
    def get_settings(self) -> dict:
        c = self._config
        return {
            "wake_word": c.wake_word,
            "require_wake_word": c.require_wake_word,
            "tts_engine": c.tts_engine,
            "stt_engine": c.stt_engine,
            "silero_speaker": c.silero_speaker,
            "piper_model_path": c.piper_model_path,
            "vad_threshold": c.vad_threshold,
            "semantic_top_k": c.semantic_top_k,
            "embedding_model_key": c.embedding_model_key,
            "embedding_instruction": c.embedding_instruction,
            "search_engine_priority": c.search_engine_priority,
            "web_search_max_results": c.web_search_max_results,
            "web_fetch_max_chars": c.web_fetch_max_chars,
            "max_tool_iterations": c.max_tool_iterations,
            "searxng_enabled": c.searxng_enabled,
            "searxng_auto_start": c.searxng_auto_start,
            "searxng_auto_stop": c.searxng_auto_stop,
            "searxng_stop_mode": c.searxng_stop_mode,
            "lm_studio_url": c.lm_studio_url,
            "lm_studio_auto_start": c.lm_studio_auto_start,
            "lm_studio_cli_path": c.lm_studio_cli_path,
            "lm_studio_auto_stop": c.lm_studio_auto_stop,
            "lm_studio_idle_stop_seconds": c.lm_studio_idle_stop_seconds,
            "lm_studio_ttl_seconds": c.lm_studio_ttl_seconds,
            "llm_limits_enabled": c.llm_limits_enabled,
            "llm_gpu_ratio": c.llm_gpu_ratio,
            "llm_context_length": c.llm_context_length,
            "llm_process_priority": c.llm_process_priority,
            "history_file": c.history_file,
            "embeddings_file": c.embeddings_file,
            "log_file": c.log_file,
            "verbose_logging": c.verbose_logging,
            "default_model_key": c.default_model_key,
        }

    # ============================================================
    # Настройки: сохранение
    # ============================================================
    def save_settings(self, data: dict) -> dict:
        """Сохраняет настройки в config.json И применяет их к работающему ядру."""
        try:
            c = self._config

            if "wake_word" in data:
                c.wake_word = str(data["wake_word"]).strip() or "джарвис"
            if "require_wake_word" in data:
                c.require_wake_word = bool(data["require_wake_word"])
            if "tts_engine" in data:
                c.tts_engine = str(data["tts_engine"])
            if "stt_engine" in data:
                c.stt_engine = str(data["stt_engine"])
            if "silero_speaker" in data:
                c.silero_speaker = str(data["silero_speaker"]).strip()
            if "piper_model_path" in data:
                c.piper_model_path = str(data["piper_model_path"]).strip()
            if "vad_threshold" in data:
                try:
                    c.vad_threshold = int(data["vad_threshold"])
                except (TypeError, ValueError):
                    pass
            if "semantic_top_k" in data:
                c.semantic_top_k = int(data["semantic_top_k"])
                c.semantic_search_enabled = c.semantic_top_k > 0
            if "embedding_model_key" in data:
                c.embedding_model_key = str(data["embedding_model_key"]).strip()
            if "embedding_instruction" in data:
                c.embedding_instruction = str(data["embedding_instruction"]).strip()
            if "search_engine_priority" in data:
                c.search_engine_priority = str(data["search_engine_priority"])
            if "web_search_max_results" in data:
                try:
                    c.web_search_max_results = max(1, int(data["web_search_max_results"]))
                except (TypeError, ValueError):
                    pass
            if "web_fetch_max_chars" in data:
                try:
                    c.web_fetch_max_chars = max(200, int(data["web_fetch_max_chars"]))
                except (TypeError, ValueError):
                    pass
            if "max_tool_iterations" in data:
                try:
                    c.max_tool_iterations = max(1, int(data["max_tool_iterations"]))
                except (TypeError, ValueError):
                    pass
            if "searxng_enabled" in data:
                c.searxng_enabled = bool(data["searxng_enabled"])
            if "searxng_auto_start" in data:
                c.searxng_auto_start = bool(data["searxng_auto_start"])
            if "searxng_auto_stop" in data:
                c.searxng_auto_stop = bool(data["searxng_auto_stop"])
            if "searxng_stop_mode" in data:
                c.searxng_stop_mode = str(data["searxng_stop_mode"])
            if "lm_studio_url" in data:
                c.lm_studio_url = str(data["lm_studio_url"]).strip()
            if "lm_studio_auto_start" in data:
                c.lm_studio_auto_start = bool(data["lm_studio_auto_start"])
            if "lm_studio_cli_path" in data:
                c.lm_studio_cli_path = str(data["lm_studio_cli_path"]).strip()
            if "lm_studio_auto_stop" in data:
                c.lm_studio_auto_stop = bool(data["lm_studio_auto_stop"])
            if "lm_studio_idle_stop_seconds" in data:
                try:
                    c.lm_studio_idle_stop_seconds = max(
                        0, int(data["lm_studio_idle_stop_seconds"])
                    )
                except (TypeError, ValueError):
                    pass
            if "lm_studio_ttl_seconds" in data:
                try:
                    c.lm_studio_ttl_seconds = max(10, int(data["lm_studio_ttl_seconds"]))
                except (TypeError, ValueError):
                    pass

            # --- Ресурсы LLM ---
            if "llm_limits_enabled" in data:
                c.llm_limits_enabled = bool(data["llm_limits_enabled"])
            if "llm_gpu_ratio" in data:
                try:
                    c.llm_gpu_ratio = max(0.0, min(1.0, float(data["llm_gpu_ratio"])))
                except (TypeError, ValueError):
                    pass
            if "llm_context_length" in data:
                try:
                    c.llm_context_length = max(0, int(data["llm_context_length"]))
                except (TypeError, ValueError):
                    pass
            if "llm_process_priority" in data:
                prio = str(data["llm_process_priority"]).strip().lower()
                if prio in ("normal", "below_normal", "low"):
                    c.llm_process_priority = prio

            if "history_file" in data:
                c.history_file = str(data["history_file"]).strip() or c.history_file
            if "embeddings_file" in data:
                c.embeddings_file = str(data["embeddings_file"]).strip() or c.embeddings_file
            if "log_file" in data:
                c.log_file = str(data["log_file"]).strip() or c.log_file
            if "verbose_logging" in data:
                c.verbose_logging = bool(data["verbose_logging"])

            c.save(C.CONFIG_FILE)

            result = self._core.apply_settings(data)
            applied = result.get("applied", [])
            restart_required = result.get("restart_required", [])

            parts = ["Настройки сохранены."]
            if applied:
                parts.append(f"Применено: {', '.join(applied)}.")
            if restart_required:
                parts.append(
                    f"⚠ Требуют перезапуска приложения: "
                    f"{', '.join(restart_required)}."
                )
            return {"ok": True, "message": " ".join(parts)}
        except Exception as e:
            LOGGER.write("ERROR", f"save_settings: {e}")
            return {"ok": False, "message": f"Ошибка: {e}"}

    # ============================================================
    # Системный промпт
    # ============================================================
    def get_system_prompt(self) -> str:
        from pathlib import Path
        try:
            p = Path(C.SYSTEM_PROMPT_FILE)
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception as e:
            LOGGER.write("ERROR", f"get_system_prompt: {e}")
        return ""

    def save_system_prompt(self, text: str) -> dict:
        from pathlib import Path
        try:
            text = str(text or "")
            Path(C.SYSTEM_PROMPT_FILE).write_text(text, encoding="utf-8")
            stripped = text.strip()
            if stripped:
                self._core.system_prompt = stripped
            return {
                "ok": True,
                "message": "Промпт сохранён. Применится к следующим запросам.",
            }
        except Exception as e:
            LOGGER.write("ERROR", f"save_system_prompt: {e}")
            return {"ok": False, "message": f"Ошибка: {e}"}

    # ============================================================
    # Память
    # ============================================================
    def clear_history(self, mode: str = "dialogs") -> dict:
        try:
            mem = self._core.memory
            if mem is None:
                return {"ok": False, "message": "Память не инициализирована"}
            if mode == "dialogs":
                mem.history.clear()
                mem.embeddings.clear()
                return {"ok": True, "message": "История и эмбеддинги очищены"}
            elif mode == "all":
                mem.clear(clear_embeddings_too=True)
                return {"ok": True, "message": "Вся память очищена"}
            elif mode == "rebuild":
                n = mem.rebuild()
                return {"ok": True, "message": f"Пересчитано {n} записей"}
            return {"ok": False, "message": "Неизвестный режим"}
        except Exception as e:
            LOGGER.write("ERROR", f"clear_history: {e}")
            return {"ok": False, "message": str(e)}

    # ============================================================
    # Списки моделей
    # ============================================================
    def list_llm_models(self) -> list:
        try:
            models = self._core.lm.list_llm_models()
            current = self._core.lm.current_model_key or ""
            return [
                {
                    "key": m["key"],
                    "display_name": m.get("display_name") or m["key"],
                    "params_string": m.get("params_string") or "",
                    "active": m["key"] == current,
                }
                for m in models
            ]
        except Exception as e:
            LOGGER.write("ERROR", f"list_llm_models: {e}")
            return []

    def list_embedding_models(self) -> list:
        try:
            models = self._core.lm.list_embedding_models()
            current = self._config.embedding_model_key or ""
            return [
                {
                    "key": m["key"],
                    "display_name": m.get("display_name") or m["key"],
                    "active": m["key"] == current,
                }
                for m in models
            ]
        except Exception as e:
            LOGGER.write("ERROR", f"list_embedding_models: {e}")
            return []

    # ============================================================
    # Смена LLM на лету
    # ============================================================
    def set_llm_model(self, model_key: str) -> dict:
        model_key = (model_key or "").strip()
        if not model_key:
            return {"ok": False, "message": "Не указан ключ модели"}
        try:
            lm = self._core.lm
            lm.unload()
            ok = lm.load_model(model_key)
            if not ok:
                return {"ok": False,
                        "message": f"Не удалось загрузить '{model_key}'"}
            lm.current_model_key = model_key
            lm.default_model_key = model_key
            self._config.default_model_key = model_key
            self._config.save(C.CONFIG_FILE)
            return {"ok": True, "message": f"Модель загружена: {model_key}"}
        except Exception as e:
            LOGGER.write("ERROR", f"set_llm_model: {e}")
            return {"ok": False, "message": str(e)}

    # ============================================================
    # ОСТАНОВКА ПРИЛОЖЕНИЯ (кнопка ⏻)
    # ============================================================
    def quit(self) -> dict:
        LOGGER.write("APP", "Получена команда quit() из UI")
        # Взводим флаг ДО destroy(): иначе хук closing вернёт False, окно
        # лишь свернётся в трей, webview.start() не вернётся и
        # shutdown_cleanup() (остановка LM Studio) не выполнится.
        if self._exiting is not None:
            self._exiting.set()
        try:
            self._core.request_shutdown()
        except Exception as e:
            LOGGER.write("ERROR", f"quit: request_shutdown: {e}")

        def _async_destroy():
            import time
            time.sleep(0.4)
            try:
                import webview
                for w in list(webview.windows):
                    try:
                        w.destroy()
                    except Exception as e:
                        LOGGER.write("ERROR", f"quit: destroy window: {e}")
            except Exception as e:
                LOGGER.write("ERROR", f"quit: import webview: {e}")

        threading.Thread(target=_async_destroy, name="JarvisQuit",
                         daemon=True).start()
        return {"ok": True, "message": "Останавливаю J.A.R.V.I.S."}


def run_desktop(core: JarvisCore, config: AppConfig, host: str = "127.0.0.1",
                port: int = 8765) -> None:
    """Запускает Web UI в нативном окне pywebview + системный трей."""
    try:
        import webview
    except ImportError:
        ui_error("pywebview не установлен. Выполните: pip install pywebview")
        ui_warning("Переключаюсь на консольный режим.")
        from jarvis.ui.console_ui import ConsoleUI
        ConsoleUI(core).run()
        return

    from jarvis.web import WebServer
    from jarvis.web.tray import PYSTRAY_AVAILABLE, create_tray_icon

    server = WebServer(core, host=host, port=port)
    server_thread = threading.Thread(target=server.run, name="WebServer", daemon=True)
    server_thread.start()

    ui_step("⏳", "Ожидание запуска Web-сервера...", "yellow")
    if not server.ready_event.wait(timeout=30):
        ui_error("Web-сервер не запустился за 30 секунд.")
        return

    ui_success("Web-сервер готов, открываю окно...")
    url = f"http://{host}:{port}"
    # Флаг «идёт настоящий выход»: взводится кнопкой ⏻ и пунктом «Выход»
    # в трее. Пока снят, клик по «X» сворачивает окно в трей.
    exiting = threading.Event()
    js_api = JsApi(config, core, exiting=exiting)

    window = webview.create_window(
        "J.A.R.V.I.S.",
        url,
        width=1400,
        height=900,
        min_size=(900, 600),
        background_color="#04090e",
        text_select=True,
        js_api=js_api,
    )

    tray_icon = None
    if PYSTRAY_AVAILABLE:
        def _quit_from_tray():
            exiting.set()
            try:
                core.request_shutdown()
            finally:
                try:
                    window.destroy()
                except Exception:
                    pass

        tray_icon = create_tray_icon(window, on_quit=_quit_from_tray)
        if tray_icon is not None:
            tray_thread = threading.Thread(
                target=tray_icon.run, name="Tray", daemon=True,
            )
            tray_thread.start()
            ui_step("📌", "Иконка добавлена в системный трей", "cyan")
        else:
            ui_warning("Не удалось создать иконку трея.")
    else:
        ui_warning("pystray не установлен — трей недоступен. "
                   "Установите: pip install pystray Pillow")

    if tray_icon is not None:
        def on_closing():
            # Настоящий выход (кнопка ⏻ / «Выход» в трее): разрешаем окну
            # закрыться, чтобы webview.start() вернулся и в finally
            # выполнилась очистка LM Studio. Возврат False отменил бы и
            # программный destroy() — окно стало бы «неубиваемым».
            if exiting.is_set():
                return
            # Пользователь закрыл окно кнопкой «X» — просто прячем в трей.
            try:
                window.hide()
            except Exception:
                pass
            return False

        try:
            window.events.closing += on_closing
        except Exception as e:
            LOGGER.write("TRAY", f"closing hook error: {e}")

    try:
        webview.start(debug=False, http_server=False, private_mode=False)
    finally:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        core.request_shutdown()
        ui_step("🛑", "Окно закрыто, сервер остановлен", "dim")