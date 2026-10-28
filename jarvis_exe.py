"""Точка входа для сборки в .exe."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    BASE_DIR = Path(sys.executable).parent
    MEIPASS_DIR = Path(sys._MEIPASS)
    sys.path.insert(0, str(BASE_DIR))
else:
    BASE_DIR = Path(__file__).parent
    MEIPASS_DIR = BASE_DIR

from jarvis import constants as C
from jarvis.audio.recorder import check_microphone
from jarvis.config import AppConfig
from jarvis.core import JarvisCore
from jarvis.llm.lm_studio import get_lm_studio_gui_pids
from jarvis.memory import EmbeddingStore, HistoryStore, SemanticMemory
from jarvis.tools.searxng import SearxngManager
from jarvis.ui.console import ui_error, ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


def _ensure_user_file(name: str) -> None:
    target = BASE_DIR / name
    if target.exists():
        return
    candidates = [
        MEIPASS_DIR / "jarvis_data" / name,
        MEIPASS_DIR / name,
    ]
    for src in candidates:
        try:
            if src.exists():
                shutil.copy2(src, target)
                print(f"[Jarvis] Создан {name} из встроенного шаблона")
                return
        except Exception as e:
            print(f"[Jarvis] Не удалось создать {name}: {e}")
            return
    print(f"[Jarvis] Встроенный шаблон {name} не найден")


def _ensure_lm_server(url: str, cli_path: str = "") -> tuple[bool, bool, set[int]]:
    """Возвращает (ok, started_by_us, gui_pids_before)."""
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=3)
        if r.status_code == 200:
            LOGGER.write("LM", "LM Studio уже запущен (не нами)")
            return True, False, set()
    except requests.RequestException:
        pass

    gui_pids_before = get_lm_studio_gui_pids()
    LOGGER.write("LM",
                 f"Снимок GUI до старта сервера: "
                 f"{len(gui_pids_before)} процессов")

    lms_exe: str | None = None
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            lms_exe = str(p)
    if not lms_exe:
        for cand in C.LM_STUDIO_CLI_CANDIDATES:
            try:
                if cand.exists():
                    lms_exe = str(cand)
                    break
            except Exception:
                continue

    if not lms_exe:
        ui_warning("lms.exe не найден — LM Studio не запущен. "
                   "Запустите его вручную.")
        return False, False, set()

    ui_step("🚀", f"Запуск LM Studio server через {Path(lms_exe).name}...", "yellow")
    try:
        subprocess.Popen(
            [lms_exe, "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception as e:
        ui_error(f"Не удалось запустить lms server start: {e}")
        return False, False, set()

    for _ in range(15):
        time.sleep(2)
        try:
            r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=3)
            if r.status_code == 200:
                ui_success("LM Studio server запущен (запущен Jarvis'ом)")
                LOGGER.write("LM", "LM Studio server запущен Jarvis'ом")
                return True, True, gui_pids_before
        except requests.RequestException:
            continue

    ui_error("LM Studio server не ответил за 30 секунд.")
    return False, False, set()


def main():
    os.chdir(BASE_DIR)

    _ensure_user_file("config.json")
    _ensure_user_file("system_prompt.txt")

    config = AppConfig.load(BASE_DIR / "config.json")
    LOGGER.configure(enabled=config.verbose_logging,
                     path=BASE_DIR / config.log_file)

    try:
        check_microphone()
    except Exception as e:
        ui_warning(f"Проверка микрофона: {e}")

    prompt_path = BASE_DIR / "system_prompt.txt"
    if not prompt_path.exists():
        ui_error(f"Файл {prompt_path} не найден и не удалось создать шаблон.")
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    lm_we_started_server = False
    lm_gui_before: set[int] = set()
    if config.lm_studio_auto_start:
        ok, lm_we_started_server, lm_gui_before = _ensure_lm_server(
            config.lm_studio_url, config.lm_studio_cli_path,
        )
        if not ok:
            ui_warning("LM Studio не отвечает. Запустите его вручную "
                       "или укажите lm_studio_cli_path в настройках.")

    searxng = SearxngManager(config)
    searxng_running = False
    if config.searxng_enabled and config.search_engine_priority != "ddg-only":
        searxng_running = searxng.ensure_running()

    core: JarvisCore | None = None

    try:
        core = JarvisCore(
            config, system_prompt,
            stt_engine=config.stt_engine,
            tts_engine=config.tts_engine,
            searxng=searxng,
            memory=None,
        )

        if lm_we_started_server:
            core.lm.mark_server_started_by_us(gui_pids_before=lm_gui_before)
        if config.lm_studio_auto_stop:
            core.lm.idle_stop_seconds = int(config.lm_studio_idle_stop_seconds)
        else:
            core.lm.idle_stop_seconds = 0

        models = core.lm.list_llm_models()
        chosen: str | None = None
        if config.default_model_key and any(
            m["key"] == config.default_model_key for m in models
        ):
            chosen = config.default_model_key
        elif models:
            chosen = models[0]["key"]

        if chosen:
            core.lm.current_model_key = chosen
            core.lm.default_model_key = chosen
            config.default_model_key = chosen
            ui_step("🧠", f"Модель LLM: {chosen}", "cyan")
        else:
            ui_warning("Нет доступных LLM-моделей в LM Studio.")

        if config.semantic_search_enabled and config.semantic_top_k > 0:
            emb_models = core.lm.list_embedding_models()
            default_emb = (
                config.embedding_model_key
                or (emb_models[0]["key"] if emb_models else "")
            )
            if default_emb:
                config.embedding_model_key = default_emb
                core.lm.embedding_model_key = default_emb
                if core.lm.check_embedding_available(default_emb):
                    history = HistoryStore(BASE_DIR / config.history_file)
                    embeddings = EmbeddingStore(BASE_DIR / config.embeddings_file)
                    history.load()
                    embeddings.load()
                    valid_ids = {e["id"] for e in history.all()}
                    embeddings.prune(valid_ids)
                    memory = SemanticMemory(
                        history=history,
                        embeddings=embeddings,
                        llm=core.lm,
                        embedding_model=default_emb,
                        instruction=config.embedding_instruction,
                    )
                    memory.ensure_embeddings()
                    core.memory = memory
                else:
                    ui_warning(f"Эмбеддер '{default_emb}' недоступен. "
                               f"Семантический поиск отключён.")

        try:
            config.save(BASE_DIR / "config.json")
        except Exception:
            pass

        from jarvis.web.desktop import run_desktop
        run_desktop(core, config)

    finally:
        if searxng_running or searxng.owned:
            try:
                searxng.stop()
            except Exception:
                pass

        if core is not None:
            try:
                for _ in range(150):
                    if not core.is_busy:
                        break
                    time.sleep(0.1)
                if core.is_busy:
                    LOGGER.write("LM",
                                 "shutdown: is_busy не снялся за 15 сек — "
                                 "продолжаю очистку")

                if config.lm_studio_auto_stop:
                    result = core.lm.shutdown_cleanup()
                    if result.get("server_stopped"):
                        ui_success("LM Studio server остановлен")
                    elif result.get("owned_by_us"):
                        ui_warning("Не удалось остановить LM Studio server")
                else:
                    LOGGER.write("LM",
                                 "lm_studio_auto_stop=False — только unload модели")
                    try:
                        core.lm.unload()
                    except Exception as ue:
                        LOGGER.write("LM", f"unload error: {ue}")
            except Exception as e:
                ui_warning(f"Ошибка при завершении LM Studio: {e}")


if __name__ == "__main__":
    main()