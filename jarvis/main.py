"""Точка входа: Desktop UI по умолчанию, консольные меню — по флагу --console."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from jarvis import constants as C
from jarvis.audio.recorder import check_microphone
from jarvis.config import AppConfig
from jarvis.core import JarvisCore
from jarvis.llm.lm_studio import get_lm_studio_gui_pids
from jarvis.memory import EmbeddingStore, HistoryStore, SemanticMemory
from jarvis.stt.kairos_engine import KAIROS_AVAILABLE
from jarvis.stt.vosk_engine import VOSK_AVAILABLE, validate_vosk_model
from jarvis.tools.searxng import SearxngManager
from jarvis.tts import PIPER_AVAILABLE, SILERO_AVAILABLE
from jarvis.ui.console import (
    ui_banner, ui_error, ui_info_table, ui_step, ui_success, ui_warning,
)
from jarvis.ui.logger import get_logger
from jarvis.ui.menu import (
    menu_ask_clear_logs,
    menu_clear_history,
    menu_select_docker_stop_mode,
    menu_select_embedding_model,
    menu_select_logging_mode,
    menu_select_model,
    menu_select_resource_limits,
    menu_select_search_engine,
    menu_select_semantic_top_k,
    menu_select_stt,
    menu_select_tts,
    menu_select_wake_mode,
)
from jarvis.utils import build_datetime_context

LOGGER = get_logger()


def parse_args():
    p = argparse.ArgumentParser(
        prog="jarvis",
        description="J.A.R.V.I.S. — голосовой ассистент",
    )
    p.add_argument("--console", action="store_true",
                   help="Запустить консольный режим с интерактивными меню "
                        "(по умолчанию — Desktop UI)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Адрес для Web UI (по умолчанию 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765,
                   help="Порт для Web UI (по умолчанию 8765)")
    args = p.parse_args()
    args.desktop = not args.console
    return args


# ============================================================
# LM Studio: запуск сервера
# ============================================================
def ensure_lm_studio_server(url: str, cli_path: str = "") -> tuple[bool, bool, set[int]]:
    """Проверяет, что LM Studio отвечает. Если нет — запускает через lms.exe.

    Возвращает (success, we_started_it, gui_pids_before):
      * success         — True, если сервер отвечает
      * we_started_it   — True, если Jarvis сам запустил сервер
      * gui_pids_before — снимок PID-ов 'LM Studio.exe' ДО старта сервера.
                          На shutdown убьём только НОВЫЕ процессы.
    """
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=3)
        if r.status_code == 200:
            LOGGER.write("LM", "LM Studio уже запущен (не нами)")
            return True, False, set()
    except requests.RequestException:
        pass

    # --- Снимок GUI ДО старта сервера ---
    gui_pids_before = get_lm_studio_gui_pids()
    LOGGER.write("LM",
                 f"Снимок GUI до старта сервера: "
                 f"{len(gui_pids_before)} процессов")

    lms_exe = None
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            lms_exe = str(p)
        else:
            ui_warning(f"Указанный путь к lms.exe не существует: {cli_path}")
    else:
        for candidate in C.LM_STUDIO_CLI_CANDIDATES:
            if candidate.exists():
                lms_exe = str(candidate)
                break

    if not lms_exe:
        ui_error("Не найден lms.exe. Укажите путь в config.json → lm_studio_cli_path.")
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
                ui_success("LM Studio server запущен и отвечает (запущен Jarvis'ом)")
                LOGGER.write("LM", "LM Studio server запущен Jarvis'ом")
                return True, True, gui_pids_before
        except requests.RequestException:
            continue

    ui_error("LM Studio server не ответил за 30 секунд.")
    return False, False, set()


# ============================================================
# Основной сценарий
# ============================================================
def main():
    args = parse_args()
    config = AppConfig.load(C.CONFIG_FILE)

    if args.desktop:
        _run_autonomous_desktop(config, args)
        return

    _run_console_with_menus(config, args)


# ============================================================
# Console с меню
# ============================================================
def _run_console_with_menus(config: AppConfig, args) -> None:
    if config.ask_logging_mode_on_start:
        config.verbose_logging = menu_select_logging_mode(config)
        LOGGER.configure(
            enabled=config.verbose_logging,
            path=Path(config.log_file),
        )
        if config.verbose_logging:
            if menu_ask_clear_logs():
                if LOGGER.clear():
                    ui_success("Файл логов очищен.")
                else:
                    ui_warning("Не удалось очистить файл логов.")
            ui_success(f"Подробное логирование включено → {LOGGER.path}")
    else:
        LOGGER.configure(enabled=config.verbose_logging, path=Path(config.log_file))

    LOGGER.write("START", "=" * 60)
    LOGGER.write("START", "Запуск Джарвиса (UI: console)")

    check_microphone()

    if config.ask_wake_mode_on_start:
        config.require_wake_word = menu_select_wake_mode(config)

    if config.ask_search_engine_on_start:
        config.search_engine_priority = menu_select_search_engine(config)

    if config.ask_resource_limits_on_start:
        limits = menu_select_resource_limits(config)
        config.llm_limits_enabled = bool(limits.get("enabled", False))
        config.llm_gpu_ratio = float(limits.get("gpu_ratio", 1.0))
        config.llm_context_length = int(limits.get("context_length", 0))
        config.llm_process_priority = str(limits.get("priority", "normal"))
        ui_success(f"Ресурсы LLM: {config.get_resource_limits().describe()}")

    if (config.searxng_enabled
            and config.search_engine_priority != "ddg-only"
            and config.ask_docker_stop_mode_on_start):
        config.searxng_stop_mode = menu_select_docker_stop_mode(config)

    if config.ask_tts_on_start:
        config.tts_engine = menu_select_tts(config, SILERO_AVAILABLE, PIPER_AVAILABLE)

    if config.ask_stt_on_start:
        config.stt_engine = menu_select_stt(
            config, KAIROS_AVAILABLE, VOSK_AVAILABLE, validate_vosk_model,
        )

    if config.ask_semantic_on_start:
        config.semantic_top_k = menu_select_semantic_top_k(config)
        config.semantic_search_enabled = config.semantic_top_k > 0

    history_path = Path(config.history_file)
    embeddings_path = Path(config.embeddings_file)
    rebuild_requested = False

    if config.ask_clear_history_on_start:
        choice = menu_clear_history()
        if choice == "dialogs":
            if history_path.exists():
                history_path.unlink()
            if embeddings_path.exists():
                embeddings_path.unlink()
            ui_success("История диалогов очищена.")
        elif choice == "all":
            if history_path.exists():
                history_path.unlink()
            if embeddings_path.exists():
                embeddings_path.unlink()
            ui_success("История и эмбеддинги очищены.")
        elif choice == "rebuild":
            rebuild_requested = True
            ui_success("Эмбеддинги будут пересчитаны при старте.")

    ui_banner()

    _setup_core_and_run(config, args, rebuild_requested, autonomous=False)


# ============================================================
# Автономный Desktop
# ============================================================
def _run_autonomous_desktop(config: AppConfig, args) -> None:
    print("[Jarvis] Режим Desktop. Все настройки — в панели ⚙ НАСТРОЙКИ.")
    print(f"[Jarvis] Конфиг: {Path(C.CONFIG_FILE).resolve()}")
    print()

    LOGGER.configure(enabled=config.verbose_logging, path=Path(config.log_file))
    LOGGER.write("START", "=" * 60)
    LOGGER.write("START", "Запуск Джарвиса (UI: desktop)")

    try:
        check_microphone()
    except Exception as e:
        ui_warning(f"Проверка микрофона: {e}")

    _setup_core_and_run(config, args, rebuild_requested=False, autonomous=True)


# ============================================================
# Общая часть
# ============================================================
def _setup_core_and_run(config: AppConfig, args, rebuild_requested: bool,
                        autonomous: bool = False) -> None:
    try:
        system_prompt = Path(C.SYSTEM_PROMPT_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        ui_error(f"Файл {C.SYSTEM_PROMPT_FILE} не найден")
        sys.exit(1)

    lm_we_started_server = False
    lm_gui_before: set[int] = set()

    if config.lm_studio_auto_start:
        ok, we_started, gui_before = ensure_lm_studio_server(
            config.lm_studio_url, config.lm_studio_cli_path,
        )
        lm_we_started_server = we_started
        lm_gui_before = gui_before
        if not ok:
            ui_warning(
                "LM Studio не отвечает. Запустите его вручную или проверьте "
                "путь к lms.exe в config.json → lm_studio_cli_path."
            )
    else:
        try:
            requests.get(f"{config.lm_studio_url.rstrip('/')}/v1/models", timeout=4)
        except requests.ConnectionError:
            ui_error("LM Studio не запущена! Включите 'lm_studio_auto_start' "
                     "в config.json.")
        except Exception:
            pass

    searxng = SearxngManager(config)
    searxng_running = False
    core: JarvisCore | None = None

    try:
        search_priority = config.search_engine_priority

        if searxng.enabled and search_priority != "ddg-only":
            ui_step("🌐", "Подготовка SearXNG...", "cyan")
            searxng_running = searxng.ensure_running()
            if not searxng_running:
                ui_warning("SearXNG недоступен, поиск пойдёт через DuckDuckGo.")
        elif search_priority == "ddg-only":
            ui_step("🔎", "Поисковик: только DuckDuckGo (SearXNG пропущен).", "cyan")

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

        if autonomous:
            chosen: str | None = None
            if config.default_model_key:
                if any(m["key"] == config.default_model_key for m in models):
                    chosen = config.default_model_key
                else:
                    ui_warning(
                        f"default_model_key '{config.default_model_key}' "
                        f"не найдена среди LLM-моделей"
                    )
            if chosen is None and models:
                chosen = models[0]["key"]

            if chosen:
                core.lm.current_model_key = chosen
                core.lm.default_model_key = chosen
                config.default_model_key = chosen
                ui_step("🧠", f"Модель LLM: {chosen}", "cyan")
            else:
                ui_warning("Нет доступных LLM-моделей в LM Studio. "
                           "Загрузите модель в LM Studio → My Models.")
        else:
            if models:
                default_key = config.default_model_key or core.lm.default_model_key
                chosen = menu_select_model(
                    models,
                    current_key=core.lm.current_model_key or default_key,
                    default_key=default_key,
                )
                if chosen:
                    core.lm.current_model_key = chosen
                    core.lm.default_model_key = chosen
                    config.default_model_key = chosen
                    ui_success(f"Модель выбрана: {chosen}")
                elif default_key:
                    core.lm.current_model_key = default_key
                    ui_success(f"Оставлена модель по умолчанию: {default_key}")
            else:
                ui_warning("Нет доступных LLM-моделей в LM Studio.")

        if core.lm.current_model_key:
            ui_step("⏳", f"Загрузка модели '{core.lm.current_model_key}'...", "yellow")
            ok = core.lm.load_model(core.lm.current_model_key)
            if ok:
                ui_success(f"Модель в памяти: {core.lm.current_model_key}")
            else:
                ui_warning(
                    f"Не удалось загрузить '{core.lm.current_model_key}'. "
                    f"Откройте LM Studio → Developer → My Models и загрузите модель."
                )

        history_path = Path(config.history_file)
        embeddings_path = Path(config.embeddings_file)
        memory: SemanticMemory | None = None

        if config.semantic_search_enabled and config.semantic_top_k > 0:
            ui_step("🧠", "Подготовка семантической памяти...", "cyan")
            emb_models = core.lm.list_embedding_models()
            default_emb = ""
            if config.embedding_model_key:
                default_emb = config.embedding_model_key
            elif emb_models:
                default_emb = emb_models[0]["key"]

            chosen_emb: str | None = None

            if autonomous:
                chosen_emb = config.embedding_model_key or default_emb
                if chosen_emb:
                    ui_step("🧠", f"Эмбеддер (из config): {chosen_emb}", "cyan")
            else:
                if config.ask_embedding_model_on_start:
                    if not emb_models:
                        ui_warning("В LM Studio не найдено embedding-моделей. "
                                   "Можно ввести имя модели вручную.")
                    chosen_emb = menu_select_embedding_model(
                        emb_models,
                        current_key=config.embedding_model_key,
                        default_key=default_emb,
                    )
                    if chosen_emb is None and default_emb:
                        chosen_emb = default_emb
                        ui_step("↩️", f"Оставлена модель по умолчанию: "
                                      f"{default_emb}", "dim")
                else:
                    chosen_emb = default_emb

            if not chosen_emb:
                ui_warning("Embedding-модель не выбрана. "
                           "Семантический поиск отключён.")
                config.semantic_top_k = 0
                config.semantic_search_enabled = False
            else:
                low = chosen_emb.lower()
                if not any(h in low for h in C.EMBEDDING_DEFAULT_MODEL_HINTS):
                    ui_warning(
                        f"Модель '{chosen_emb}' не похожа на специализированную "
                        f"embedding-модель. Качество поиска может быть ниже."
                    )
                config.embedding_model_key = chosen_emb
                core.lm.embedding_model_key = chosen_emb

                ui_step("⏳", f"Проверка эмбеддера '{chosen_emb}'...", "yellow")
                if not core.lm.check_embedding_available(chosen_emb):
                    ui_warning(f"Эмбеддер '{chosen_emb}' не отвечает. "
                               "Семантический поиск отключён.")
                    config.semantic_top_k = 0
                    config.semantic_search_enabled = False
                else:
                    ui_success(f"Эмбеддер готов: {chosen_emb}")
                    try:
                        sample = core.lm.embed(["test"], model=chosen_emb)
                        if sample and sample[0]:
                            ui_step("📏", f"Размерность вектора: "
                                          f"{len(sample[0])}", "dim")
                    except Exception:
                        pass

                    history = HistoryStore(history_path)
                    embeddings = EmbeddingStore(embeddings_path)
                    history.load()
                    embeddings.load()
                    valid_ids = {e["id"] for e in history.all()}
                    pruned = embeddings.prune(valid_ids)
                    if pruned:
                        ui_step("🧹", f"Удалено устаревших векторов: "
                                      f"{pruned}", "dim")

                    memory = SemanticMemory(
                        history=history,
                        embeddings=embeddings,
                        llm=core.lm,
                        embedding_model=chosen_emb,
                        instruction=config.embedding_instruction,
                    )
                    if rebuild_requested:
                        memory.rebuild()
                    else:
                        memory.ensure_embeddings()
                    core.memory = memory

        search_human = {
            "searxng-first": "SearXNG → DuckDuckGo" if searxng_running
                             else "DuckDuckGo (SearXNG недоступен)",
            "ddg-first": "DuckDuckGo → SearXNG" if searxng_running else "DuckDuckGo",
            "ddg-only": "только DuckDuckGo",
        }.get(search_priority, search_priority)

        ui_mode = "Desktop" if args.desktop else "Console"
        lm_owner = ("Jarvis (остановит при выходе)"
                    if core.lm.is_server_owned_by_us()
                    else "пользователь (Jarvis не тронет)")
        lm_idle = (f"{config.lm_studio_idle_stop_seconds} сек"
                   if config.lm_studio_auto_stop
                   else "отключён")
        info = {
            "UI": ui_mode,
            "Режим": "автономный" if autonomous else "интерактивный",
            "Модель LM Studio": config.lm_studio_url,
            "LM Studio server": lm_owner,
            "Стоп при простое": lm_idle,
            "Режим LM Studio": "native API" if core.lm._native_available else "JIT",
            "Модель LLM": core.lm.current_model_key or "по умолчанию",
            "Ресурсы LLM": core.lm.resource_limits.describe(),
            "Модель эмбеддингов": config.embedding_model_key or "—",
            "Семантический top-K": config.semantic_top_k
                                   if config.semantic_search_enabled else "отключён",
            "Движок STT": config.stt_engine,
            "Движок TTS": config.tts_engine,
            "Голос TTS": core.tts.speaker,
            "Поиск": search_human,
            "Логирование": f"подробно → {LOGGER.path}" if LOGGER.enabled else "кратко",
        }
        ui_info_table(info)
        LOGGER.write("START", f"Дата/время: {build_datetime_context()}")

        try:
            config.save(C.CONFIG_FILE)
        except Exception as e:
            ui_warning(f"Не удалось сохранить config.json: {e}")

        if args.desktop:
            from jarvis.web.desktop import run_desktop
            run_desktop(core, config, args.host, args.port)
        else:
            from jarvis.ui.console_ui import ConsoleUI
            ConsoleUI(core).run()

    finally:
        if searxng_running or searxng.owned:
            try:
                searxng.stop()
            except Exception as e:
                ui_warning(f"Ошибка при остановке SearXNG: {e}")

        if core is not None:
            try:
                # Ждём завершения запроса до 15 сек
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

        LOGGER.write("STOP", "Завершение работы Джарвиса")
        LOGGER.write("STOP", "=" * 60)


if __name__ == "__main__":
    main()