"""Менеджер LM Studio: чат + эмбеддинги + загрузка/выгрузка моделей.

Управляет жизненным циклом LM Studio server:
  * запускает, если он не отвечает (через lms.exe)
  * останавливает при выходе Jarvis или после N секунд простоя
  * НЕ трогает сервер, если его запустил пользователь
  * НЕ останавливает сервер, пока есть активные запросы

Ключевые принципы (важно для shutdown):
  * self._lock НИКОГДА не удерживается во время subprocess/HTTP-вызовов.
  * Порядок очистки строго: unload модели → stop server → kill orphan → kill GUI.
  * Осиротевший llama-server.exe убивается только если сервер запускал Jarvis.
  * Процессы LM Studio.exe, появившиеся ПОСЛЕ старта сервера Jarvis'ом,
    тоже убиваются. Те, что были до, — не трогаем (снимок PID-ов).

Ограничения ресурсов (ResourceLimits):
  * gpu_ratio — доля слоёв на GPU (`--gpu 0.0..1.0|off|max`).
  * context_length — размер контекста (`--context-length N`).
  * process_priority — через PowerShell `PriorityClass` для llama-server.exe
    и LM Studio.exe. Приоритет применяется ПОСЛЕ каждой успешной загрузки
    модели (llama-server порождается LM Studio только при load).

  Если LM Studio не знает какого-то флага, load_model тихо откатывается
  на простой `lms load <model>`.
"""
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import requests
from openai import OpenAI

from jarvis import constants as C
from jarvis.config import ResourceLimits
from jarvis.ui.console import ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


# Хинты в имени модели, по которым отличаем эмбеддеры от LLM.
EMBEDDING_HINTS = (
    "embed", "bge", "e5-", "nomic-embed", "gte-", "minilm",
    "text-embedding", "m3-",
)


def _looks_like_embedding(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in EMBEDDING_HINTS)


def get_lm_studio_gui_pids() -> set[int]:
    """Возвращает PID-ы процессов 'LM Studio.exe' (Electron GUI / headless)."""
    if os.name != "nt":
        return set()
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq LM Studio.exe",
             "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        return set()

    pids: set[int] = set()
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("INFO:"):
            continue
        m = re.match(r'^"([^"]+)","(\d+)"', line)
        if m:
            try:
                pids.add(int(m.group(2)))
            except ValueError:
                pass
    return pids


class LMStudioManager:
    def __init__(self, base_url: str, default_model_key: str = "",
                 ttl_seconds: int = 180, embedding_model_key: str = "",
                 idle_stop_seconds: int = 0, cli_path: str = "",
                 resource_limits: ResourceLimits | None = None):
        self.root_url = base_url.rstrip("/")
        if self.root_url.endswith("/v1"):
            self.root_url = self.root_url[:-3]

        self.default_model_key = default_model_key or ""
        self.current_model_key = ""
        self.embedding_model_key = embedding_model_key or ""
        self.ttl = ttl_seconds
        self.idle_stop_seconds = int(idle_stop_seconds or 0)
        self.cli_path = (cli_path or "").strip()
        # ResourceLimits — настраивается пользователем при старте / из панели.
        self.resource_limits: ResourceLimits = resource_limits or ResourceLimits()

        self.instance_id: str | None = None
        self.last_used = time.time()
        self._lock = threading.RLock()

        # --- Флаги жизненного цикла сервера ---
        self._server_started_by_us: bool = False
        self._last_server_stop_log: float = 0.0

        # --- Снимок GUI-процессов до старта сервера ---
        self._gui_pids_before: set[int] = set()
        self._gui_snapshot_taken: bool = False

        # --- Счётчик активных запросов ---
        self._active_requests: int = 0

        self.client = OpenAI(base_url=f"{self.root_url}/v1", api_key="not-needed")
        self._api_prefix = self._detect_api_prefix()
        self._native_available = self._api_prefix is not None

    # ============================================================
    # Ограничения ресурсов
    # ============================================================
    def set_resource_limits(self, limits: ResourceLimits) -> None:
        """Обновляет ResourceLimits. Применится при следующей загрузке модели."""
        self.resource_limits = limits
        LOGGER.write("LM", f"Лимиты обновлены: {limits.describe()}")

    def _build_load_cli_args(self, model_key: str) -> list[str]:
        """Строит argv для `lms load`, включая флаги ресурсов.

        Если ограничения выключены — возвращает простой `load <model>`.
        Флаги пробуем «мягко»: если LM Studio их не знает, load_model
        откатится на простую команду.
        """
        args = ["load", model_key]
        rl = self.resource_limits
        if not rl.enabled:
            return args

        # --- GPU ---
        if rl.gpu_ratio <= 0.0:
            args += ["--gpu", "off"]
        elif rl.gpu_ratio >= 1.0:
            args += ["--gpu", "max"]
        else:
            args += ["--gpu", f"{rl.gpu_ratio:.2f}"]

        # --- Context length ---
        if rl.context_length > 0:
            args += ["--context-length", str(rl.context_length)]

        return args

    def _apply_process_priority(self, force: bool = False) -> None:
        """Устанавливает приоритет процессов LLM через PowerShell.

        Применяется к llama-server.exe (основной инференс) и LM Studio.exe
        (оболочка/сервер). Тихо выходит, если процессов нет или не Windows.

        force=True — применять приоритет даже когда он 'normal'. Нужен при
        смене настройки на лету: у уже работающего процесса приоритет надо
        вернуть в Normal явно. При загрузке модели force не требуется —
        свежий llama-server рождается с обычным приоритетом сам, поэтому
        режим 'normal' здесь пропускается (не тратим лишний PowerShell).
        """
        if os.name != "nt":
            return
        prio = self.resource_limits.effective_priority()
        if prio == "normal" and not force:
            return

        target_map = {
            "normal": "Normal",
            "below_normal": "BelowNormal",
            "low": "Idle",
        }
        target = target_map.get(prio)
        if not target:
            return

        script = (
            f"$ErrorActionPreference='SilentlyContinue';"
            f"Get-Process -Name 'llama-server','LM Studio' "
            f"-ErrorAction SilentlyContinue | "
            f"ForEach-Object {{ $_.PriorityClass = '{target}' }}"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if r.returncode == 0:
                LOGGER.write("LM",
                             f"Приоритет процессов LLM → {target}")
            else:
                err = (r.stderr or r.stdout or "").strip()
                LOGGER.write("LM",
                             f"Не удалось установить приоритет: {err[:150]}")
        except Exception as e:
            LOGGER.write("LM", f"_apply_process_priority error: {e}")

    # ============================================================
    # Native API
    # ============================================================
    def _detect_api_prefix(self):
        for prefix in ("/api/v0", "/api/v1"):
            try:
                r = requests.get(f"{self.root_url}{prefix}/models", timeout=4)
                if r.status_code == 200 and \
                        r.headers.get("content-type", "").startswith("application/json"):
                    return prefix
            except Exception:
                continue
        return None

    def _native_get(self, path: str):
        if not self._native_available:
            return None
        try:
            r = requests.get(f"{self.root_url}{self._api_prefix}{path}", timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _native_post(self, path: str, data: dict, timeout: int = 30):
        if not self._native_available:
            return None
        try:
            r = requests.post(f"{self.root_url}{self._api_prefix}{path}",
                              json=data, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # ============================================================
    # Проверка живости
    # ============================================================
    def _is_server_responding(self) -> bool:
        try:
            r = requests.get(f"{self.root_url}/v1/models", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    # ============================================================
    # CLI-хелперы
    # ============================================================
    def _find_lms_exe(self) -> str | None:
        if self.cli_path:
            p = Path(self.cli_path)
            try:
                if p.exists():
                    return str(p)
                LOGGER.write("LM",
                             f"cli_path указан, но не найден: {self.cli_path}")
            except Exception as e:
                LOGGER.write("LM", f"cli_path check error: {e}")
        for c in C.LM_STUDIO_CLI_CANDIDATES:
            try:
                if c.exists():
                    return str(c)
            except Exception:
                continue
        return None

    def _cli_run(self, args: list[str],
                 timeout: int = 60) -> subprocess.CompletedProcess | None:
        exe = self._find_lms_exe()
        if not exe:
            return None
        try:
            return subprocess.run(
                [exe] + args,
                capture_output=True, text=True, timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired:
            LOGGER.write("LM", f"lms {' '.join(args)}: таймаут {timeout} сек")
            return None
        except Exception as e:
            LOGGER.write("LM", f"lms {' '.join(args)} error: {e}")
            return None

    # ============================================================
    # Жизненный цикл сервера
    # ============================================================
    def mark_server_started_by_us(self,
                                  gui_pids_before: set[int] | None = None) -> None:
        """Помечает сервер как 'запущенный Jarvis'.

        Args:
            gui_pids_before: снимок PID-ов 'LM Studio.exe', сделанный
                ДО вызова `lms server start`. На shutdown будут убиты
                только НОВЫЕ процессы. Если None — сработает агрессивный
                режим (убьём все текущие LM Studio.exe).
        """
        self._server_started_by_us = True
        if gui_pids_before is not None:
            self._gui_pids_before = set(gui_pids_before)
            self._gui_snapshot_taken = True
            LOGGER.write("LM",
                         f"Снимок GUI: {len(self._gui_pids_before)} процессов "
                         f"до старта сервера")
        else:
            self._gui_snapshot_taken = False
            LOGGER.write("LM",
                         "mark_server_started_by_us: снимок GUI не передан — "
                         "на выходе будет использован агрессивный режим "
                         "(убьём все LM Studio.exe)")
        LOGGER.write("LM", "Сервер помечен как 'запущен Jarvis'")

    def is_server_owned_by_us(self) -> bool:
        return self._server_started_by_us

    def _ensure_server_running(self, timeout_sec: int = 30) -> bool:
        if self._is_server_responding():
            return True

        exe = self._find_lms_exe()
        if not exe:
            LOGGER.write("LM", "lms.exe не найден — не могу запустить сервер")
            return False

        ui_step("🚀", "Перезапуск LM Studio server...", "yellow")
        try:
            subprocess.Popen(
                [exe, "server", "start"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as e:
            LOGGER.write("LM", f"server start error: {e}")
            return False

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            time.sleep(2)
            if self._is_server_responding():
                ui_success("LM Studio server снова отвечает")
                # Приоритет можно применить уже сейчас — LM Studio.exe есть.
                self._apply_process_priority()
                return True
        LOGGER.write("LM", f"server не ответил за {timeout_sec} сек")
        return False

    def stop_server(self, force: bool = False) -> bool:
        if not force:
            with self._lock:
                if not self._server_started_by_us:
                    LOGGER.write("LM",
                                 "stop_server: сервер запущен не нами, пропуск")
                    return False
                if self._active_requests > 0:
                    LOGGER.write("LM",
                                 f"stop_server: {self._active_requests} активных "
                                 f"запросов, не останавливаю")
                    return False

        if not self._is_server_responding():
            LOGGER.write("LM", "stop_server: сервер уже не отвечает")
            self._reset_state_after_stop(success=True)
            return True

        success = self._stop_server_cli()
        self._reset_state_after_stop(success=success)
        return success

    def _stop_server_cli(self) -> bool:
        exe = self._find_lms_exe()
        if not exe:
            LOGGER.write("LM", "lms.exe не найден — не могу остановить сервер")
            return False

        try:
            r = subprocess.run(
                [exe, "server", "stop"],
                capture_output=True, text=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired:
            LOGGER.write("LM", "lms server stop: таймаут 15 сек")
            return False
        except Exception as e:
            LOGGER.write("LM", f"lms server stop error: {e}")
            return False

        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            LOGGER.write("LM",
                         f"lms server stop вернул {r.returncode}: {err[:200]}")
            return False

        for _ in range(5):
            if not self._is_server_responding():
                LOGGER.write("LM", "LM Studio server остановлен (CLI)")
                return True
            time.sleep(1)

        LOGGER.write("LM",
                     "lms server stop вернул 0, но сервер всё ещё отвечает")
        return False

    def _reset_state_after_stop(self, success: bool) -> None:
        if success:
            self._server_started_by_us = False
        self.instance_id = None
        self.current_model_key = ""

    def _kill_orphan_llama_server(self) -> bool:
        if os.name != "nt":
            return False
        try:
            r = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if not r.stdout or "llama-server.exe" not in r.stdout:
                return False

            r = subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if r.returncode == 0:
                LOGGER.write("LM", "llama-server.exe остановлен (taskkill)")
                return True
            err = (r.stderr or r.stdout or "").strip()
            LOGGER.write("LM",
                         f"taskkill llama-server.exe вернул {r.returncode}: "
                         f"{err[:200]}")
            return False
        except Exception as e:
            LOGGER.write("LM", f"_kill_orphan_llama_server error: {e}")
            return False

    def _kill_new_lm_studio_gui(self) -> int:
        if os.name != "nt":
            return 0

        current = get_lm_studio_gui_pids()
        if not current:
            return 0

        if self._gui_snapshot_taken:
            new_pids = current - self._gui_pids_before
            mode = "точный"
        else:
            new_pids = current
            mode = "агрессивный"

        if not new_pids:
            return 0

        LOGGER.write("LM",
                     f"[{mode}] Убиваю процессы LM Studio.exe: "
                     f"{sorted(new_pids)}")
        killed = 0
        for pid in sorted(new_pids):
            try:
                r = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if r.returncode == 0:
                    killed += 1
                else:
                    err = (r.stderr or r.stdout or "").strip()
                    # Код 128 = процесс уже завершился сам — не ошибка
                    if r.returncode != 128:
                        LOGGER.write("LM",
                                     f"taskkill PID {pid} код {r.returncode}: "
                                     f"{err[:150]}")
            except Exception as e:
                LOGGER.write("LM", f"taskkill PID {pid} error: {e}")
        if killed:
            LOGGER.write("LM",
                         f"Остановлено процессов LM Studio.exe: {killed}")
        return killed

    def maybe_stop_server_by_idle(self, idle_seconds: int | None = None) -> bool:
        if idle_seconds is None:
            idle_seconds = self.idle_stop_seconds
        if idle_seconds <= 0:
            return False
        if not self._server_started_by_us:
            return False
        if not self._is_server_responding():
            return False

        if self._active_requests > 0:
            return False

        idle = time.time() - self.last_used
        if idle < idle_seconds:
            return False

        if time.time() - self._last_server_stop_log < 30:
            return False
        self._last_server_stop_log = time.time()

        ui_step("💤",
                f"LM Studio простаивает {int(idle)} с → останавливаю сервер",
                "yellow")
        stopped = self.stop_server()
        if stopped:
            ui_success("LM Studio server остановлен (простой)")
        return stopped

    # ============================================================
    # Списки моделей
    # ============================================================
    def list_llm_models(self) -> list:
        data = self._native_get("/models")
        if data:
            models = []
            for m in data.get("models", []):
                key = m.get("key") or m.get("id") or ""
                if not key:
                    continue
                mtype = (m.get("type") or "").lower()
                if mtype in ("embeddings", "embedding"):
                    continue
                if mtype != "llm" and _looks_like_embedding(key):
                    continue
                models.append({
                    "key": key,
                    "display_name": m.get("display_name") or key,
                    "type": "llm",
                    "params_string": m.get("params_string") or "",
                })
            if models:
                return models

        try:
            resp = self.client.models.list()
            out = []
            for m in resp.data:
                key = getattr(m, "id", None) or str(m)
                if _looks_like_embedding(key):
                    continue
                out.append({
                    "key": key,
                    "display_name": key,
                    "type": "llm",
                    "params_string": "",
                })
            return out
        except Exception as e:
            ui_warning(f"Не удалось получить список LLM: {e}")
            return []

    def list_embedding_models(self) -> list:
        data = self._native_get("/models")
        if data:
            models = []
            for m in data.get("models", []):
                key = m.get("key") or m.get("id") or ""
                if not key:
                    continue
                mtype = (m.get("type") or "").lower()
                if mtype in ("embeddings", "embedding"):
                    models.append({
                        "key": key,
                        "display_name": m.get("display_name") or key,
                        "type": "embedding",
                        "params_string": m.get("params_string") or "",
                    })
            if models:
                return models

        try:
            resp = self.client.models.list()
            result = []
            for m in resp.data:
                key = getattr(m, "id", None) or str(m)
                if _looks_like_embedding(key):
                    result.append({
                        "key": key, "display_name": key,
                        "type": "embedding", "params_string": "",
                    })
            return result
        except Exception as e:
            ui_warning(f"Не удалось получить список эмбеддеров: {e}")
            return []

    # ============================================================
    # Загрузка / выгрузка
    # ============================================================
    def load_model(self, model_key: str) -> bool:
        with self._lock:
            if not model_key:
                return False

            if not self._is_server_responding():
                if self._server_started_by_us or self.idle_stop_seconds > 0:
                    if not self._ensure_server_running():
                        return False

            if self.instance_id and self.current_model_key == model_key:
                self.last_used = time.time()
                return True

            self._active_requests += 1
            try:
                if self._native_available:
                    if self.instance_id:
                        self._native_post("/models/unload",
                                          {"instance_id": self.instance_id})
                        self.instance_id = None

                    ui_step("🧠", f"Загрузка модели через API: {model_key}", "yellow")
                    resp = self._native_post("/models/load",
                                             {"model": model_key}, timeout=30)
                    if resp and "instance_id" in resp:
                        self.instance_id = resp["instance_id"]
                        self.current_model_key = model_key
                        self.last_used = time.time()
                        ui_success(f"Модель загружена · {model_key}")
                        self._apply_process_priority()
                        return True
                    else:
                        LOGGER.write("LM",
                                     "native load не ответил — отключаю native path")
                        self._native_available = False

                # --- CLI с флагами ресурсов (если ограничения включены) ---
                load_args = self._build_load_cli_args(model_key)
                if len(load_args) > 2:
                    LOGGER.write("LM",
                                 f"lms load с лимитами: "
                                 f"{self.resource_limits.describe()}")

                r = self._cli_run(load_args, timeout=180)

                # Откат на простой `load <model>`, если расширенный вызов
                # не сработал (старая LM Studio, неизвестный флаг, конфликт).
                if (r is None or r.returncode != 0) and len(load_args) > 2:
                    code = r.returncode if r is not None else "N/A"
                    err = ""
                    if r is not None:
                        err = (r.stderr or r.stdout or "").strip()[:200]
                    LOGGER.write("LM",
                                 f"lms load с флагами не удался (код {code}): "
                                 f"{err}. Пробую без флагов...")
                    r = self._cli_run(["load", model_key], timeout=180)

                if r is not None and r.returncode == 0:
                    self.instance_id = f"cli:{model_key}"
                    self.current_model_key = model_key
                    self.last_used = time.time()
                    ui_success(f"Модель загружена (CLI) · {model_key}")
                    # Приоритет применяем ПОСЛЕ загрузки: llama-server
                    # порождается LM Studio только в этот момент.
                    self._apply_process_priority()
                    return True

                ui_warning(
                    f"Не удалось загрузить '{model_key}' явно. "
                    f"Попробую отправить запрос — если включён JIT, "
                    f"LM Studio загрузит её сам."
                )
                self.current_model_key = model_key
                self.last_used = time.time()
                return True
            finally:
                self._active_requests -= 1

    def _resolve_model_key(self):
        if self.current_model_key:
            return self.current_model_key
        if self.default_model_key:
            models = self.list_llm_models()
            if any(m["key"] == self.default_model_key for m in models):
                return self.default_model_key
            ui_warning(f"Модель по умолчанию '{self.default_model_key}' не найдена.")
        models = self.list_llm_models()
        return models[0]["key"] if models else None

    def ensure_loaded(self) -> bool:
        with self._lock:
            if self.instance_id:
                self.last_used = time.time()
                return True
            key = self._resolve_model_key()
            if not key:
                return False
            return self.load_model(key)

    def unload_if_idle(self) -> None:
        with self._lock:
            instance_id = self.instance_id
            if not instance_id:
                return
            if self._active_requests > 0:
                return
            if time.time() - self.last_used < self.ttl:
                return
            model_key = self.current_model_key
            self.instance_id = None

        ui_step("💤", "Выгрузка модели (таймаут бездействия)...", "yellow")
        self._unload_internal(instance_id, model_key)

    def unload(self) -> None:
        with self._lock:
            instance_id = self.instance_id
            if not instance_id:
                return
            model_key = self.current_model_key
            self.instance_id = None

        self._unload_internal(instance_id, model_key)

    def _unload_internal(self, instance_id: str, model_key: str) -> bool:
        if not instance_id:
            return False

        is_cli = instance_id.startswith("cli:")

        if not is_cli and self._native_available:
            resp = self._native_post("/models/unload",
                                     {"instance_id": instance_id}, timeout=10)
            if resp is not None:
                LOGGER.write("LM", "Модель выгружена через native API")
                return True
            LOGGER.write("LM", "native unload не сработал, пробую CLI")

        if not model_key:
            LOGGER.write("LM", "model_key неизвестен — CLI unload пропущен")
            return False

        r = self._cli_run(["unload", model_key], timeout=15)
        if r is None:
            LOGGER.write("LM", "lms unload: команда не выполнена (lms.exe?)")
            return False
        if r.returncode == 0:
            LOGGER.write("LM", f"Модель выгружена через CLI: {model_key}")
            return True
        err = (r.stderr or r.stdout or "").strip()
        LOGGER.write("LM",
                     f"lms unload {model_key} вернул {r.returncode}: {err[:200]}")
        return False

    # ============================================================
    # Полная очистка при завершении
    # ============================================================
    def shutdown_cleanup(self) -> dict:
        result = {
            "model_unloaded": False,
            "server_stopped": False,
            "orphan_killed": False,
            "gui_killed": 0,
            "owned_by_us": False,
        }

        with self._lock:
            owned_by_us = self._server_started_by_us
            instance_id = self.instance_id
            model_key = self.current_model_key
            self.instance_id = None

        result["owned_by_us"] = owned_by_us

        if instance_id:
            ui_step("💤", "Выгрузка модели из памяти...", "yellow")
            result["model_unloaded"] = self._unload_internal(instance_id, model_key)
            if result["model_unloaded"]:
                ui_success("Модель выгружена из памяти")
            else:
                ui_warning("Не удалось выгрузить модель из памяти")

        if owned_by_us:
            if self._is_server_responding():
                ui_step("🛑", "Остановка LM Studio server...", "yellow")
                result["server_stopped"] = self._stop_server_cli()
            else:
                result["server_stopped"] = True

            if result["server_stopped"] or not self._is_server_responding():
                self._reset_state_after_stop(success=True)
            else:
                LOGGER.write("LM",
                             "shutdown_cleanup: сервер всё ещё отвечает")
        else:
            LOGGER.write("LM",
                         "shutdown_cleanup: сервер запущен не нами — не трогаем")

        if owned_by_us:
            result["orphan_killed"] = self._kill_orphan_llama_server()
            if result["orphan_killed"]:
                ui_success("Процесс llama-server.exe остановлен (VRAM освобождена)")

        if owned_by_us:
            killed = self._kill_new_lm_studio_gui()
            result["gui_killed"] = killed
            if killed:
                ui_success(f"Остановлено процессов LM Studio: {killed}")

        return result

    # ============================================================
    # Чат и эмбеддинги
    # ============================================================
    def chat(self, messages: list, temperature: float = 0.3,
             max_tokens: int = 200) -> str:
        with self._lock:
            self._active_requests += 1
            try:
                if not self.ensure_loaded():
                    raise RuntimeError("Не удалось загрузить локальную модель")
                resp = self.client.chat.completions.create(
                    model=self.current_model_key or "local-model",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.last_used = time.time()
                return resp.choices[0].message.content.strip()
            finally:
                self._active_requests -= 1

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.embedding_model_key
        if not model:
            raise RuntimeError("Не задана модель эмбеддингов")
        if not texts:
            return []
        with self._lock:
            self._active_requests += 1
            try:
                if not self._is_server_responding():
                    if self._server_started_by_us or self.idle_stop_seconds > 0:
                        self._ensure_server_running()
                resp = self.client.embeddings.create(model=model, input=texts)
                self.last_used = time.time()
                return [d.embedding for d in resp.data]
            finally:
                self._active_requests -= 1

    def check_embedding_available(self, model: str | None = None) -> bool:
        try:
            self.embed(["ping"], model=model)
            return True
        except Exception:
            return False