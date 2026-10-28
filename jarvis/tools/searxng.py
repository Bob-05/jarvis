"""Управление SearXNG: Docker Desktop + контейнер."""
import os
import subprocess
import time
from pathlib import Path

import requests

from jarvis import constants as C
from jarvis.config import AppConfig
from jarvis.ui.console import ui_error, ui_status, ui_step, ui_success, ui_warning
from jarvis.ui.logger import get_logger

LOGGER = get_logger()


class SearxngManager:
    def __init__(self, config: AppConfig):
        self.enabled = bool(config.searxng_enabled)
        self.auto_start = bool(config.searxng_auto_start)
        self.auto_stop = bool(config.searxng_auto_stop)
        self.stop_mode = config.searxng_stop_mode
        self.url = config.searxng_url.rstrip("/")
        self.container_name = config.searxng_container_name
        self.compose_path = Path(config.searxng_compose_path).resolve()
        self.startup_timeout = int(config.searxng_startup_timeout)
        self.owned = False
        self._stopped = False

    @staticmethod
    def _run(cmd, timeout=15):
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None

    def _docker_daemon_ok(self) -> bool:
        r = self._run(["docker", "info"], timeout=10)
        return bool(r and r.returncode == 0)

    def _docker_desktop_running(self) -> bool:
        if os.name != "nt":
            return False
        r = self._run(
            ["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe", "/NH"],
            timeout=8,
        )
        return bool(r and "Docker Desktop.exe" in r.stdout)

    def _find_docker_desktop(self) -> str | None:
        for c in C.DOCKER_DESKTOP_EXE_CANDIDATES:
            try:
                if c.exists():
                    return str(c)
            except Exception:
                continue

        r = self._run(["where", "docker"], timeout=8)
        if r and r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    p = Path(line)
                    for levels in (1, 2, 3, 4):
                        q = p
                        for _ in range(levels):
                            q = q.parent
                        for name in ("Docker Desktop.exe", "Docker Desktop"):
                            cand = q / name
                            if cand.exists():
                                return str(cand)
                except Exception:
                    continue

        r = self._run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-AppxPackage *DockerDesktop*).InstallLocation"],
            timeout=12,
        )
        if r and r.returncode == 0 and r.stdout:
            for line in r.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    loc = Path(line)
                    for sub in ("Docker Desktop.exe", "app/Docker Desktop.exe",
                                "Docker/Docker Desktop.exe"):
                        cand = loc / sub
                        if cand.exists():
                            return str(cand)
                except Exception:
                    continue
        return None

    def _start_docker_desktop(self) -> bool:
        exe = self._find_docker_desktop()
        if exe:
            try:
                subprocess.Popen(
                    [exe],
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                LOGGER.write("DOCKER", f"Запущен Docker Desktop: {exe}")
                return True
            except Exception as e:
                LOGGER.write("DOCKER", f"Не удалось запустить по пути: {e}")

        if os.name == "nt":
            try:
                r = self._run(
                    ["powershell", "-NoProfile", "-Command",
                     "Start-Process 'shell:AppsFolder\\Docker.DockerDesktop_*!DockerDesktop'"],
                    timeout=15,
                )
                if r is not None and r.returncode == 0:
                    LOGGER.write("DOCKER", "Docker Desktop запущен через shell:AppsFolder")
                    return True
            except Exception as e:
                LOGGER.write("DOCKER", f"Store-запуск не удался: {e}")
        return False

    def _container_exists(self) -> bool:
        r = self._run(
            ["docker", "ps", "-a", "--filter",
             f"name=^{self.container_name}$", "--format", "{{.Names}}"],
            timeout=10,
        )
        return bool(r and self.container_name in (r.stdout or ""))

    def _count_running_containers(self) -> int:
        r = self._run(["docker", "ps", "-q"], timeout=10)
        if not r or r.returncode != 0:
            return 0
        return len([l for l in (r.stdout or "").splitlines() if l.strip()])

    def _http_ok(self) -> bool:
        try:
            r = requests.get(self.url + "/", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _wait_ready(self) -> bool:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._http_ok():
                return True
            time.sleep(1)
        return False

    def ensure_running(self) -> bool:
        if not self.enabled:
            return False

        if self._http_ok():
            ui_status("SearXNG уже запущен (не мной) — использую как есть.", "dim")
            self.owned = False
            return True

        if not self.auto_start:
            ui_warning("SearXNG не отвечает, автозапуск отключён.")
            return False

        if not self._docker_daemon_ok():
            ui_step("🐳", "Docker не отвечает, пробую запустить Docker Desktop...", "yellow")
            if not self._docker_desktop_running():
                if not self._start_docker_desktop():
                    ui_error("Не удалось найти Docker Desktop.exe.")
                    return False
            for _ in range(60):
                if self._docker_daemon_ok():
                    break
                time.sleep(2)
            else:
                ui_error("Docker daemon не запустился за 120 секунд.")
                return False
            ui_success("Docker daemon готов.")

        if self._container_exists():
            ui_step("🐳", f"Перезапуск контейнера '{self.container_name}'...", "yellow")
            r = self._run(["docker", "restart", self.container_name], timeout=60)
            if not r or r.returncode != 0:
                err = getattr(r, "stderr", "") or ""
                ui_error(f"docker restart не удался: {err.strip()[:200]}")
                return False
        else:
            compose_file = self.compose_path / "docker-compose.yml"
            if not compose_file.exists():
                ui_error(f"Не найден {compose_file}")
                return False
            ui_step("🐳", "Первый запуск SearXNG через docker compose...", "yellow")
            r = self._run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                timeout=300,
            )
            if not r or r.returncode != 0:
                err = getattr(r, "stderr", "") or ""
                ui_error(f"docker compose up не удался: {err.strip()[:300]}")
                return False

        self.owned = True
        ui_step("⏳", f"Ожидание ответа SearXNG ({self.startup_timeout} сек)...", "yellow")
        if not self._wait_ready():
            ui_error(f"SearXNG не ответил за {self.startup_timeout} сек.")
            return False
        ui_success("SearXNG готов к работе.")
        return True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if not self.enabled or not self.auto_stop:
            return

        if self.stop_mode == "container":
            if not self.owned:
                ui_status("SearXNG был запущен не мной — оставляю контейнер как есть.", "dim")
                return
            ui_step("🛑", f"Остановка контейнера '{self.container_name}'...", "yellow")
            r = self._run(["docker", "stop", self.container_name], timeout=30)
            if r and r.returncode == 0:
                ui_success("Контейнер остановлен.")
            else:
                ui_warning("Не удалось остановить контейнер SearXNG.")

        elif self.stop_mode == "docker":
            running = self._count_running_containers()
            if running > 1:
                ui_warning(f"Обнаружено {running} запущенных контейнеров — все будут остановлены.")
            ui_step("🛑", "Остановка Docker Desktop...", "yellow")

            graceful = False
            for cli in C.DOCKER_CLI_CANDIDATES:
                if cli.exists():
                    r = self._run([str(cli), "-Shutdown"], timeout=45)
                    if r is not None:
                        graceful = True
                        break
            if not graceful:
                self._run(["taskkill", "/IM", "Docker Desktop.exe", "/F"], timeout=15)
            for _ in range(20):
                if not self._docker_daemon_ok():
                    break
                time.sleep(1)
            ui_success("Docker Desktop остановлен.")