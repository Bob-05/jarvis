"""FastAPI-сервер: WebSocket-трансляция событий ядра + приём сообщений.

Архитектура:
  * JarvisCore работает в daemon-потоке (запускается в WebServer.run).
  * uvicorn работает в главном потоке (или в отдельном в desktop-режиме).
  * _event_pump читает события из core.event_queue и рассылает клиентам.
  * Текстовые запросы из UI идут через core.submit_text (неблокирующая
    очередь), а не через to_thread — чтобы не плодить потоки.
"""
import asyncio
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from jarvis.core.events import Event
from jarvis.core.jarvis_core import JarvisCore
from jarvis.ui.logger import get_logger

LOGGER = get_logger()

STATIC_DIR = Path(__file__).parent / "static"


class WebServer:
    def __init__(self, core: JarvisCore, host: str = "127.0.0.1", port: int = 8765):
        self.core = core
        self.host = host
        self.port = port
        self.app = FastAPI(title="J.A.R.V.I.S.", docs_url=None, redoc_url=None,
                           lifespan=self._create_lifespan())

        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._core_thread: threading.Thread | None = None

        # Подтверждение опасных команд
        self._confirm_event: threading.Event | None = None
        self._confirm_result: bool | None = None

        # Сигнал, что сервер готов (используется desktop-режимом)
        self.ready_event = threading.Event()

        self._setup_routes()

    # ============================================================
    # Lifespan
    # ============================================================
    def _create_lifespan(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._on_startup()
            yield
            await self._on_shutdown()
        return lifespan

    async def _on_startup(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.core.set_confirmation_callback(self._confirm_handler)
        asyncio.create_task(self._event_pump())
        self.ready_event.set()
        LOGGER.write("WEB", f"Сервер запущен на http://{self.host}:{self.port}")

    async def _on_shutdown(self) -> None:
        LOGGER.write("WEB", "Сервер останавливается")

    # ============================================================
    # Роуты
    # ============================================================
    def _setup_routes(self) -> None:
        @self.app.get("/")
        async def index():
            return FileResponse(STATIC_DIR / "index.html")

        @self.app.get("/api/status")
        async def status():
            return self._build_initial_state()

        @self.app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await self._handle_ws(ws)

        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
                           name="static")

    # ============================================================
    # WebSocket
    # ============================================================
    async def _handle_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._clients_lock:
            self._clients.add(ws)
        LOGGER.write("WEB", f"Клиент подключён (всего: {len(self._clients)})")

        try:
            await ws.send_json({
                "type": "initial_state",
                "data": self._build_initial_state(),
            })

            while True:
                msg = await ws.receive_json()
                await self._handle_client_message(msg)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            LOGGER.write("WEB", f"Ошибка WebSocket: {e}")
        finally:
            async with self._clients_lock:
                self._clients.discard(ws)
            LOGGER.write("WEB", f"Клиент отключён (осталось: {len(self._clients)})")

    async def _handle_client_message(self, msg: dict) -> None:
        t = msg.get("type")

        if t == "user_message":
            text = (msg.get("text") or "").strip()
            if not text:
                return
            # Неблокирующе кладём в очередь ядра
            self.core.submit_text(text)

        elif t == "confirmation_response":
            if self._confirm_event is not None:
                self._confirm_result = bool(msg.get("ok", False))
                self._confirm_event.set()

        elif t == "ping":
            pass

        else:
            LOGGER.write("WEB", f"Неизвестное сообщение: {t}")

    # ============================================================
    # Трансляция событий
    # ============================================================
    async def _event_pump(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                event: Event = await loop.run_in_executor(
                    None, lambda: self.core.event_queue.get(timeout=0.2)
                )
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.write("WEB", f"Ошибка event_pump: {e}")
                await asyncio.sleep(0.5)
                continue

            await self._broadcast({
                "type": event.type,
                "data": event.data,
            })

            if event.type == "shutdown":
                break

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        async with self._clients_lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._clients_lock:
                for ws in dead:
                    self._clients.discard(ws)

    # ============================================================
    # Подтверждение опасных команд
    # ============================================================
    def _confirm_handler(self) -> bool:
        if self._loop is None:
            return False

        self._confirm_event = threading.Event()
        self._confirm_result = None

        coro = self._broadcast({
            "type": "confirmation_request",
            "message": "⚠ Опасная команда. Подтвердите выполнение.",
        })
        asyncio.run_coroutine_threadsafe(coro, self._loop)

        if not self._confirm_event.wait(timeout=60):
            LOGGER.write("WEB", "Timeout подтверждения")
            return False
        return bool(self._confirm_result)

    # ============================================================
    # Начальное состояние
    # ============================================================
    def _build_initial_state(self) -> dict:
        mem = self.core.memory
        return {
            "models": {
                "llm": self.core.lm.current_model_key or "—",
                "embedding": self.core.lm.embedding_model_key or "—",
            },
            "memory": {
                "entries": len(mem.history.all()) if mem else 0,
                "top_k": self.core.config.semantic_top_k if mem else 0,
            },
            "stt": self.core.stt.name,
            "tts": self.core.tts.name,
            "voice": self.core.tts.speaker,
            "wake_word": self.core.wake_word if self.core.require_wake_word else None,
        }

    # ============================================================
    # Запуск
    # ============================================================
    def run(self) -> None:
        self._core_thread = threading.Thread(
            target=self.core.run, name="JarvisCore", daemon=True,
        )
        self._core_thread.start()

        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        finally:
            self.core.request_shutdown()
            if self._core_thread is not None:
                self._core_thread.join(timeout=5)
            LOGGER.write("WEB", "Сервер остановлен")