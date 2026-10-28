"""Все константы проекта Jarvis."""
import os
from pathlib import Path

# ---------- Аудио / VAD ----------
VOSK_RATE = 16000
CHUNK_SEC = 0.03
VAD_SILENCE_SEC = 1.0
MAX_RECORD_SEC = 15
MIN_RECORD_SEC = 0.5

VAD_INITIAL_THRESHOLD = 300
VAD_NOISE_FLOOR_RATIO = 2.0
VAD_UPDATE_RATE = 0.05

# ---------- TTS ----------
TTS_SAMPLE_RATE = 48000
POST_TTS_COOLDOWN_SEC = 0.4

# ---------- Файлы по умолчанию ----------
CONFIG_FILE = "config.json"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
LOG_FILE_DEFAULT = "jarvis_log.txt"
HISTORY_FILE_DEFAULT = "dialog_history.json"
EMBEDDINGS_FILE_DEFAULT = "history_embeddings.json"

# ---------- Пути к исполняемым файлам ----------
LM_STUDIO_CLI_CANDIDATES = [
    Path.home() / ".lmstudio" / "bin" / "lms.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "lms.exe",
    Path("C:/Program Files/LM Studio/lms.exe"),
]

DOCKER_DESKTOP_EXE_CANDIDATES = [
    Path("C:/Program Files/Docker/Docker/Docker Desktop.exe"),
    Path("C:/Program Files (x86)/Docker/Docker/Docker Desktop.exe"),
    Path(os.environ.get("PROGRAMFILES", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "Docker Desktop.exe",
]

DOCKER_CLI_CANDIDATES = [
    Path("C:/Program Files/Docker/Docker/DockerCli.exe"),
]

# ---------- Безопасность команд ----------
DANGEROUS_KEYWORDS = (
    "del ", "del/", "erase ", "rmdir", "rd /s", "rd /q",
    "format ", "diskpart", "chkdsk /f", "chkdsk /r",
    "shutdown", "restart", "reg delete", "reg add",
    "takeown", "icacls", "cacls", "taskkill", "tskill",
    "schtasks", "xcopy /y", "robocopy /purge",
    "invoke-expression", "iex ", "remove-item",
    "netsh ", "net user", "net localgroup",
    "curl ", "wget ", "certutil -urlcache",
)

SAFE_KEYWORDS = (
    "start notepad", "start calc", "start mspaint",
    "start explorer", "start cmd", "start powershell",
    "mkdir ", "start www.", "start msedge",
    "start chrome", "start firefox", "appopener open ",
)

URL_PREFIXES = ("start https://", "start http://")
INJECTION_CHARS = ("&", "|", ";", ">", "<", "`")
NO_COMMAND_VALUES = {"none", "null", "нет", "no", "-", "—", "n/a", "пусто"}

# ---------- Семантический поиск ----------
SEMANTIC_TOP_K_CHOICES = [0, 3, 5, 8, 10, 15]
SEMANTIC_TOP_K_DEFAULT = 5
SEMANTIC_MIN_PROMPT_LEN = 4
SEMANTIC_SKIP_UTTERANCES = {
    "да", "нет", "ок", "окей", "хорошо", "ясно", "ага", "угу", "стоп",
    "yes", "no", "ok", "stop", "cancel",
}
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEFAULT_MODEL_HINTS = ("embed", "bge", "e5", "nomic", "gte", "minilm", "giga")

# ---------- Инструкция для instruct-эмбеддеров ----------
EMBEDDING_DEFAULT_INSTRUCTION = "Given a query, retrieve relevant passages"

# ---------- Контекст истории ----------
# Сколько последних записей гарантированно подмешивать в запрос к LLM
# (в дополнение к семантическому поиску).
RECENT_HISTORY_COUNT = 10