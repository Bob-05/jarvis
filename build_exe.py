"""Сборка Jarvis в один .exe через PyInstaller.

Использование:
    python build_exe.py

Результат: dist/Jarvis.exe

Файлы config.json и system_prompt.txt встраиваются в exe как шаблоны
(папка jarvis_data внутри бандла). При первом запуске они будут
распакованы рядом с .exe, после чего их можно править через UI.
"""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.resolve()
    static_dir = root / "jarvis" / "web" / "static"
    config_src = root / "config.json"
    prompt_src = root / "system_prompt.txt"

    if not static_dir.exists():
        print(f"ОШИБКА: не найдена папка {static_dir}")
        sys.exit(1)
    if not config_src.exists():
        print(f"ОШИБКА: не найден {config_src} (нужен как шаблон)")
        sys.exit(1)
    if not prompt_src.exists():
        print(f"ОШИБКА: не найден {prompt_src} (нужен как шаблон)")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "Jarvis",

        # --- Встроенные ресурсы ---
        "--add-data", f"{static_dir};jarvis/web/static",
        "--add-data", f"{config_src};jarvis_data",
        "--add-data", f"{prompt_src};jarvis_data",

        # --- Обязательные для pywebview / uvicorn / fastapi ---
        "--hidden-import", "clr",
        "--hidden-import", "webview",
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "websockets",
        "--hidden-import", "fastapi",
        "--hidden-import", "starlette",
        "--hidden-import", "pydantic",

        # --- Аудио / системы ---
        "--hidden-import", "sounddevice",
        "--hidden-import", "numpy",

        # --- Прочее (нужно, если используются эти фичи) ---
        "--hidden-import", "pystray",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "PIL.ImageDraw",
        "--hidden-import", "requests",
        "--hidden-import", "openai",
        "--hidden-import", "ddgs",
        "--hidden-import", "trafilatura",
        "--hidden-import", "bs4",
        "--hidden-import", "rich",

        # --- Точка входа ---
        str(root / "jarvis_exe.py"),
    ]

    # --- Опционально: если нужен полностью автономный exe с torch
    # (STT Kairos / TTS Silero), раскомментируйте:
    # cmd += ["--collect-all", "torch", "--collect-all", "torchaudio"]

    print("Сборка Jarvis.exe...")
    print("Команда:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print("\nСБОРКА НЕ УДАЛАСЬ")
        sys.exit(result.returncode)

    exe_path = root / "dist" / "Jarvis.exe"
    print()
    print("=" * 60)
    print(f"ГОТОВО: {exe_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()