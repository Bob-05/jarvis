"""Системный трей для Jarvis (Windows)."""
from jarvis.ui.logger import get_logger

LOGGER = get_logger()

try:
    import pystray
    from PIL import Image, ImageDraw
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False


def _create_icon_image():
    """Генерирует иконку 64x64 — голографический круг с 'J'."""
    if not PYSTRAY_AVAILABLE:
        return None
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Внешний циан-круг
    draw.ellipse([2, 2, 62, 62], fill=(0, 229, 255, 255))
    # Тёмная внутренность
    draw.ellipse([10, 10, 54, 54], fill=(4, 9, 14, 255))
    # Внутреннее кольцо
    draw.ellipse([16, 16, 48, 48], outline=(0, 229, 255, 255), width=2)
    # Буква J (рисуем линиями, чтобы не зависеть от шрифтов)
    # Вертикальная линия
    draw.line([(32, 22), (32, 42)], fill=(0, 229, 255, 255), width=3)
    # Крючок снизу
    draw.arc([20, 32, 40, 48], start=0, end=180, fill=(0, 229, 255, 255), width=3)
    return img


def create_tray_icon(window, on_quit=None):
    """Создаёт pystray.Icon или None, если библиотека недоступна.

    window: pywebview-окно (должно уметь .show()/.hide())
    on_quit: callable, вызывается перед выходом из приложения
    """
    if not PYSTRAY_AVAILABLE:
        LOGGER.write("TRAY", "pystray недоступен, трей не создан")
        return None

    def show_window(icon, item):
        try:
            window.show()
        except Exception as e:
            LOGGER.write("TRAY", f"show error: {e}")

    def hide_window(icon, item):
        try:
            window.hide()
        except Exception as e:
            LOGGER.write("TRAY", f"hide error: {e}")

    def quit_app(icon, item):
        LOGGER.write("TRAY", "Выход через трей")
        try:
            icon.stop()
        except Exception:
            pass
        if on_quit:
            try:
                on_quit()
            except Exception as e:
                LOGGER.write("TRAY", f"on_quit error: {e}")

    menu = pystray.Menu(
        pystray.MenuItem("Показать", show_window, default=True),
        pystray.MenuItem("Скрыть", hide_window),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Выход", quit_app),
    )

    try:
        icon = pystray.Icon("jarvis", _create_icon_image(), "J.A.R.V.I.S.", menu)
        return icon
    except Exception as e:
        LOGGER.write("TRAY", f"create error: {e}")
        return None