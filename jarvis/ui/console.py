"""Rich-консольные примитивы и хелперы вывода."""
from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jarvis.ui.logger import get_logger

console = Console()
LOGGER = get_logger()


def ui_banner() -> None:
    console.clear()
    text = Text()
    text.append("ДЖАРВИС", style="bold bright_cyan")
    text.append("\n")
    text.append("Just A Rather Very Intelligent System", style="italic dim cyan")
    console.print(Panel(
        Align.center(text),
        border_style="bright_cyan", box=DOUBLE, padding=(1, 6),
        title="[bold bright_cyan]● СИСТЕМА АКТИВНА[/bold bright_cyan]",
        subtitle="[dim]локальный голосовой ассистент · Windows 11[/dim]",
    ))
    console.print()


def ui_step(icon: str, text: str, color: str = "cyan") -> None:
    console.print(f"  [{color}]{icon}[/{color}] {text}")
    LOGGER.write("STEP", f"{icon} {text}")


def ui_status(text: str, color: str = "green") -> None:
    console.print(f"[{color}]●[/{color}] [dim italic]{text}[/dim italic]")
    LOGGER.write("STATUS", text)


def ui_user(text: str) -> None:
    console.print(Panel(
        Text(text, style="bold white"),
        title="[bold bright_blue]🎤  ВЫ[/bold bright_blue]",
        border_style="bright_blue", box=ROUNDED, padding=(0, 2),
    ))
    LOGGER.write("USER", text)


def ui_jarvis(text: str) -> None:
    console.print(Panel(
        Text(text, style="bold bright_cyan"),
        title="[bold bright_cyan]🤖  ДЖАРВИС[/bold bright_cyan]",
        border_style="bright_cyan", box=ROUNDED, padding=(0, 2),
    ))
    LOGGER.write("JARVIS", text)


def ui_command(cmd: str, dangerous: bool = False) -> None:
    if dangerous:
        console.print(Panel(
            Text(cmd, style="bold white on dark_red"),
            title="[bold white]⚠   ОПАСНАЯ КОМАНДА   ⚠[/bold white]",
            border_style="bold red", box=DOUBLE, padding=(0, 2),
        ))
        LOGGER.write("DANGER_CMD", cmd)
    else:
        console.print(Panel(
            Text(cmd, style="bold yellow"),
            title="[bold yellow]⚙  КОМАНДА[/bold yellow]",
            border_style="yellow", box=ROUNDED, padding=(0, 2),
        ))
        LOGGER.write("CMD", cmd)


def ui_error(text: str) -> None:
    console.print(f"[bold red]✗[/bold red] [red]{text}[/red]")
    LOGGER.write("ERROR", text)


def ui_success(text: str) -> None:
    console.print(f"[bold green]✓[/bold green] [green]{text}[/green]")
    LOGGER.write("OK", text)


def ui_warning(text: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] [yellow]{text}[/yellow]")
    LOGGER.write("WARN", text)


def ui_info_table(settings: dict) -> None:
    table = Table(show_header=False, box=ROUNDED, border_style="dim cyan")
    table.add_column("Параметр", style="dim cyan")
    table.add_column("Значение", style="white")
    for k, v in settings.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print()
    LOGGER.write("CONFIG", " | ".join(f"{k}={v}" for k, v in settings.items()))