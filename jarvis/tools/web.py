"""Веб-инструменты: поиск (SearXNG/DDG) и чтение страниц."""
import re

import requests

from jarvis.tools.searxng import SearxngManager
from jarvis.ui.console import ui_step, ui_warning

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class WebTools:
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )

    def __init__(self, max_results: int = 5, max_chars: int = 4000,
                 timeout: int = 12, searxng: SearxngManager | None = None,
                 priority: str = "searxng-first"):
        self.max_results = max_results
        self.max_chars = max_chars
        self.timeout = timeout
        self.searxng = searxng
        self.priority = priority
        self._searxng_last_error: str | None = None

    def _search_searxng(self, query: str) -> str:
        if not self.searxng or not self.searxng.enabled:
            return ""
        try:
            r = requests.get(
                f"{self.searxng.url}/search",
                params={"q": query, "format": "json",
                        "safesearch": 0, "language": "ru"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self._searxng_last_error = str(e)
            ui_warning(f"SearXNG недоступен: {e}")
            return ""

        results = (data.get("results") or [])[: self.max_results]
        self._searxng_last_error = None
        if not results:
            return ""
        lines = []
        for i, res in enumerate(results, 1):
            title = (res.get("title") or "").strip()
            content = (res.get("content") or "").strip()
            url = (res.get("url") or "").strip()
            lines.append(f"{i}. {title}\n   {content}\n   URL: {url}")
        return "\n\n".join(lines)

    def _search_ddg(self, query: str) -> str:
        if not DDGS_AVAILABLE:
            return "Поиск недоступен: не установлен 'ddgs'."
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=self.max_results))
        except Exception as e:
            return f"Ошибка поиска DuckDuckGo: {e}"
        if not results:
            return f"Ничего не найдено по запросу «{query}»."
        lines = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or r.get("url") or "").strip()
            lines.append(f"{i}. {title}\n   {body}\n   URL: {href}")
        return "\n\n".join(lines)

    @staticmethod
    def _is_failure(text: str) -> bool:
        if not text:
            return True
        bad_starts = ("Ошибка", "Ничего не найдено", "Поиск недоступен",
                      "Пустой", "SearXNG не дал")
        return text.startswith(bad_starts)

    def search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Пустой поисковый запрос."

        if self.priority == "ddg-only":
            ui_step("🔎", f"DuckDuckGo: {query}", "cyan")
            return self._search_ddg(query)

        if self.priority == "ddg-first":
            ui_step("🔎", f"DuckDuckGo (приоритет): {query}", "cyan")
            out = self._search_ddg(query)
            if not self._is_failure(out):
                return out
            if self.searxng and self.searxng.enabled:
                ui_warning("DuckDuckGo не дал результатов, пробую SearXNG.")
                sx = self._search_searxng(query)
                if sx:
                    return sx
            return out

        if self.searxng and self.searxng.enabled:
            ui_step("🌐", f"SearXNG: {query}", "cyan")
            out = self._search_searxng(query)
            if out:
                return out
            if self._searxng_last_error:
                ui_warning("SearXNG недоступен, переключаюсь на DuckDuckGo.")
            else:
                ui_warning("SearXNG не дал результатов, переключаюсь на DuckDuckGo.")
        ui_step("🔎", f"DuckDuckGo: {query}", "cyan")
        return self._search_ddg(query)

    def fetch(self, url: str) -> str:
        url = (url or "").strip().strip("\"'`")
        if not url.startswith(("http://", "https://")):
            return f"Некорректный URL: {url}"
        ui_step("📄", f"Чтение: {url}", "cyan")
        try:
            r = requests.get(
                url, headers={"User-Agent": self.USER_AGENT},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except Exception as e:
            return f"Не удалось загрузить {url}: {e}"

        text = ""
        if TRAFILATURA_AVAILABLE:
            try:
                text = trafilatura.extract(
                    r.text, include_comments=False, include_tables=False
                ) or ""
            except Exception:
                text = ""

        if not text and BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                    tag.decompose()
                text = " ".join(soup.get_text(separator=" ").split())
            except Exception:
                text = ""

        if not text:
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = " ".join(text.split())

        if not text:
            return f"Страница {url} не содержит извлекаемого текста."

        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + " … [обрезано]"

        return f"Источник: {url}\n\n{text}"