from __future__ import annotations

import re
import shutil
import textwrap
import time
from typing import Any
from urllib.parse import urlparse

from config.settings import settings
from src.interfaces.mcp_client import MCPBackend

try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover
    Console = None
    Columns = None
    Group = None
    Panel = None
    Rule = None
    Table = None
    Text = None
    Align = None
    box = None


class TerminalApp:
    """Интерактивный терминал для поиска."""

    _C_PRIMARY = "bold cyan"
    _C_ACCENT = "bold bright_green"
    _C_LABEL = "bold white"
    _C_DIM = "grey70"
    _C_MUTED = "grey50"
    _C_SUBTLE = "grey35"
    _C_FRAME = "grey23"
    _C_ERROR = "bold red"
    _C_LINK = "cyan underline"

    def __init__(self, backend: MCPBackend, title: str):
        self.backend = backend
        self.title = title
        self.console = Console(highlight=False, soft_wrap=True) if Console is not None else None
        self._status_snapshot: dict[str, str] | None = None
        self._request_started_at: float | None = None
        self._current_stage_key: str | None = None
        self._current_stage_label: str | None = None
        self._current_stage_started_at: float | None = None
        self._last_stage_metrics: list[tuple[str, float]] = []
        self._last_total_duration: float | None = None

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def run(self) -> None:
        self._print_title()

        try:
            while True:
                user_input = self._ask_input()
                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    self._print_goodbye()
                    break
                if user_input.startswith("/") and self._handle_command(user_input):
                    continue
                if not user_input:
                    continue

                try:
                    result = self._run_search(user_input)
                    self._print_answer(result)
                except Exception as exc:
                    self._print_error(str(exc))
        except (KeyboardInterrupt, EOFError):
            self._print_goodbye()
        finally:
            self.backend.close()

    # ---------------------------------------------------------------
    # Commands
    # ---------------------------------------------------------------

    def _handle_command(self, command: str) -> bool:
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._print_help()
            return True

        if cmd == "/provider":
            if len(parts) == 1:
                status_text = self.backend.llm_status()
                self._status_snapshot = self._parse_status_lines(status_text)
                self._print_message(status_text)
                return True
            provider = parts[1]
            model = parts[2] if len(parts) > 2 else None
            self._status_snapshot = None
            self._print_message(self.backend.llm_set(provider, model=model))
            return True

        if cmd == "/model":
            if len(parts) < 2:
                self._print_message(
                    "Укажите имя модели. Пример: /model qwen2.5:7b или /model openai/gpt-4o-mini"
                )
                return True
            model = " ".join(parts[1:]).strip()
            self._status_snapshot = None
            self._print_message(self.backend.llm_model(model))
            return True

        if cmd == "/status":
            status_text = self.backend.llm_status()
            self._status_snapshot = self._parse_status_lines(status_text)
            self._print_message(status_text)
            return True

        return False

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------

    def _run_search(self, user_input: str) -> str:
        self._reset_stage_metrics()

        def handle_notification(message: dict[str, Any]) -> None:
            stage_info = self._extract_stage_info(message)
            if stage_info is None:
                return
            self._advance_stage(stage_info["key"], stage_info["label"])

        try:
            if self.console:
                if settings.show_logs:
                    return self.backend.search(user_input, limit=5, notification_handler=handle_notification)

                with self.console.status(
                    self._build_stage_status("Агент решает, нужен ли веб-поиск"),
                    spinner="dots",
                    spinner_style="cyan",
                ) as status:
                    def spinner_handler(message: dict[str, Any]) -> None:
                        handle_notification(message)
                        stage_info = self._extract_stage_info(message)
                        if stage_info is not None:
                            status.update(self._build_stage_status(stage_info["message"]))

                    return self.backend.search(user_input, limit=5, notification_handler=spinner_handler)

            return self.backend.search(user_input, limit=5, notification_handler=handle_notification)
        finally:
            self._finish_search_with_metrics()

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    def _print_title(self) -> None:
        if not self.console:
            print(self.title)
            print("Команды: /provider  /model  /status  /help  /exit")
            return

        status_snapshot = self._load_status_snapshot()

        self.console.print()
        self.console.print(
            Panel(
                Group(
                    Align.center(Text(self.title, style="bold white")),
                    Align.center(Text("поиск, реранк и суммаризация через MCP", style=self._C_MUTED)),
                ),
                border_style=self._C_PRIMARY,
                box=box.ROUNDED,
                padding=(1, 3),
            ),
            crop=False,
        )

        panels = [
            Panel(
                self._build_session_table(status_snapshot),
                title="Сеанс",
                title_align="left",
                border_style=self._C_FRAME,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            Panel(
                self._build_commands_table(),
                title="Команды",
                title_align="left",
                border_style=self._C_FRAME,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
        ]

        if Columns is not None:
            self.console.print(Columns(panels, equal=True, expand=True))
        else:
            for panel in panels:
                self.console.print(panel, crop=False)

        self.console.print(Text("Введите запрос или используйте /help", style=self._C_MUTED))
        self.console.print()

    def _ask_input(self) -> str:
        if self.console:
            return self.console.input(f" [{self._C_PRIMARY}]>[/{self._C_PRIMARY}] ").strip()
        return input("\n > ").strip()

    def _print_answer(self, text: str) -> None:
        normalized = self._normalize_answer_sources(text)
        body, sources = self._split_sources_block(normalized)
        display_body = body.strip() if body.strip() else normalized
        display_body = self._clean_answer_body(display_body)
        model_label = self._get_model_display_name()

        if not self.console:
            print(f"\n{model_label}\n{display_body}")
            if sources:
                print("\nИсточники")
                for idx, link in enumerate(sources, 1):
                    print(f"[{idx}] {link}")
            return

        self.console.print()
        self._print_framed_block(model_label, display_body.strip(), kind="answer")

        if sources:
            self._print_framed_block("Источники", self._format_sources_text(sources), kind="sources")

        if settings.show_logs and self._last_stage_metrics:
            self._print_framed_block("Тайминг запроса", self._format_metrics_text(), kind="metrics")

    def _print_message(self, text: str) -> None:
        if not self.console:
            print(text)
            return

        status_table = self._render_status_table(text)
        title = "Сеанс" if status_table is not None else "Система"

        self.console.print()
        self.console.print(
            Panel(
                status_table or self._render_text_block(text),
                title=title,
                title_align="left",
                border_style=self._C_PRIMARY,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            crop=False,
        )

    def _print_error(self, text: str) -> None:
        if not self.console:
            print(f"Ошибка: {text}")
            return

        self.console.print()
        self.console.print(
            Panel(
                Text(text, style=self._C_ERROR),
                title="Ошибка",
                title_align="left",
                border_style=self._C_ERROR,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            crop=False,
        )

    def _print_goodbye(self) -> None:
        if not self.console:
            print("Завершение.")
            return

        self.console.print()
        self.console.print(Rule(style=self._C_SUBTLE))
        self.console.print(Text("Сессия завершена. До связи.", style=self._C_MUTED))
        self.console.print()

    def _print_help(self) -> None:
        if not self.console:
            help_text = (
                "Команды:\n"
                "  /provider                - показать текущий провайдер\n"
                "  /provider <name>         - переключить (direct, ollama или openrouter)\n"
                "  /provider <name> <model> - переключить и задать модель\n"
                "  /model <name>            - сменить модель текущего провайдера\n"
                "  /status                  - показать конфигурацию\n"
                "  /exit                    - выход"
            )
            print(help_text)
            return

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=self._C_DIM,
            border_style=self._C_SUBTLE,
            padding=(0, 2),
            expand=False,
        )
        table.add_column("Команда", style="bold bright_cyan", no_wrap=True)
        table.add_column("Описание", style="white")

        rows = [
            ("/provider", "показать активный провайдер"),
            ("/provider [name]", "переключить: direct, ollama или openrouter"),
            ("/provider [name] [model]", "сменить провайдера и задать модель"),
            ("/model [name]", "сменить текущую модель"),
            ("/status", "показать текущую конфигурацию"),
            ("/exit", "закрыть сессию"),
        ]
        for cmd, desc in rows:
            table.add_row(Text(cmd), Text(desc))

        self.console.print()
        self.console.print(
            Panel(
                table,
                title="Команды",
                title_align="left",
                border_style=self._C_PRIMARY,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            crop=False,
        )

    def _load_status_snapshot(self) -> dict[str, str]:
        if self._status_snapshot is None:
            try:
                self._status_snapshot = self._parse_status_lines(self.backend.llm_status())
            except Exception:
                self._status_snapshot = {}
        return self._status_snapshot

    def _build_session_table(self, snapshot: dict[str, str]):
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style=self._C_MUTED, no_wrap=True, width=10)
        table.add_column(style=self._C_LABEL)

        provider = snapshot.get("провайдер")
        model = snapshot.get("модель")
        searxng = snapshot.get("searxng")

        if provider:
            table.add_row("LLM", self._humanize_provider(provider))
        if model:
            table.add_row("Модель", self._humanize_model_name(model))
        if searxng:
            table.add_row("Поиск", self._shorten(self._humanize_search_label(searxng), max_len=42))
        return table

    def _build_commands_table(self):
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="bold bright_cyan", no_wrap=True)
        table.add_column(style="white")

        rows = [
            ("/provider", "сменить LLM-провайдера"),
            ("/model", "сменить активную модель"),
            ("/status", "посмотреть текущую конфигурацию"),
            ("/help", "показать все команды"),
            ("/exit", "закрыть сессию"),
            ("Ctrl+C", "быстрый выход"),
        ]
        for command, description in rows:
            table.add_row(command, description)
        return table

    def _render_text_block(self, text: str):
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        return Text(normalized, overflow="fold")

    def _render_sources_table(self, sources: list[str]):
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=self._C_DIM,
            border_style=self._C_FRAME,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", style=self._C_MUTED, no_wrap=True, width=3)
        table.add_column("Сайт", style="white", no_wrap=True, width=18)
        table.add_column("URL", style=self._C_LINK, overflow="fold")

        for idx, link in enumerate(sources, 1):
            table.add_row(str(idx), self._source_host(link), link)
        return table

    def _print_framed_block(self, title: str, content: str, kind: str = "answer") -> None:
        width = self._frame_width()
        inner_width = max(20, width - 6)
        frame_style = self._C_ACCENT if kind == "answer" else self._C_FRAME

        self.console.print(self._frame_top(title, width, frame_style))
        self.console.print(self._empty_frame_line(inner_width, frame_style), soft_wrap=False)
        for line in self._styled_content_lines(content, inner_width, kind=kind):
            self.console.print(self._compose_frame_line(line, inner_width, frame_style), soft_wrap=False)
        self.console.print(self._empty_frame_line(inner_width, frame_style), soft_wrap=False)
        self.console.print(self._frame_bottom(width, frame_style))

    def _frame_width(self) -> int:
        if self.console is not None:
            try:
                width = int(self.console.size.width)
            except Exception:
                width = 100
        else:
            width = shutil.get_terminal_size((100, 24)).columns
        return max(40, width)

    def _frame_top(self, title: str, width: int, frame_style: str):
        clean_title = (title or "").strip() or "Ответ"
        available = max(0, width - 4)
        prefix = f"─ {clean_title} "
        if len(prefix) > available:
            prefix = prefix[:available]
        line = Text("╭", style=frame_style)
        line.append("─ ", style=frame_style)
        line.append(clean_title[: max(0, available - 2)], style=self._title_style_for_block(title))
        line.append(" ", style=frame_style)
        line.append("─" * max(0, width - 2 - len(prefix)), style=frame_style)
        line.append("╮", style=frame_style)
        return line

    @staticmethod
    def _frame_bottom(width: int, frame_style: str):
        line = Text("╰", style=frame_style)
        line.append("─" * (width - 2), style=frame_style)
        line.append("╯", style=frame_style)
        return line

    @staticmethod
    def _wrap_content_lines(content: str, width: int) -> list[str]:
        text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
        wrapper = textwrap.TextWrapper(
            width=max(10, width),
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        lines: list[str] = []
        for raw_line in text.split("\n"):
            if not raw_line.strip():
                lines.append("")
                continue
            expanded = raw_line.expandtabs(4)
            wrapped = wrapper.wrap(expanded)
            lines.extend(wrapped or [""])
        if not lines:
            lines.append("")
        return lines

    def _format_sources_text(self, sources: list[str]) -> str:
        chunks = []
        for idx, link in enumerate(sources, 1):
            host = self._source_host(link)
            chunks.append(f"[{idx}] {host}\n    {link}")
        return "\n\n".join(chunks)

    def _styled_content_lines(self, content: str, width: int, kind: str) -> list[Text]:
        lines: list[Text] = []
        for raw_line in self._wrap_content_lines(content, width):
            if kind == "sources":
                lines.append(self._style_source_line(raw_line))
            elif kind == "metrics":
                lines.append(self._style_metrics_line(raw_line))
            else:
                lines.append(self._style_answer_line(raw_line))
        return lines

    def _compose_frame_line(self, content: Text, inner_width: int, frame_style: str):
        line = Text("│  ", style=frame_style)
        line.append_text(content)
        pad = max(0, inner_width - content.cell_len)
        if pad:
            line.append(" " * pad)
        line.append("  │", style=frame_style)
        return line

    def _empty_frame_line(self, inner_width: int, frame_style: str):
        return Text("│  " + (" " * inner_width) + "  │", style=frame_style)

    def _style_answer_line(self, line: str) -> Text:
        text = Text(line, style="white")

        label_match = re.match(r"^\s*([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9 /_-]{1,28}:)", line)
        if label_match:
            text.stylize(self._C_ACCENT, label_match.start(1), label_match.end(1))

        bullet_match = re.match(r"^(\s*(?:[-*•]|\d+[.)]))(\s+)", line)
        if bullet_match:
            text.stylize(self._C_PRIMARY, bullet_match.start(1), bullet_match.end(1))

        for match in re.finditer(r"\b(?:MCP|LLM|API|JSON|HTTP|HTTPS|SDK|CLI|RAG|AI|ИИ)\b", line):
            text.stylize("bold bright_magenta", match.start(), match.end())

        for match in re.finditer(r"\b\d+(?:[.,]\d+)?(?:%|x)?\b", line):
            text.stylize(self._C_PRIMARY, match.start(), match.end())

        return text

    def _style_source_line(self, line: str) -> Text:
        text = Text(line, style="white")
        stripped = line.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            return Text(line, style=self._C_LINK)

        ref_match = re.match(r"^(\[\d+\])\s+(.+)$", line.strip())
        if ref_match:
            start = line.find(ref_match.group(1))
            mid = start + len(ref_match.group(1))
            end = line.find(ref_match.group(2), mid)
            text.stylize(self._C_PRIMARY, start, mid)
            if end != -1:
                text.stylize(self._C_ACCENT, end, end + len(ref_match.group(2)))
        return text

    def _title_style_for_block(self, title: str) -> str:
        if title == "Источники":
            return self._C_PRIMARY
        if title == "Тайминг запроса":
            return self._C_PRIMARY
        return self._C_ACCENT

    def _render_status_table(self, text: str):
        snapshot = self._parse_status_lines(text)
        if not snapshot:
            return None

        table = Table.grid(padding=(0, 1), expand=False)
        table.add_column(style=self._C_MUTED, no_wrap=True)
        table.add_column(style=self._C_LABEL)

        label_map = {
            "провайдер": "Провайдер",
            "модель": "Модель",
            "searxng": "Поиск",
        }
        for label in ("провайдер", "модель", "searxng"):
            value = snapshot.get(label)
            if not value:
                continue
            if label == "провайдер":
                value = self._humanize_provider(value)
            elif label == "модель":
                value = self._humanize_model_name(value)
            elif label == "searxng":
                value = self._humanize_search_label(value)
            table.add_row(label_map[label], value)
        return table

    @staticmethod
    def _parse_status_lines(text: str) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("- ") or ":" not in line:
                continue
            label, value = line[2:].split(":", 1)
            snapshot[label.strip().lower()] = value.strip()
        return snapshot

    @staticmethod
    def _shorten(text: str, max_len: int = 48) -> str:
        clean = (text or "").strip()
        if len(clean) <= max_len:
            return clean
        head = max(1, (max_len - 3) // 2)
        tail = max(1, max_len - head - 3)
        return clean[:head].rstrip() + "..." + clean[-tail:].lstrip()

    @staticmethod
    def _normalize_answer_sources(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw
        marker = "Источники:"
        if marker not in raw:
            return raw

        before, _, after = raw.partition(marker)
        links = []
        for line in after.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            match = re.match(r"^\[(\d+)\]\s*(.+)$", cleaned)
            if match:
                links.append(match.group(2).strip())
                continue
            if cleaned.startswith("http://") or cleaned.startswith("https://"):
                links.append(cleaned)

        if not links:
            return raw

        formatted = "\n".join([f"[{i}] {url}" for i, url in enumerate(links, 1)])
        body = before.rstrip()
        return f"{body}\n\nИсточники:\n{formatted}"

    @staticmethod
    def _clean_answer_body(text: str) -> str:
        raw = (text or "").strip()
        return re.sub(r"^\s*Ответ\s*:\s*", "", raw, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _split_sources_block(text: str) -> tuple[str, list[str]]:
        marker = "Источники:"
        if marker not in text:
            return text, []

        before, _, after = text.partition(marker)
        links = []
        for line in after.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            match = re.match(r"^\[(\d+)\]\s*(.+)$", cleaned)
            if match:
                links.append(match.group(2).strip())

        if not links:
            return text, []
        return before.rstrip(), links

    def _get_model_display_name(self) -> str:
        snapshot = self._load_status_snapshot()
        provider = snapshot.get("провайдер", "")
        model = snapshot.get("модель", "")
        if provider == "direct" or not model or model == "-":
            return "Прямой ответ"
        return self._humanize_model_name(model)

    @staticmethod
    def _humanize_provider(provider: str) -> str:
        mapping = {
            "direct": "прямой",
            "local": "локально",
            "ollama": "Ollama",
            "openrouter": "OpenRouter",
        }
        return mapping.get((provider or "").strip().lower(), provider)

    @staticmethod
    def _humanize_model_name(model: str) -> str:
        raw = (model or "").strip()
        if not raw or raw == "-":
            return "Прямой ответ"

        cleaned = raw.split("/")[-1]
        cleaned = cleaned.split(":", 1)[0]
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        token_map = {
            "claude": "Claude",
            "deepseek": "DeepSeek",
            "gemma": "Gemma",
            "gpt": "GPT",
            "haiku": "Haiku",
            "llama": "Llama",
            "mini": "mini",
            "mistral": "Mistral",
            "opus": "Opus",
            "phi": "Phi",
            "qwen": "Qwen",
            "sonnet": "Sonnet",
            "turbo": "Turbo",
        }

        pretty_tokens = []
        for token in cleaned.split():
            low = token.lower()
            if low in token_map:
                pretty_tokens.append(token_map[low])
            elif re.fullmatch(r"[a-z]\d+(?:\.\d+)?", low):
                pretty_tokens.append(low[0].upper() + low[1:])
            else:
                pretty_tokens.append(token[:1].upper() + token[1:])
        return " ".join(pretty_tokens)

    @staticmethod
    def _humanize_search_label(label: str) -> str:
        raw = (label or "").strip()
        if not raw:
            return "-"

        if "->" in raw:
            _, _, target = raw.rpartition("->")
            host = TerminalApp._source_host(target.strip())
            if host:
                return f"SearxNG через {host}"

        host = TerminalApp._source_host(raw)
        if host and host != raw:
            return host
        return raw

    @staticmethod
    def _source_host(link: str) -> str:
        raw = (link or "").strip()
        if not raw:
            return "-"
        if "://" not in raw:
            return raw.replace("MCP:", "MCP ").strip()

        parsed = urlparse(raw)
        host = parsed.netloc or parsed.path
        return host.removeprefix("www.") if host else raw

    def _build_stage_status(self, message: str) -> str:
        badge = self._stage_badge(message)
        return f"[{self._C_PRIMARY}]{badge}[/{self._C_PRIMARY}]"

    @staticmethod
    def _stage_badge(message: str) -> str:
        lowered = (message or "").lower()
        if "решает" in lowered or "нужен ли" in lowered:
            return "Думаю"
        if "ищу" in lowered or "поиск" in lowered or "searxng" in lowered:
            return "Ищу"
        if "ранжир" in lowered or "реранк" in lowered:
            return "Ранжирую"
        if "извлекаю" in lowered or "загружаю" in lowered:
            return "Извлекаю"
        if "суммариз" in lowered or "собираю" in lowered:
            return "Суммаризирую"
        if "формулирую" in lowered or "ответ без веб-поиска" in lowered:
            return "Отвечаю"
        if "готов" in lowered:
            return "Готово"
        return "Работаю"

    @staticmethod
    def _extract_stage_message(notification: dict[str, Any]) -> str | None:
        method = notification.get("method")
        params = notification.get("params")
        if not isinstance(params, dict):
            return None

        if method == "notifications/progress":
            message = params.get("message")
            return message if isinstance(message, str) and message.strip() else None

        if method == "notifications/message":
            data = params.get("data")
            return data if isinstance(data, str) and data.strip() else None

        return None

    def _extract_stage_info(self, notification: dict[str, Any]) -> dict[str, str] | None:
        message = self._extract_stage_message(notification)
        if not message:
            return None

        lowered = message.lower()
        if "решает" in lowered or "нужен ли" in lowered:
            return {"key": "decision", "label": "Решение агента", "message": message}
        if "формулирую" in lowered or "ответ без веб-поиска" in lowered:
            return {"key": "direct_answer", "label": "Ответ без веб-поиска", "message": message}
        if "ищу источники" in lowered or ("ищу" in lowered and "searxng" in lowered):
            return {"key": "search", "label": "Поиск SearxNG", "message": message}
        if "первично ранжирую" in lowered:
            return {"key": "stage1_rerank", "label": "Первичный реранк", "message": message}
        if "загружаю страницы" in lowered or "извлекаю текст" in lowered:
            return {"key": "extract", "label": "Извлечение и обработка", "message": message}
        if "финальный реранк" in lowered:
            return {"key": "stage2_rerank", "label": "Финальный реранк", "message": message}
        if "суммаризирую" in lowered or "собираю итоговый ответ" in lowered:
            return {"key": "summarize", "label": "Суммаризация", "message": message}
        if "ответ готов" in lowered:
            return {"key": "done", "label": "Готово", "message": message}
        if "не найдено" in lowered:
            return {"key": "done", "label": "Без результатов", "message": message}
        return None

    def _reset_stage_metrics(self) -> None:
        self._request_started_at = time.perf_counter()
        self._current_stage_key = None
        self._current_stage_label = None
        self._current_stage_started_at = None
        self._last_stage_metrics = []
        self._last_total_duration = None

    def _advance_stage(self, stage_key: str, stage_label: str) -> None:
        now = time.perf_counter()

        if stage_key == "done":
            self._close_current_stage(now)
            return

        if self._current_stage_key == stage_key:
            return

        self._close_current_stage(now)
        self._current_stage_key = stage_key
        self._current_stage_label = stage_label
        self._current_stage_started_at = now

    def _close_current_stage(self, finished_at: float) -> None:
        if self._current_stage_key is None or self._current_stage_started_at is None or self._current_stage_label is None:
            return
        duration = max(0.0, finished_at - self._current_stage_started_at)
        self._last_stage_metrics.append((self._current_stage_label, duration))
        self._current_stage_key = None
        self._current_stage_label = None
        self._current_stage_started_at = None

    def _finish_search_with_metrics(self) -> None:
        finished_at = time.perf_counter()
        self._close_current_stage(finished_at)
        if self._request_started_at is not None:
            self._last_total_duration = max(0.0, finished_at - self._request_started_at)

    def _format_metrics_text(self) -> str:
        rows = list(self._last_stage_metrics)
        if self._last_total_duration is not None:
            rows.append(("Всего", self._last_total_duration))
        if not rows:
            return ""

        label_width = max(len("Этап"), max(len(label) for label, _ in rows))
        value_width = max(len("Время"), max(len(self._format_duration(value)) for _, value in rows))

        lines = [
            f"{'Этап'.ljust(label_width)}  {'Время'.rjust(value_width)}",
            f"{'─' * label_width}  {'─' * value_width}",
        ]
        for label, duration in rows:
            lines.append(f"{label.ljust(label_width)}  {self._format_duration(duration).rjust(value_width)}")
        return "\n".join(lines)

    def _style_metrics_line(self, line: str) -> Text:
        text = Text(line, style="white")
        stripped = line.strip()

        if stripped.startswith("Этап") or stripped.startswith("─"):
            return Text(line, style=self._C_MUTED)

        if stripped.startswith("Всего"):
            text.stylize(self._C_ACCENT, 0, len(line))

        match = re.search(r"(\d+(?:[.,]\d+)?\s*с)$", line)
        if match:
            text.stylize(self._C_PRIMARY, match.start(1), match.end(1))
        return text

    @staticmethod
    def _format_duration(duration: float) -> str:
        return f"{duration:.2f} с"
