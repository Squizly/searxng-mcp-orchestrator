from __future__ import annotations

import json
import logging
import re

from mcp.server.fastmcp import Context, FastMCP
from src.searxng import SearxngHttpClient

logger = logging.getLogger("searxng_agent")


def create_mcp_server(agent):
    """Создаёт и настраивает MCP-сервер."""
    mcp = FastMCP("searxng-agent")

    @mcp.tool()
    async def search(query: str, limit: int = 10, ctx: Context | None = None) -> str:
        """Агентный поиск (2-й пайплайн): решение искать или отвечать напрямую."""
        logger.info("Получен запрос пользователя: %r", query)

        async def emit_progress(message: str, current: float, total: float = 100) -> None:
            if ctx is None:
                return
            await ctx.report_progress(current, total=total, message=message)

        try:
            return await agent.smart_search(query, limit=limit, progress=emit_progress)
        except Exception as exc:
            logger.error("search: ошибка пайплайна: %s", exc)
            return f"Ошибка: {exc}"

    @mcp.tool()
    async def llm_status() -> str:
        """Показать текущий LLM-провайдер и модель."""
        status = agent.get_status()
        
        return (
            "Текущая конфигурация LLM:\n"
            f"- Провайдер: {status['provider']}\n"
            f"- Модель: {status['model']}\n"
            f"- SearxNG: {status['searxng']}"
        )

    @mcp.tool()
    async def llm_set(provider: str, model: str | None = None) -> str:
        """Переключить LLM-провайдера (direct, ollama или openrouter)."""
        return agent.set_provider(provider, model=model)

    @mcp.tool()
    async def llm_model(model: str) -> str:
        """Сменить модель текущего провайдера."""
        return agent.set_model(model)

    return mcp


def create_searxng_search_server():
    """Создаёт внутренний MCP-сервер с raw search tool для SearxNG."""
    mcp = FastMCP("searxng-search-backend")
    client = SearxngHttpClient()

    @mcp.tool()
    async def searxng_search(
        query: str,
        limit: int = 10,
        language: str | None = None,
        categories: str | None = None,
    ) -> str:
        """Поиск в SearxNG и возврат сырых результатов в JSON."""
        logger.info("Внутренний MCP search tool | query=%r | limit=%d", query, limit)
        results = await client.search(
            query,
            language=language,
            categories=categories,
            limit=limit,
        )
        return json.dumps({"results": results}, ensure_ascii=False)

    return mcp


def _format_simple_results(results, limit: int) -> str:
    if not results:
        return "Ничего не найдено."

    output = []
    separator = "\n" + ("-" * 80) + "\n"

    for i, res in enumerate(results[:limit], 1):
        source = str(res.get("content_source") or "unknown")
        content = res.get("display_content") or res.get("content") or ""
        full_text = _plain_text(str(content))
        if not full_text:
            full_text = "[Пустой извлеченный текст]"

        output.append(
            f"[{i}] {res.get('title', '')}\n"
            f"Источник поиска: {res.get('source', '')}\n"
            f"Контент: {source}\n"
            f"URL: {res.get('url', '')}\n"
            f"Извлеченный текст:\n{full_text}"
        )

    return separator.join(output)


def _plain_text(text: str) -> str:
    if not text:
        return ""

    no_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    no_md = re.sub(r"[#*_`>|~]{1,}", " ", no_links)

    lines = []
    for raw_line in no_md.replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split()).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
