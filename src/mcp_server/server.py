import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("searxng_agent")

def create_mcp_server(agent):
    """Создаёт и настраивает MCP-сервер."""
    mcp = FastMCP("searxng-agent")

    @mcp.tool()
    async def search(query: str, limit: int = 10) -> str:
        """Поиск с автоматическим выбором режима (direct или LLM)."""
        logger.info("Получен запрос пользователя: %r", query)

        provider = agent.get_status().get("provider", "direct")
        logger.info("Текущий провайдер поиска: %s", provider)

        if provider == "direct":
            try:
                results = agent.search(query, limit=limit)
            except Exception as exc:
                logger.error("search: ошибка direct: %s", exc)
                return f"Ошибка: {exc}"
            return _format_simple_results(results, limit)

        try:
            return agent.smart_search(query, limit=limit)
        except Exception as exc:
            logger.error("search: ошибка LLM: %s", exc)
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


def _format_simple_results(results, limit: int) -> str:
    if not results:
        return "Ничего не найдено."
    
    output = []

    for i, res in enumerate(results[:limit], 1):
        content = res.get("content", "") or ""
        snippet = content[:300]

        if len(content) > 300:
            snippet += "..."

        output.append(
            f"[{i}] {res.get('title', '')}\n"
            f"    Источник: {res.get('source', '')}\n"
            f"    URL: {res.get('url', '')}\n"
            f"    Фрагмент: {snippet}"
        )
    
    return "\n\n".join(output)
