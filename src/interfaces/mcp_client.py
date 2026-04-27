from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

from src.mcp_stdio import MCPStdioClient


class MCPBackend:
    """Бэкенд, который общается с MCP-сервером."""

    def __init__(self, repo_root: Path):
        server_cmd = [
            sys.executable,
            str(repo_root / "src" / "main.py"),
            "--mcp-server",
        ]
        self.client = MCPStdioClient(server_cmd, cwd=repo_root)

    def search(
        self,
        query: str,
        limit: int = 5,
        notification_handler: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        return self.client.call_tool(
            "search",
            {"query": query, "limit": limit},
            notification_handler=notification_handler,
        )

    def llm_status(self) -> str:
        return self.client.call_tool("llm_status", {})

    def llm_set(self, provider: str, model: str | None = None) -> str:
        return self.client.call_tool("llm_set", {"provider": provider, "model": model})

    def llm_model(self, model: str) -> str:
        return self.client.call_tool("llm_model", {"model": model})

    def close(self) -> None:
        self.client.close()
