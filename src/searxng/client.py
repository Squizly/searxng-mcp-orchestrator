from __future__ import annotations

import atexit
import asyncio
import json
import logging
import queue
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config.settings import settings
from src.mcp_stdio import MCPStdioClient

logger = logging.getLogger("searxng_agent")

class SearxngClient:
    """MCP-клиент для поиска через внутренний tool server."""

    def __init__(self, server_cmd: Optional[Sequence[str]] = None, cwd: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parents[2]
        instances = settings.searxng_instances_list
        if not instances:
            raise ValueError("Не задан ни один инстанс SearxNG")
        self.instance_url = instances[0].rstrip("/")
        self.tool_name = "searxng_search"
        self.pool_size = min(max(1, settings.searxng_parallelism), 3)
        self._closed = False

        resolved_cmd = list(server_cmd) if server_cmd else [
            sys.executable,
            str(repo_root / "src" / "main.py"),
            "--searxng-mcp-server",
        ]
        resolved_cwd = cwd or repo_root

        self._clients: List[MCPStdioClient] = []
        self._client_pool: queue.Queue[MCPStdioClient] = queue.Queue()
        for _ in range(self.pool_size):
            client = MCPStdioClient(resolved_cmd, cwd=resolved_cwd)
            self._clients.append(client)
            self._client_pool.put(client)

        self.backend_label = f"MCP:{self.tool_name} x{self.pool_size} -> {self.instance_url}"
        atexit.register(self.close)
        logger.info("MCP-клиент поиска инициализирован: %s", self.backend_label)

    async def search(
        self,
        query: str,
        language: Optional[str] = None,
        categories: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Выполняет поиск через MCP tool calling."""

        params = {
            "query": query,
            "language": language,
            "categories": categories,
            "limit": limit,
        }
        return await asyncio.to_thread(self._search_via_mcp, query, params)

    def _search_via_mcp(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        client = self._client_pool.get()
        try:
            response_text = client.call_tool(self.tool_name, params)
        finally:
            self._client_pool.put(client)
        return self._parse_results(query, response_text)

    def _parse_results(self, query: str, response_text: str) -> List[Dict[str, Any]]:
        if not response_text:
            raise RuntimeError("MCP search tool вернул пустой ответ")

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("MCP search tool | Некорректный JSON: %r", response_text[:300])
            raise RuntimeError("MCP search tool вернул некорректный JSON") from exc

        raw_results = self._extract_results(data)

        logger.info("MCP search tool | query=%r | results=%d", query, len(raw_results))
        return raw_results

    def _extract_results(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            if not payload:
                return []
            if all(isinstance(item, dict) for item in payload):
                known_result_fields = {"url", "title", "content", "engine", "score", "source"}
                if any(set(item.keys()) & known_result_fields for item in payload):
                    return payload
            raise RuntimeError("MCP search tool вернул список неожиданной структуры")

        if isinstance(payload, str):
            if not payload.strip():
                return []
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError("MCP search tool вернул неожиданный строковый формат ответа") from exc
            return self._extract_results(decoded)

        if not isinstance(payload, dict):
            raise RuntimeError("MCP search tool вернул неожиданный формат ответа")

        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))

        results = payload.get("results")
        if isinstance(results, list):
            return results
        if results is not None:
            return self._extract_results(results)

        for key in ("result", "data", "payload", "structuredContent", "content"):
            if key not in payload:
                continue
            nested = payload.get(key)
            try:
                extracted = self._extract_results(nested)
                return extracted
            except RuntimeError:
                continue

        payload_keys = ", ".join(sorted(payload.keys())) or "<empty>"
        raise RuntimeError(f"MCP search tool вернул неожиданный формат: {payload_keys}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for client in self._clients:
            client.close()
