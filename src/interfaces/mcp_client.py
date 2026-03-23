from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from mcp import types as mcp_types

logger = logging.getLogger("searxng_agent")


class MCPStdioClient:
    """Мини-клиент MCP поверх stdio."""

    def __init__(self, server_cmd: list[str], cwd: Optional[Path] = None):
        self._req_id = 0
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=False,
            bufsize=0,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        self._initialize()

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc.stdin:
            raise RuntimeError("Ввод MCP-сервера закрыт (stdin)")

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if not self.proc.stdout:
            raise RuntimeError("Вывод MCP-сервера закрыт (stdout)")

        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("MCP-сервер закрыл соединение")

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Пропуск не-JSON строки от MCP-сервера: %r", line)

    def _request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._req_id,
        }
        self._send(payload)

        while True:
            msg = self._read()
            if msg.get("id") == self._req_id:
                return msg
            logger.debug("Пропуск уведомления MCP: %s", msg)

    def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(payload)

    def _initialize(self) -> None:
        init_params = {
            "protocolVersion": mcp_types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "terminal-client", "version": "0.1"},
        }
        resp = self._request("initialize", init_params)

        if "error" in resp:
            raise RuntimeError(f"Инициализация MCP не удалась: {resp['error']}")

        self._notify("notifications/initialized", None)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return self._extract_text(resp.get("result"))

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "content" in result:
                content = result.get("content")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
                    return "\n".join([p for p in parts if p])
                if isinstance(content, str):
                    return content
            if "text" in result and isinstance(result["text"], str):
                return result["text"]
        return json.dumps(result, ensure_ascii=False, indent=2)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MCPBackend:
    """Бэкенд, который общается с MCP-сервером."""

    def __init__(self, repo_root: Path):
        server_cmd = [
            sys.executable,
            str(repo_root / "src" / "main.py"),
            "--mcp-server",
        ]
        self.client = MCPStdioClient(server_cmd, cwd=repo_root)

    def search(self, query: str, limit: int = 5) -> str:
        return self.client.call_tool("search", {"query": query, "limit": limit})

    def llm_status(self) -> str:
        return self.client.call_tool("llm_status", {})

    def llm_set(self, provider: str, model: str | None = None) -> str:
        return self.client.call_tool("llm_set", {"provider": provider, "model": model})

    def llm_model(self, model: str) -> str:
        return self.client.call_tool("llm_model", {"model": model})

    def close(self) -> None:
        self.client.close()
