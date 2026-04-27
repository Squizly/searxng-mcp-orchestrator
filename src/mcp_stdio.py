from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from mcp import types as mcp_types

logger = logging.getLogger("searxng_agent")


class MCPStdioClient:
    """Мини-клиент MCP поверх stdio."""

    def __init__(self, server_cmd: Sequence[str], cwd: Optional[Path] = None):
        self._req_id = 0
        self._io_lock = threading.Lock()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            list(server_cmd),
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

    def _request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        notification_handler: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        with self._io_lock:
            self._req_id += 1
            req_id = self._req_id
            request_params = dict(params or {})
            if method == "tools/call" and notification_handler is not None:
                meta = dict(request_params.get("_meta") or {})
                meta.setdefault("progressToken", f"tool-call-{req_id}")
                request_params["_meta"] = meta
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": request_params,
                "id": req_id,
            }
            self._send(payload)

            while True:
                msg = self._read()
                if msg.get("id") == req_id:
                    return msg
                if notification_handler is not None and "method" in msg:
                    try:
                        notification_handler(msg)
                    except Exception:
                        logger.debug("Ошибка обработки MCP-уведомления", exc_info=True)
                logger.debug("Пропуск уведомления MCP: %s", msg)

    def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        with self._io_lock:
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

    def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        notification_handler: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        resp = self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            notification_handler=notification_handler,
        )
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
                    joined = "\n".join([p for p in parts if p])
                    if joined.strip():
                        return joined
                if isinstance(content, str):
                    if content.strip():
                        return content
            if "structuredContent" in result:
                structured = result["structuredContent"]
                if isinstance(structured, str) and structured.strip():
                    return structured
                if isinstance(structured, dict):
                    nested_result = structured.get("result")
                    if isinstance(nested_result, str) and nested_result.strip():
                        return nested_result
                    if structured:
                        return json.dumps(structured, ensure_ascii=False)
            if "text" in result and isinstance(result["text"], str):
                return result["text"]
            if "result" in result:
                nested = result["result"]
                if isinstance(nested, str):
                    return nested
                return json.dumps(nested, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
