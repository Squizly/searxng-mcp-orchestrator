from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _prepare_sys_path() -> Path:
    """Добавляет корень репозитория в sys.path и возвращает путь к нему."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> None:
    """Точка входа приложения."""
    repo_root = _prepare_sys_path()

    from src.utils.logger import setup_logger
    from src.agent import SearchAgent
    from src.interfaces.terminal import TerminalApp, MCPBackend
    from src.mcp_server import create_mcp_server

    setup_logger()
    if "--mcp-server" in sys.argv:
        try:
            agent = SearchAgent()
        except Exception as exc:
            print(f"Не удалось создать агента: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            mcp_server = create_mcp_server(agent)
            mcp_server.run()
        except Exception as exc:
            print(f"Не удалось запустить MCP-сервер: {exc}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        return

    try:
        backend = MCPBackend(repo_root)
    except Exception as exc:
        print(f"Не удалось запустить MCP-клиент: {exc}", file=sys.stderr)
        sys.exit(1)
    TerminalApp(backend, title="SearxNG Search Agent (режим MCP)").run()
    return


if __name__ == "__main__":
    main()
