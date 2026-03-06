# 🔍 SearXNG MCP Orchestrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-Supported-green.svg)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

**SearXNG MCP Orchestrator** — это серверная архитектура, реализующая интеграцию метапоисковой системы [SearxNG](https://docs.searxng.org/) и больших языковых моделей (LLM) через протокол **Model Context Protocol (MCP)**. 

Проект разработан в рамках научно-исследовательской работы и решает проблему изоляции языковых моделей, обеспечивая им приватный, анонимный и защищенный от Rate-Limit блокировок доступ к актуальной информации в интернете.

## Структура проекта
```text
searxng-mcp-orchestrator/
├── docker/                     # Инфраструктура: SearxNG + Redis (docker-compose)
├── src/
│   ├── mcp_server/             # Ядро системы: FastMCP сервер, маршрутизация
│   │   └── logic/              # HTTP-клиент с Retry-логикой и очистка HTML (BeautifulSoup)
│   └── integration/            # Клиентские скрипты
└── tests/                      # Модульные тесты