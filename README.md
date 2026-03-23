# SearxNG MCP Оркестратор

Интерактивный поисковый агент на базе SearxNG с поддержкой MCP (Model Context Protocol) и подключаемых LLM.

## Быстрый старт
1. Запустите SearxNG (см. `searxng-docker/docker-compose.yml`).
2. Убедитесь, что Ollama запущена и нужная модель доступна.
3. Скопируйте `.env.example` в `.env` и при необходимости обновите параметры.
4. Запустите терминальный клиент:
`python src/main.py`

## Запуск
Запуск всегда через терминал с MCP (сервер поднимается автоматически внутри).

## Команды терминала
`/provider` — показать текущего провайдера.
`/provider <name>` — переключить провайдера (direct, ollama или openrouter).
`/provider <name> <model>` — переключить провайдера и задать модель.
`/model <name>` — сменить модель текущего провайдера.
`/status` — показать конфигурацию.
`/help` — подсказка.
`/exit` — выход.

## Пример .env
```
LLM_PROVIDER=ollama

OLLAMA_MODEL_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3.2

OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api

SEARXNG_INSTANCES=http://localhost:8080
SEARXNG_DEFAULT_LANGUAGE=ru
SEARXNG_DEFAULT_CATEGORIES=general
SEARXNG_PARALLELISM=3

LOG_LEVEL=INFO
LOG_CONSOLE_LEVEL=INFO
LOG_FILE=logs/results.log
```

## Логи
Логи пишутся в `logs/results.log`.

## Структура
`config/` — настройки.
`src/` — исходный код.
`src/main.py` — точка входа.
`src/agent/` — логика агента.
`src/interfaces/` — терминальный интерфейс.
`src/llm/` — провайдеры LLM.
`src/mcp_server/` — MCP-сервер.
`src/searxng/` — клиент SearxNG.
