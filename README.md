<p align="center">
  <img
    src="https://github.com/user-attachments/assets/58157f17-37e6-424d-a1cf-f847de14ec72"
    alt="SearxNG MCP Orchestrator Banner"
    width="100%"
    style="max-width: 1100px; border-radius: 18px;"
  />
</p>

<div align="center">

# SearxNG MCP Orchestrator

CLI-агент для поиска и ответов на естественном языке поверх **SearxNG**, **MCP** и подключаемых **LLM**.

Маршрутизирует запросы, вызывает веб-поиск только когда это действительно нужно, делает двухступенчатый реранк результатов и возвращает аккуратный терминальный ответ с источниками и таймингом пайплайна.

<p>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MCP-Model%20Context%20Protocol-111111" alt="MCP">
  <img src="https://img.shields.io/badge/Search-SearxNG-4B8BBE" alt="SearxNG">
  <img src="https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenRouter-0F172A" alt="LLM providers">
  <img src="https://img.shields.io/badge/UI-Rich%20CLI-2D3748" alt="Rich CLI">
  <img src="https://img.shields.io/badge/Rerank-FlashRank%20%2B%20Jina-0A7E5C" alt="Rerank">
</p>

</div>

## Что внутри

Проект состоит из трёх связанных частей:

- **терминальный клиент** с интерактивным CLI-интерфейсом;
- **агент** с роутингом `поиск / прямой ответ`;
- **retrieval-пайплайн** поверх SearxNG, FlashRank, `trafilatura` и Jina reranker.

Внутренне это MCP-приложение: терминальный клиент говорит с MCP-сервером агента, а агент, в свою очередь, вызывает внутренний MCP search tool для SearxNG. Такое разделение упрощает эксперименты с пайплайном, LLM-провайдерами и интерфейсом, не смешивая эти слои в одном модуле.

## Как это выглядит

### Обычный режим

```text
 > Что такое Model Context Protocol?

╭─ Qwen 2.5 ───────────────────────────────────────────────────────────────────────────────╮
│                                                                                          │
│  Model Context Protocol (MCP) — это открытый протокол прикладного уровня для             │
│  взаимодействия языковых моделей с внешними инструментами, данными и                     │
│  удаленными или локальными системами.                                                    │
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Источники ──────────────────────────────────────────────────────────────────────────────╮
│                                                                                          │
│  [1] docs.anthropic.com                                                                  │
│      https://docs.anthropic.com/en/docs/agents-and-tools/mcp                             │
│                                                                                          │
│  [2] ru.wikipedia.org                                                                    │
│      https://ru.wikipedia.org/wiki/Model_Context_Protocol                                │
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

### Режим логирования

```text
 > Объясни психологию Пиаже
01:50:07 · Получен запрос пользователя: 'Объясни психологию Пиаже'
01:50:08 · Второй пайплайн | Решение агента | use_search=True
01:50:11 · SearxNG HTTP | query='Объясни психологию Пиаже' | found=21 | returning=20
01:50:12 · Retrieval pipeline | query='Объясни психологию Пиаже' | stage1=20 -> top=10
01:50:14 · Retrieval pipeline | query='Объясни психологию Пиаже' | fulltext_extracted=9/10
01:50:17 · Retrieval pipeline | query='Объясни психологию Пиаже' | stage2=9 -> final=2

╭─ Тайминг запроса ────────────────────────────────────────────────────────────────────────╮
│                                                                                          │
│  Этап                     Время                                                          │
│  ──────────────────────  ──────                                                          │
│  Решение агента          0.81 с                                                          │
│  Поиск SearxNG           2.34 с                                                          │
│  Первичный реранк        0.29 с                                                          │
│  Извлечение и обработка  3.02 с                                                          │
│  Финальный реранк        1.12 с                                                          │
│  Суммаризация            1.23 с                                                          │
│  Всего                   8.81 с                                                          │
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

## Быстрый старт

### 1. Установить зависимости Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Поднять SearxNG

```bash
docker compose -f searxng-docker/docker-compose.yml up -d
```

### 3. Настроить окружение

```bash
cp .env.example .env
```

Если используется локальная модель через Ollama, убедитесь, что сервис поднят и нужная модель доступна:

```bash
ollama pull qwen2.5:3b
```

### 4. Запустить CLI

```bash
python src/main.py
```

## CLI и режимы работы

| Провайдер | Что делает | Когда использовать |
| --- | --- | --- |
| `direct` | отключает LLM и оставляет только retrieval/поиск | отладка пайплайна поиска |
| `ollama` | использует локальную модель через Ollama API | локальные эксперименты и офлайн-сценарии |
| `openrouter` | использует облачную модель через OpenRouter | когда нужен внешний провайдер или более сильная модель |

Команды CLI:

| Команда | Назначение |
| --- | --- |
| `/provider` | показать текущий провайдер |
| `/provider <name>` | переключить провайдера |
| `/provider <name> <model>` | переключить провайдера и сразу задать модель |
| `/model <name>` | сменить модель текущего провайдера |
| `/status` | показать активную конфигурацию |
| `/help` | вывести справку |
| `/exit` | завершить сессию |

## Архитектура



## Пайплайн retrieval

Используемая схема:

1. Запрос уходит в SearxNG и получает начальный пул документов.
2. Первый реранк выполняется через `FlashRank` (`ms-marco-MultiBERT-L-12`).
3. Для top-k документов скачиваются страницы и извлекается основной текст через `trafilatura`.
4. Второй реранк выполняется через `jinaai/jina-reranker-v2-base-multilingual`.
5. В итог идут самые релевантные документы, которые затем суммаризирует агент.

В agent mode этот пайплайн запускается только если роутер решает, что прямого ответа без веб-поиска недостаточно.

## Конфигурация

Ключевые переменные окружения:

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | активный провайдер: `direct`, `ollama`, `openrouter` |
| `OLLAMA_MODEL_NAME` | `qwen2.5:3b` | локальная модель Ollama |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | модель OpenRouter |
| `SEARXNG_INSTANCES` | `http://localhost:8080` | один или несколько SearxNG URL |
| `SEARXNG_PARALLELISM` | `3` | размер внутреннего пула MCP-клиентов поиска |
| `RETRIEVAL_INITIAL_RESULTS_N` | `20` | размер первичного пула документов |
| `RETRIEVAL_STAGE1_TOP_K` | `5` | сколько документов оставить после FlashRank |
| `RETRIEVAL_STAGE2_TOP_K` | `2` | сколько документов оставить после второго реранка |
| `SMART_PIPELINE_STAGE1_TOP_K` | `10` | top-k для агентного retrieval-пайплайна |
| `SMART_PIPELINE_FINAL_TOP_K` | `2` | сколько документов суммаризировать в agent mode |
| `LLM_TIMEOUT` | `120` | таймаут запросов к модели |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | целевой лимит длины ответа |
| `SHOW_LOGS` | `1` | включить пошаговые логи и таблицу тайминга |

<details>
<summary>Полный пример <code>.env</code></summary>

```env
# --- LLM provider ---
LLM_PROVIDER=ollama

# --- Ollama ---
OLLAMA_MODEL_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5:3b

# --- OpenRouter ---
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api

# --- SearxNG ---
SEARXNG_INSTANCES=http://localhost:8080
SEARXNG_DEFAULT_LANGUAGE=ru
SEARXNG_DEFAULT_CATEGORIES=general
REQUEST_TIMEOUT=10
SEARXNG_PARALLELISM=3

# --- Retrieval ---
RETRIEVAL_INITIAL_RESULTS_N=20
RETRIEVAL_STAGE1_TOP_K=5
RETRIEVAL_STAGE2_TOP_K=2
RETRIEVAL_DOC_MAX_CHARS=6000
RETRIEVAL_MIN_EXTRACTED_CHARS=300
SMART_PIPELINE_STAGE1_TOP_K=10
SMART_PIPELINE_FINAL_TOP_K=2

# --- LLM runtime ---
LLM_TIMEOUT=120
LLM_MAX_OUTPUT_TOKENS=2048

# --- Logging ---
SHOW_LOGS=1
LOG_LEVEL=INFO
LOG_CONSOLE_LEVEL=INFO
LOG_FILE=logs/results.log
```

</details>

## Логи и наблюдаемость

Если `SHOW_LOGS=1`, приложение:

- печатает шаги пайплайна в реальном времени;
- сохраняет подробный лог в `logs/results.log`;
- показывает после ответа таблицу с длительностью этапов.

Это в первую очередь полезно для сравнения конфигураций retrieval, оценки времени на каждом шаге и отладки поведения роутера.

## Структура репозитория

```text
config/
  settings.py               # конфигурация через .env

src/
  agent/                    # роутинг, прямой ответ, суммаризация
  interfaces/               # терминальный UI и MCP-клиент
  llm/                      # адаптеры direct / Ollama / OpenRouter
  mcp_server/               # MCP-сервер агента и внутренний search tool
  searxng/                  # HTTP-клиент, MCP-клиент и retrieval pipeline
  utils/                    # runtime и логирование
  main.py                   # точка входа

searxng-docker/
  docker-compose.yml        # локальный SearxNG
```

## Технологии

- **[Model Context Protocol](https://modelcontextprotocol.io)** для взаимодействия между клиентом, агентом и search tool
- **[SearxNG](https://docs.searxng.org)** как метапоисковый backend
- **[Ollama](https://ollama.com) / [OpenRouter](https://openrouter.ai)** как LLM-провайдеры
- **[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)** для первого реранка
- **[Sentence Transformers](https://www.sbert.net) / Jina reranker** для второго реранка
- **[Trafilatura](https://github.com/adbar/trafilatura)** для извлечения основного текста страниц
- **[Rich](https://github.com/textualize/rich)** для терминального интерфейса
