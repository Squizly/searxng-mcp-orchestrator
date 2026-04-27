from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings

logger = logging.getLogger("searxng_agent")


class SearxngHttpClient:
    """Прямой HTTP-клиент для внутреннего MCP search-server."""

    def __init__(self, instance_url: Optional[str] = None):
        if instance_url:
            self.instance_url = instance_url.rstrip("/")
        else:
            instances = settings.searxng_instances_list
            if not instances:
                raise ValueError("Не задан ни один инстанс SearxNG")
            self.instance_url = instances[0].rstrip("/")

        self.timeout = settings.request_timeout
        logger.info("HTTP-клиент SearxNG инициализирован: %s", self.instance_url)

    async def search(
        self,
        query: str,
        language: Optional[str] = None,
        categories: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "format": "json",
            "language": language or settings.searxng_default_language,
            "categories": categories or settings.searxng_default_categories,
        }
        url = f"{self.instance_url}/search"
        timeout = httpx.Timeout(self.timeout, connect=min(10, self.timeout))

        logger.info("SearxNG HTTP | query=%r | url=%s", query, url)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            logger.error("SearxNG HTTP | Не удалось подключиться к %s", self.instance_url)
            raise RuntimeError(f"Не удалось подключиться к SearxNG: {self.instance_url}") from exc
        except httpx.TimeoutException as exc:
            logger.error("SearxNG HTTP | Таймаут запроса: %s", exc)
            raise RuntimeError("Таймаут запроса к SearxNG") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            logger.error("SearxNG HTTP | HTTP %s: %s", status_code, exc.response.text)
            raise RuntimeError(f"SearxNG вернул HTTP {status_code}") from exc
        except Exception as exc:
            logger.error("SearxNG HTTP | Ошибка запроса %r: %s", query, exc)
            raise RuntimeError(f"Ошибка запроса к SearxNG: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("SearxNG HTTP | Некорректный JSON в ответе")
            raise RuntimeError("SearxNG вернул некорректный JSON") from exc

        if data.get("error"):
            logger.warning("SearxNG HTTP | Ошибка в теле ответа: %s", data["error"])
            raise RuntimeError(str(data["error"]))

        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise RuntimeError("SearxNG вернул неожиданный формат results")

        final_results = raw_results[:limit]
        logger.info(
            "SearxNG HTTP | query=%r | found=%d | returning=%d",
            query,
            len(raw_results),
            len(final_results),
        )
        return final_results
