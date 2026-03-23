import logging
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings

logger = logging.getLogger("searxng_agent")

class SearxngClient:
    """Клиент для запроса SearxNG."""

    def __init__(self, instance_url: Optional[str] = None):
        if instance_url:
            self.instance_url = instance_url.rstrip("/")
        else:
            instances = settings.searxng_instances_list
            if not instances:
                raise ValueError("Не задан ни один инстанс SearxNG")
            self.instance_url = instances[0].rstrip("/")

        self.timeout = settings.request_timeout

    def search(
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
        logger.info("Запрос к SearxNG: %s, параметры=%s", url, params)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params)

            response.raise_for_status()
            data = response.json()

            logger.debug("Ответ SearxNG: %s", data)
        except Exception as exc:
            logger.error("Ошибка запроса к SearxNG: %s", exc)
            raise

        raw_results = data.get("results", [])[:limit]
        logger.info("Результатов от SearxNG: %d", len(raw_results))
        return raw_results
