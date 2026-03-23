import logging
from typing import Dict, Optional

import httpx

from config.settings import settings
from .base import LLMInterface

logger = logging.getLogger("searxng_agent")

class LocalLLM(LLMInterface):
    """Клиент для Ollama."""

    def __init__(self, model_url: Optional[str] = None, model_name: Optional[str] = None):
        self.model_url = (model_url or settings.ollama_model_url).rstrip("/")
        self.model_name = model_name or settings.ollama_model_name
        self.timeout = settings.llm_timeout

        logger.info(
            "Локальная LLM (Ollama) готова: %s, модель=%s",
            self.model_url,
            self.model_name,
        )

    def complete(self, prompt: str, **kwargs) -> str:
        logger.debug("Запрос к LLM (первые 50 символов): %s", prompt[:50])
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        timeout = kwargs.get("timeout", self.timeout)
        timeout_cfg = httpx.Timeout(timeout, connect=min(10, timeout))

        try:
            with httpx.Client(timeout=timeout_cfg) as client:
                response = client.post(f"{self.model_url}/api/generate", json=payload)

            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "")
            
            logger.debug("Ответ LLM (первые 200 символов): %s", answer[:200])
            return answer
        except httpx.TimeoutException as exc:
            logger.warning("Таймаут запроса к LLM: %s. Увеличьте LLM_TIMEOUT в .env при необходимости.", exc)
            return "[Ошибка LLM: таймаут]"
        except Exception as exc:
            logger.error("Ошибка запроса к LLM: %s", exc)
            return f"[Ошибка LLM: {exc}]"
