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

        logger.info(f"Агент инициализирован локально (модель: {self.model_name})")

    def complete(self, prompt: str, **kwargs) -> str:
        logger.debug("Запрос к LLM (первые 50 символов): %s", prompt[:50])
        max_tokens = int(kwargs.get("max_tokens", settings.llm_max_output_tokens))
        timeout = kwargs.get("timeout", self.timeout)
        try:
            answer, done_reason = self._request_completion(prompt, max_tokens=max_tokens, timeout=timeout)
            if done_reason == "length" and max_tokens < 8192:
                retry_tokens = min(max_tokens * 2, 8192)
                logger.warning(
                    "Ollama остановила генерацию по лимиту длины. Повторяю запрос с увеличенным лимитом: %d -> %d.",
                    max_tokens,
                    retry_tokens,
                )
                retry_answer, retry_reason = self._request_completion(prompt, max_tokens=retry_tokens, timeout=timeout)
                if len(retry_answer.strip()) >= len(answer.strip()):
                    answer, done_reason = retry_answer, retry_reason

            logger.debug("Ответ LLM (первые 200 символов): %s", answer[:200])
            return answer
        except httpx.TimeoutException as exc:
            logger.warning("Таймаут запроса к LLM: %s. Увеличьте LLM_TIMEOUT в .env при необходимости.", exc)
            return "[Ошибка LLM: таймаут]"
        except Exception as exc:
            logger.error("Ошибка запроса к LLM: %s", exc)
            return f"[Ошибка LLM: {exc}]"

    def _request_completion(self, prompt: str, max_tokens: int, timeout: int) -> tuple[str, str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }
        timeout_cfg = httpx.Timeout(timeout, connect=min(10, timeout))

        with httpx.Client(timeout=timeout_cfg) as client:
            response = client.post(f"{self.model_url}/api/generate", json=payload)

        response.raise_for_status()
        result = response.json()
        answer = str(result.get("response", ""))
        done_reason = str(result.get("done_reason") or "").strip().lower()
        return answer, done_reason
