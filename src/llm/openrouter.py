import logging
from typing import Dict, List, Optional

import httpx

from config.settings import settings
from .base import LLMInterface

logger = logging.getLogger("searxng_agent")


class OpenRouterLLM(LLMInterface):
    """Клиент OpenRouter через OpenAI-совместимый API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("Не задан OPENROUTER_API_KEY")
        
        self.model_name = model_name or settings.openrouter_model
        base = (base_url or settings.openrouter_base_url).rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        
        self.base_url = base
        self.timeout = settings.llm_timeout

        logger.info("OpenRouter LLM готова: модель=%s", self.model_name)

    def complete(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._chat_completion(messages, **kwargs)

    def _chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        max_tokens = int(kwargs.get("max_tokens", settings.llm_max_output_tokens))
        timeout_value = kwargs.get("timeout", self.timeout)
        try:
            text, finish_reason = self._request_completion(
                messages=messages,
                max_tokens=max_tokens,
                timeout_value=timeout_value,
                **kwargs,
            )
            if finish_reason == "length" and max_tokens < 8192:
                retry_tokens = min(max_tokens * 2, 8192)
                logger.warning(
                    "OpenRouter остановила генерацию по лимиту длины. Повторяю запрос с увеличенным лимитом: %d -> %d.",
                    max_tokens,
                    retry_tokens,
                )
                retry_text, retry_finish_reason = self._request_completion(
                    messages=messages,
                    max_tokens=retry_tokens,
                    timeout_value=timeout_value,
                    **kwargs,
                )
                if len(retry_text.strip()) >= len(text.strip()):
                    text, finish_reason = retry_text, retry_finish_reason
        except httpx.TimeoutException as exc:
            logger.warning("Таймаут запроса к OpenRouter: %s. Увеличьте LLM_TIMEOUT в .env при необходимости.", exc)
            return "[Ошибка LLM: таймаут]"
        except Exception as exc:
            logger.error("Ошибка запроса к OpenRouter: %s", exc)
            return f"[Ошибка OpenRouter: {exc}]"
        
        logger.debug("Ответ OpenRouter (первые 200 символов): %s", text[:200])
        
        return text

    def _request_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        timeout_value: int,
        **kwargs,
    ) -> tuple[str, str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        for key in ("temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs:
                payload[key] = kwargs[key]

        timeout = httpx.Timeout(timeout_value, connect=min(10, timeout_value))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code == 401:
            msg = "Неверный API-ключ или нет доступа."
            logger.error("Ошибка OpenRouter 401: %s", msg)
            raise RuntimeError(f"[Ошибка OpenRouter: {msg}]")

        if response.status_code == 403:
            msg = "Доступ запрещён. Проверьте права ключа."
            logger.error("Ошибка OpenRouter 403: %s", msg)
            raise RuntimeError(f"[Ошибка OpenRouter: {msg}]")

        if response.status_code >= 400:
            logger.error("Ошибка OpenRouter %s: %s", response.status_code, response.text)
            raise RuntimeError(f"[Ошибка OpenRouter: HTTP {response.status_code}]")

        data = response.json()
        text = self._extract_text(data)
        finish_reason = self._extract_finish_reason(data)
        if not text:
            logger.error("OpenRouter вернула пустой ответ: %s", data)
            raise RuntimeError("[Ошибка OpenRouter: пустой ответ]")
        return text, finish_reason


    @staticmethod
    def _extract_text(data: Dict[str, object]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        
        message = first.get("message") or {}
        if not isinstance(message, dict):
            return ""
        
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        
        return ""

    @staticmethod
    def _extract_finish_reason(data: Dict[str, object]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        finish_reason = first.get("finish_reason")
        return str(finish_reason or "").strip().lower()
