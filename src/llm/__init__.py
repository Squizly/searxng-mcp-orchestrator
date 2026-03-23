import logging

from config.settings import settings
from .base import LLMInterface
from .local import LocalLLM
from .openrouter import OpenRouterLLM

logger = logging.getLogger("searxng_agent")

def create_llm(provider_override: str | None = None) -> LLMInterface | None:
    provider = (provider_override or settings.llm_provider).lower()

    if provider in {"ollama", "local"}:
        logger.info("Инициализация локальной LLM (Ollama)")
        return LocalLLM()
    
    if provider == "openrouter":
        logger.info("Инициализация LLM OpenRouter")
        return OpenRouterLLM()
    
    if provider == "direct":
        logger.info("LLM отключена (LLM_PROVIDER=direct)")
        return None

    logger.warning("Неизвестный провайдер LLM: %s. Работаем без LLM.", provider)
    return None
