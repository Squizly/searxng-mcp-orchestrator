from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Конфигурация приложения."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    llm_provider: Literal["direct", "ollama", "openrouter"] = Field(default="ollama")

    ollama_model_url: str = Field(default="http://localhost:11434")
    ollama_model_name: str = Field(default="llama3.2")

    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_model: str = Field(default="openai/gpt-4o-mini")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api")

    searxng_instances: str = Field(default="http://localhost:8080")
    searxng_default_language: str = Field(default="ru")
    searxng_default_categories: str = Field(default="general")
    request_timeout: int = Field(default=10, ge=1)
    searxng_parallelism: int = Field(default=3, ge=1, le=10)
    llm_timeout: int = Field(default=120, ge=1)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_file: Path = Field(default=Path("logs/results.log"))

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str):
        """Нормализует алиасы провайдера (local -> ollama)."""
        if isinstance(value, str) and value.strip().lower() == "local":
            return "ollama"
        return value

    @model_validator(mode="after")
    def check_api_keys(self):
        """Проверяет, что при выборе провайдера ключ задан."""
        provider = self.llm_provider
        if provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("Для провайдера openrouter необходимо указать OPENROUTER_API_KEY в .env")
        return self

    @property
    def searxng_instances_list(self) -> List[str]:
        """Возвращает список URL инстансов SearxNG."""
        return [url.strip() for url in self.searxng_instances.split(",") if url.strip()]

    def model_post_init(self, __context):
        """Создаёт директорию для логов, если её нет."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
