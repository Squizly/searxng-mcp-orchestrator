from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """Абстрактный интерфейс для работы с языковыми моделями."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Отправить промпт и получить ответ."""
        raise NotImplementedError
