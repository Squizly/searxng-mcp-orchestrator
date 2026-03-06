import logging
import re
from bs4 import BeautifulSoup

logger = logging.getLogger("ResponseProcessor")

class ResponseProcessor:
    """
    Helper class to parse and clean HTML from SearxNG JSON responses.
    Converts raw web data into a clean, LLM-friendly text format.
    """

    @staticmethod
    def clean_html(text: str) -> str:
        """
        Removes HTML tags and normalizes spaces using BeautifulSoup.
        """
        if not text: 
            return ""
        
        try:
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
            return re.sub(r'\s+', ' ', clean_text)
        except Exception as e:
            logger.warning(f"Failed to clean HTML text: {repr(e)}. Returning raw text.")
            return text

    @classmethod
    def process_results(cls, results: list, limit: int = 5) -> str:
        """
        Processes a list of raw search results and limits the output.
        """
        if not results:
            logger.warning("No results found in the provided payload.")
            return "Результаты не найдены. Попробуй изменить поисковый запрос."
        
        logger.info(f"Processing {len(results)} search results (applying limit: {limit}).")
        
        processed = []
        for r in results[:limit]:
            title = cls.clean_html(r.get("title", "Без заголовка"))
            content = cls.clean_html(r.get("content", ""))
            url = r.get("url", "Нет URL")
            engines = ", ".join(r.get("engines", ["unknown"]))
            
            processed.append(
                f"📌 Заголовок: {title}\n"
                f"🔗 URL: {url}\n"
                f"⚙️ Источник: {engines}\n"
                f"📄 Текст: {content}\n"
            )
            
        logger.info("Successfully cleaned and formatted search results.")
        return "\n---\n".join(processed)