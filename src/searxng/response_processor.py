from bs4 import BeautifulSoup
from typing import Dict, List

class ResponseProcessor:
    """Очищает HTML и приводит результаты SearxNG к единому формату."""

    @staticmethod
    def clean_html(text: str) -> str:
        """Удаляет HTML-теги и лишние пробелы."""
        if not text:
            return ""
        
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.get_text().split())

    @staticmethod
    def process_results(raw_results: List[Dict]) -> List[Dict[str, object]]:
        """Преобразует сырые результаты SearxNG в список словарей."""
        processed: List[Dict[str, object]] = []
        
        for item in raw_results:
            score = item.get("score")

            processed.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("engine", "unknown"),
                    "content": ResponseProcessor.clean_html(item.get("content", "")),
                    "score": float(score) if isinstance(score, (int, float)) else None,
                }
            )
        
        return processed
