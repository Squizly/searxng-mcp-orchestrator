from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from config.settings import settings
from src.agent.prompts import build_decision_prompt, build_summary_prompt
from src.llm import create_llm
from src.searxng import ResponseProcessor, SearxngClient

logger = logging.getLogger("searxng_agent")


class SearchAgent:
    """Поисковый агент: формирует запросы, ищет и суммаризирует результаты."""

    def __init__(self, instance_url: Optional[str] = None):
        self.client = SearxngClient(instance_url)
        self.processor = ResponseProcessor()
        self.llm = create_llm()
        logger.info("Агент готов. Провайдер LLM: %s", settings.llm_provider)

    def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        categories: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        start_ts = time.perf_counter()
        logger.info("Прямой поиск | Старт | Запрос=%r", query)

        try:
            raw = self.client.search(query, language=language, categories=categories, limit=limit)
        except Exception as exc:
            logger.error("Прямой поиск | Ошибка SearxNG | %s", exc)
            return []

        processed = self.processor.process_results(raw)
        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        logger.info("Прямой поиск | Готово | Результатов=%d | Время=%.0fмс", len(processed), elapsed_ms)
        return processed[:limit]

    def smart_search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        categories: Optional[str] = None,
    ) -> str:
        total_start = time.perf_counter()
        logger.info("Умный поиск | Старт | Запрос=%r", query)

        logger.info("Умный поиск | Этап 1/3 | Решение LLM о доп. запросах")
        search_queries = self._decide_queries(query)

        stage2_start = time.perf_counter()
        logger.info("Умный поиск | Этап 2/3 | Поиск в SearxNG (параллельно)")
        all_results: List[Dict[str, object]] = []
        errors: List[str] = []
        max_workers = min(len(search_queries), settings.searxng_parallelism)
        logger.info(
            "Умный поиск | Этап 2/3 | Запросов=%d | Параллельность=%d",
            len(search_queries),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for q in search_queries:
                logger.info("Умный поиск | Этап 2/3 | Отправка запроса в SearxNG | %r", q)
                future = executor.submit(
                    self.client.search,
                    q,
                    language=language,
                    categories=categories,
                    limit=limit,
                )
                future_map[future] = q

            for future in as_completed(future_map):
                q = future_map[future]
                try:
                    raw = future.result()
                    logger.info(
                        "Умный поиск | Этап 2/3 | Получен ответ SearxNG | Запрос=%r | Тип=%s",
                        q,
                        type(raw).__name__,
                    )
                    logger.debug("SearxNG JSON | Запрос=%r | Данные=%s", q, raw)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    errors.append(err)
                    logger.error("Умный поиск | Этап 2/3 | Ошибка SearxNG | Запрос=%r | %s", q, err)
                    continue

                processed = self.processor.process_results(raw)
                logger.info(
                    "Умный поиск | Этап 2/3 | Обработка ответа | Запрос=%r | Результатов=%d",
                    q,
                    len(processed),
                )
                all_results.extend(processed)
                logger.debug("SearxNG | Запрос=%r | Найдено=%d", q, len(processed))

        deduped = self._dedupe_results(all_results)
        stage2_ms = (time.perf_counter() - stage2_start) * 1000.0
        logger.info(
            "Умный поиск | Этап 2/3 | Итого результатов=%d | После дедупликации=%d | Время=%.0fмс",
            len(all_results),
            len(deduped),
            stage2_ms,
        )

        if not deduped and errors:
            return (
                "Ошибка поиска: не удалось связаться с SearxNG. "
                "Проверьте, запущен ли экземпляр SearxNG и корректен ли URL."
            )

        logger.info("Умный поиск | Этап 2/3 | Реранкинг результатов")
        reranked = self._rerank_results(query, deduped)
        top_results = reranked[: max(limit, 1)]
        logger.info("Умный поиск | Этап 2/3 | Реранкинг готов | Топ=%d из %d", len(top_results), len(reranked))
        for idx, res in enumerate(top_results, 1):
            logger.info(
                "Умный поиск | Этап 2/3 | Топ %d | score=%.3f | title=%r | url=%r",
                idx,
                float(res.get("_rank_score", 0.0) or 0.0),
                (res.get("title") or "")[:120],
                res.get("url"),
            )

        stage3_start = time.perf_counter()
        logger.info("Умный поиск | Этап 3/3 | Суммаризация")
        final_answer = self._summarize_results(query, top_results)
        stage3_ms = (time.perf_counter() - stage3_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        logger.info("Умный поиск | Этап 3/3 | Готово | Время=%.0fмс", stage3_ms)
        logger.info("Умный поиск | Завершено | Время=%.0fмс", total_ms)
        return final_answer

    def get_status(self) -> Dict[str, str]:
        provider = self._normalize_provider(settings.llm_provider)
        model = "-"
        if provider in {"ollama", "local"}:
            model = settings.ollama_model_name
        elif provider == "openrouter":
            model = settings.openrouter_model
        return {
            "provider": provider,
            "model": model,
            "searxng": getattr(self.client, "instance_url", "-"),
        }

    def set_provider(self, provider: str, model: Optional[str] = None) -> str:
        provider = self._normalize_provider(provider)
        if provider not in {"direct", "ollama", "openrouter"}:
            return "Доступные провайдеры: direct, ollama, openrouter."

        current = self._normalize_provider(settings.llm_provider)
        if provider == current and not model:
            status = self.get_status()
            return f"Провайдер уже выбран: {status['provider']} (модель: {status['model']})"

        if provider == "openrouter" and not settings.openrouter_api_key:
            return "OPENROUTER_API_KEY не задан в .env"

        old_provider = settings.llm_provider
        old_llm = self.llm
        old_ollama_model = settings.ollama_model_name
        old_openrouter_model = settings.openrouter_model

        settings.llm_provider = provider
        if model:
            if provider == "ollama":
                settings.ollama_model_name = model
            elif provider == "openrouter":
                settings.openrouter_model = model

        try:
            self.llm = create_llm(provider_override=provider)
            current_model = self.get_status()["model"]
            logger.info("LLM переключена: %s, модель=%s", provider, current_model)
            if provider == "direct":
                return "Режим direct: LLM отключена."
            return f"LLM переключена на {provider}. Модель: {current_model}"
        except Exception as exc:
            settings.llm_provider = old_provider
            settings.ollama_model_name = old_ollama_model
            settings.openrouter_model = old_openrouter_model
            self.llm = old_llm
            logger.error("Ошибка переключения LLM: %s", exc)
            return f"Не удалось переключить LLM: {exc}"

    def set_model(self, model: str) -> str:
        provider = self._normalize_provider(settings.llm_provider)
        if provider == "direct":
            return "В режиме direct модель не используется."
        if provider not in {"ollama", "openrouter"}:
            return "Текущий провайдер не поддерживает смену модели."

        model_lower = model.lower().strip()
        if provider == "ollama" and "openrouter/" in model_lower:
            return "Сейчас активна Ollama. Для OpenRouter используйте: /provider openrouter"
        if provider == "openrouter" and "/" not in model_lower:
            return "Похоже, это не OpenRouter-модель. Для Ollama используйте: /provider ollama"

        old_llm = self.llm
        old_ollama_model = settings.ollama_model_name
        old_openrouter_model = settings.openrouter_model

        if provider == "ollama":
            settings.ollama_model_name = model
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                return "OPENROUTER_API_KEY не задан в .env"
            settings.openrouter_model = model

        try:
            self.llm = create_llm(provider_override=provider)
            logger.info("Модель LLM обновлена: %s, модель=%s", provider, model)
            return f"Модель обновлена: {provider} / {model}"
        except Exception as exc:
            settings.ollama_model_name = old_ollama_model
            settings.openrouter_model = old_openrouter_model
            self.llm = old_llm
            logger.error("Ошибка смены модели: %s", exc)
            return f"Не удалось сменить модель: {exc}"

    def _decide_queries(self, user_input: str) -> List[str]:
        """Решает, нужны ли доп. запросы. Если нужны — генерирует до 3 вариантов."""
        if not self.llm:
            logger.info("Умный поиск | Этап 1/3 | LLM отключена, использую исходный запрос")
            return [user_input]

        prompt = build_decision_prompt(user_input)
        logger.debug("LLM решение | Промпт (тип=%s, длина=%d): %s", type(prompt).__name__, len(prompt), prompt)
        try:
            response = self.llm.complete(prompt)
            logger.debug("LLM решение | Ответ (тип=%s, длина=%d): %r", type(response).__name__, len(response), response)

            if self._looks_like_error(response):
                logger.warning("Умный поиск | Этап 1/3 | LLM вернула ошибку, использую исходный запрос")
                return [user_input]

            decision = self._parse_decision(response)
            if not decision["use_extra_queries"]:
                logger.info("Умный поиск | Этап 1/3 | Доп. запросы не нужны")
                return [user_input]

            queries = decision["queries"]
            if len(queries) < 3:
                logger.warning(
                    "Умный поиск | Этап 1/3 | LLM вернула меньше 3 запросов (%d), дополняю исходным",
                    len(queries),
                )
                queries = (queries + [user_input] * 3)[:3]

            logger.info("Умный поиск | Этап 1/3 | Сформированы доп. запросы: %s", queries)
            return queries[:3]
        except Exception as exc:
            logger.error("Умный поиск | Этап 1/3 | Ошибка генерации запросов: %s", exc)
            return [user_input]

    def _parse_decision(self, text: str) -> Dict[str, object]:
        if not text:
            return {"use_extra_queries": False, "queries": []}
        raw = text.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        payload = raw
        if start != -1 and end != -1 and end > start:
            payload = raw[start : end + 1]
        try:
            data = json.loads(payload)
        except Exception:
            return {"use_extra_queries": False, "queries": []}

        use_extra = bool(data.get("use_extra_queries"))
        queries = data.get("queries") or []
        if isinstance(queries, list):
            queries = [str(item).strip() for item in queries if str(item).strip()]
        else:
            queries = []
        return {"use_extra_queries": use_extra, "queries": queries}

    def _summarize_results(self, original_query: str, results: List[Dict[str, object]]) -> str:
        if not results:
            return "Результатов не найдено."

        sources = self._extract_sources(results)

        if not self.llm:
            return self._format_fallback(results, sources)

        results_text = []
        for i, res in enumerate(results[:10], 1):
            title = res.get("title", "").strip()
            url = res.get("url", "").strip()
            snippet = res.get("content", "").strip().replace("\n", " ")
            snippet = snippet[:200] if snippet else ""
            results_text.append(f"{i}. {title}\nURL: {url}\nФрагмент: {snippet}")

        prompt = build_summary_prompt(original_query, "\n\n".join(results_text))
        logger.debug("Суммаризация | Промпт (тип=%s, длина=%d): %s", type(prompt).__name__, len(prompt), prompt)
        try:
            answer = self.llm.complete(prompt).strip()
            logger.debug("Суммаризация | Ответ LLM (длина=%d): %s", len(answer), answer)
        except Exception as exc:
            logger.error("Суммаризация | Ошибка LLM: %s", exc)
            answer = ""

        if not answer or self._looks_like_error(answer):
            return self._format_fallback(results, sources)

        if self._is_cyrillic_query(original_query):
            if self._contains_cjk(answer) or not self._looks_russian(answer):
                logger.warning("Суммаризация | Ответ не на русском, возвращаю краткие результаты")
                return self._format_fallback(results, sources)
            if not self._is_answer_relevant(answer, original_query, results):
                logger.warning("Суммаризация | Ответ, вероятно, не по теме, возвращаю краткие результаты")
                return self._format_fallback(results, sources)

        return f"{answer}\n\nИсточники:\n{self._format_sources(sources)}"

    @staticmethod
    def _format_sources(urls: List[str]) -> str:
        return "\n".join([f"[{i}] {url}" for i, url in enumerate(urls, 1)])

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        if not text:
            return True
        lowered = text.lower()
        return (
            lowered.startswith("[ошибка")
            or "ошибка openrouter" in lowered
            or "ошибка llm" in lowered
            or "client error" in lowered
            or "not found" in lowered
            or "таймаут" in lowered
        )

    @staticmethod
    def _is_cyrillic_query(text: str) -> bool:
        for ch in text:
            if "а" <= ch.lower() <= "я" or ch.lower() == "ё":
                return True
        return False

    @staticmethod
    def _looks_russian(text: str) -> bool:
        letters = 0
        cyr = 0
        for ch in text:
            if ch.isalpha():
                letters += 1
                if "а" <= ch.lower() <= "я" or ch.lower() == "ё":
                    cyr += 1
        if letters == 0:
            return False
        return (cyr / letters) >= 0.6

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        for ch in text:
            code = ord(ch)
            if (
                0x4E00 <= code <= 0x9FFF
                or 0x3400 <= code <= 0x4DBF
                or 0x3040 <= code <= 0x30FF
                or 0xAC00 <= code <= 0xD7AF
            ):
                return True
        return False

    @staticmethod
    def _has_query_token(answer: str, query: str) -> bool:
        tokens = [
            t.strip(".,!?;:\"'()[]{}").lower()
            for t in query.split()
            if len(t.strip(".,!?;:\"'()[]{}")) >= 4
        ]
        answer_lower = answer.lower()
        return any(t in answer_lower for t in tokens)

    def _is_answer_relevant(
        self,
        answer: str,
        query: str,
        results: List[Dict[str, object]],
    ) -> bool:
        query_tokens = self._tokenize(query)
        if len(query_tokens) < 2:
            return True
        if self._has_query_token(answer, query):
            return True

        answer_tokens = set(self._tokenize(answer))
        if not answer_tokens:
            return False

        context_tokens = set()
        for res in results[:5]:
            title = res.get("title", "") or ""
            snippet = res.get("content", "") or ""
            context_tokens.update(self._tokenize(f"{title} {snippet}"))
        if not context_tokens:
            return False

        overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
        return overlap >= 0.15

    def _format_fallback(self, results: List[Dict[str, object]], sources: List[str]) -> str:
        lines = ["Лучшие результаты:"]
        for i, res in enumerate(results[:5], 1):
            title = res.get("title", "").strip()
            url = res.get("url", "").strip()
            snippet = res.get("content", "").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"{i}. {title}")
            lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")
        lines.append("")
        lines.append("Источники:")
        lines.append(self._format_sources(sources))
        return "\n".join(lines)

    def _rerank_results(self, query: str, results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not results:
            return results

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results

        ranked = []
        for index, res in enumerate(results):
            title = res.get("title", "") or ""
            snippet = res.get("content", "") or ""
            title_tokens = set(self._tokenize(title))
            snippet_tokens = set(self._tokenize(snippet))

            score = 0.0
            for token in query_tokens:
                if token in title_tokens:
                    score += 2.0
                if token in snippet_tokens:
                    score += 1.0

            searx_score = res.get("score")
            if isinstance(searx_score, (int, float)):
                score += min(float(searx_score), 10.0) * 0.1

            if snippet:
                score += min(len(snippet) / 200.0, 1.0) * 0.3

            res["_rank_score"] = score
            ranked.append((score, index, res))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())
        stopwords = {
            "и",
            "в",
            "во",
            "на",
            "по",
            "для",
            "с",
            "со",
            "к",
            "ко",
            "о",
            "об",
            "от",
            "до",
            "это",
            "как",
            "что",
            "где",
            "когда",
            "или",
            "ли",
            "the",
            "and",
            "of",
            "to",
            "in",
            "for",
            "on",
        }
        return [t for t in tokens if len(t) >= 3 and t not in stopwords]

    @staticmethod
    def _extract_sources(results: List[Dict[str, object]]) -> List[str]:
        seen = set()
        sources: List[str] = []
        for res in results:
            url = res.get("url", "").strip()
            if url and url not in seen:
                seen.add(url)
                sources.append(url)
        return sources

    @staticmethod
    def _dedupe_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen = set()
        deduped: List[Dict[str, object]] = []
        for res in results:
            url = res.get("url", "").strip()
            key = url or json.dumps(res, ensure_ascii=True, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(res)
        return deduped

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        value = (provider or "").strip().lower()
        if value == "local":
            return "ollama"
        return value
