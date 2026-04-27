from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config.settings import settings
from src.agent.prompts import build_direct_answer_prompt, build_search_route_prompt, build_summary_prompt
from src.llm import create_llm
from src.searxng import ResponseProcessor, RetrievalPipeline, SearxngClient

logger = logging.getLogger("searxng_agent")
ProgressCallback = Callable[[str, float, float], Awaitable[None]]


class SearchAgent:

    def __init__(self):
        self.client = SearxngClient()
        self.processor = ResponseProcessor()
        self.retrieval_pipeline = RetrievalPipeline(self.client, self.processor)
        self.llm = create_llm()
        logger.info("Агент готов. Провайдер LLM: %s", settings.llm_provider)

    async def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        categories: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Базовый retrieval-пайплайн (SearxNG -> 2-stage rerank -> top-k)."""
        start_ts = time.perf_counter()
        try:
            processed = await self.retrieval_pipeline.run(
                query=query,
                language=language,
                categories=categories,
                final_limit=limit,
            )
            elapsed = (time.perf_counter() - start_ts) * 1000
            logger.info("Поиск завершен: %d результатов за %.0fмс", len(processed), elapsed)
            return processed
        except Exception as exc:
            logger.error("Ошибка SearxNG: %s", exc)
            return []

    async def smart_search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        categories: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """Второй пайплайн:
        запрос -> решение агента (искать/не искать) -> (поиск или прямой ответ) -> ответ пользователю.
        """
        total_start = time.perf_counter()

        await self._notify_progress(progress, "Агент решает, нужен ли веб-поиск", 5, 100)
        use_search = await self._decide_use_search(query)
        logger.info("Второй пайплайн | Решение агента | use_search=%s", use_search)

        if not use_search:
            await self._notify_progress(progress, "Формулирую ответ без веб-поиска", 35, 100)
            final_answer = await self._answer_without_search(query)
        else:
            pipeline_results = await self.retrieval_pipeline.run(
                query=query,
                language=language,
                categories=categories,
                # 2-й пайплайн: top-10 после первого реранка.
                stage1_top_k=settings.smart_pipeline_stage1_top_k,
                # 2-й пайплайн: top-2 после второго реранка.
                final_limit=min(max(limit, 1), settings.smart_pipeline_final_top_k),
                progress=progress,
            )
            top_results = self._dedupe_results(pipeline_results)
            if not top_results:
                await self._notify_progress(progress, "Поиск завершён, релевантных результатов не найдено", 100, 100)
                return "К сожалению, по вашему запросу ничего не найдено."
            await self._notify_progress(progress, "Суммаризирую найденные источники", 88, 100)
            final_answer = await self._summarize_results(query, top_results[: settings.smart_pipeline_final_top_k])

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info("Smart Search завершен за %.0fмс", total_ms)
        await self._notify_progress(progress, "Ответ готов", 100, 100)
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
            "searxng": getattr(self.client, "backend_label", getattr(self.client, "instance_url", "-")),
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

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

    async def _decide_use_search(self, user_input: str) -> bool:
        """Решает, нужен ли веб-поиск для ответа."""
        if not self.llm:
            logger.info("Второй пайплайн | LLM отключена, по умолчанию включаю поиск")
            return True

        prompt = build_search_route_prompt(user_input)
        logger.debug("Роутинг | Промпт (тип=%s, длина=%d): %s", type(prompt).__name__, len(prompt), prompt)
        try:
            response = await asyncio.to_thread(self.llm.complete, prompt)
            logger.debug("Роутинг | Ответ LLM (тип=%s, длина=%d): %r", type(response).__name__, len(response), response)

            if self._looks_like_error(response):
                logger.warning("Роутинг | LLM вернула ошибку, по умолчанию включаю поиск")
                return True

            return self._parse_search_route(response)
        except Exception as exc:
            logger.error("Роутинг | Ошибка определения режима: %s", exc)
            return True

    def _parse_search_route(self, text: str) -> bool:
        if not text:
            return True
        raw = text.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        payload = raw
        if start != -1 and end != -1 and end > start:
            payload = raw[start : end + 1]
        try:
            data = json.loads(payload)
        except Exception:
            return True

        if not isinstance(data, dict):
            return True

        use_search = data.get("use_search")
        if isinstance(use_search, bool):
            return use_search
        if isinstance(use_search, str):
            normalized = use_search.strip().lower()
            if normalized in {"true", "1", "yes", "да"}:
                return True
            if normalized in {"false", "0", "no", "нет"}:
                return False
        return True

    async def _answer_without_search(self, query: str) -> str:
        """Отвечает напрямую без веб-поиска."""
        if not self.llm:
            return "LLM отключена, поэтому отвечаю только через поиск. Включите /provider ollama или /provider openrouter."

        prompt = build_direct_answer_prompt(query)
        try:
            answer = (await self._complete_with_recovery(prompt)).strip()
        except Exception as exc:
            logger.error("Прямой ответ | Ошибка LLM: %s", exc)
            return "Не удалось сгенерировать ответ без поиска."

        if not answer or self._looks_like_error(answer):
            return "Не удалось сгенерировать ответ без поиска."
        return answer

    async def _summarize_results(self, original_query: str, results: List[Dict[str, object]]) -> str:
        if not results:
            return "Результатов не найдено."

        sources = self._extract_sources(results)

        if not self.llm:
            return self._format_fallback(results, sources)

        results_text = []
        for i, res in enumerate(results[:2], 1):
            title = res.get("title", "").strip()
            content = (res.get("display_content") or res.get("content") or "").strip()
            snippet = self._build_summary_snippet(original_query, content, max_chars=3200) if content else ""
            snippet = self._strip_links_for_llm(snippet)
            results_text.append(f"{i}. {title}\nКонтент: {snippet}")

        prompt = build_summary_prompt(original_query, "\n\n".join(results_text))
        logger.debug("Суммаризация | Промпт (тип=%s, длина=%d): %s", type(prompt).__name__, len(prompt), prompt)
        try:
            answer = (await self._complete_with_recovery(prompt)).strip()
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

    def _build_summary_snippet(self, query: str, content: str, max_chars: int = 3200) -> str:
        text = " ".join((content or "").split())
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        query_tokens = [t for t in self._tokenize(query) if len(t) >= 4]
        lowered = text.lower()
        hit_pos = -1
        for token in query_tokens:
            pos = lowered.find(token)
            if pos != -1:
                hit_pos = pos
                break

        if hit_pos == -1:
            return text[:max_chars]

        half_window = max_chars // 2
        start = max(0, hit_pos - half_window)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        return text[start:end]

    @staticmethod
    def _strip_links_for_llm(text: str) -> str:
        if not text:
            return ""
        # Markdown links: [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", text, flags=re.IGNORECASE)
        # Raw URLs
        text = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwww\.\S+\b", "", text, flags=re.IGNORECASE)
        return " ".join(text.split())

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

    @staticmethod
    async def _notify_progress(
        progress: Optional[ProgressCallback],
        message: str,
        current: float,
        total: float = 100,
    ) -> None:
        if progress is None:
            return
        await progress(message, current, total)

    async def _complete_with_recovery(self, prompt: str) -> str:
        base_tokens = settings.llm_max_output_tokens
        answer = await asyncio.to_thread(self.llm.complete, prompt, max_tokens=base_tokens)
        if self._looks_truncated_answer(answer):
            retry_tokens = min(max(base_tokens * 2, 3072), 8192)
            logger.warning(
                "Ответ LLM похож на усечённый. Повторяю генерацию с увеличенным лимитом: %d -> %d.",
                base_tokens,
                retry_tokens,
            )
            retry_answer = await asyncio.to_thread(self.llm.complete, prompt, max_tokens=retry_tokens)
            if len(retry_answer.strip()) >= len(answer.strip()):
                return retry_answer
        return answer

    @staticmethod
    def _looks_truncated_answer(answer: str) -> bool:
        text = (answer or "").strip()
        if len(text) < 80:
            return False
        if text.endswith((".", "!", "?", "…", "\"", "'", "»", ")", "]")):
            return False
        return True
