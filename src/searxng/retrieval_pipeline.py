from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config.settings import settings
from src.searxng.client import SearxngClient
from src.searxng.response_processor import ResponseProcessor
from src.utils.runtime import configure_runtime_environment

logger = logging.getLogger("searxng_agent")
ProgressCallback = Callable[[str, float, float], Awaitable[None]]


class RetrievalPipeline:
    """Базовый retrieval-пайплайн:
    SearxNG -> FlashRank -> top-5 -> requests/trafilatura(markdown) -> Jina reranker -> top-2.
    """

    FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"
    JINA_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

    def __init__(self, client: SearxngClient, processor: ResponseProcessor):
        self.client = client
        self.processor = processor
        self._flashrank, self._flashrank_request_cls = self._init_flashrank()
        self._jina_reranker = self._init_jina_reranker()

    async def run(
        self,
        query: str,
        language: Optional[str] = None,
        categories: Optional[str] = None,
        final_limit: Optional[int] = None,
        stage1_top_k: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """Запускает полный retrieval-пайплайн и возвращает итоговые top-k результаты."""
        pipeline_start = time.perf_counter()
        initial_limit = settings.retrieval_initial_results_n
        effective_stage1_top_k = stage1_top_k or settings.retrieval_stage1_top_k

        await self._notify_progress(progress, "Ищу источники в SearxNG", 18, 100)
        raw_results = await self.client.search(
            query=query,
            language=language,
            categories=categories,
            limit=initial_limit,
        )
        processed_results = self.processor.process_results(raw_results)
        if not processed_results:
            logger.info("Retrieval pipeline | query=%r | пустой ответ от SearxNG", query)
            return []

        await self._notify_progress(progress, "Первично ранжирую найденные результаты", 40, 100)
        stage1_ranked = await asyncio.to_thread(self._flashrank_rerank, query, processed_results)
        stage1_top = stage1_ranked[: max(1, effective_stage1_top_k)]
        if not stage1_top:
            return []

        logger.info(
            "Retrieval pipeline | query=%r | stage1=%d -> top=%d",
            query,
            len(processed_results),
            len(stage1_top),
        )

        await self._notify_progress(progress, "Загружаю страницы и извлекаю текст", 58, 100)
        downloaded_markdown = await self._download_and_extract_markdown(stage1_top)
        extracted = [
            item
            for item in downloaded_markdown
            if item.get("content_source") == "trafilatura"
            and len((item.get("content") or "").strip()) >= settings.retrieval_min_extracted_chars
        ]
        logger.info(
            "Retrieval pipeline | query=%r | fulltext_extracted=%d/%d",
            query,
            len(extracted),
            len(downloaded_markdown),
        )

        stage2_input = extracted if extracted else downloaded_markdown
        if not extracted:
            logger.warning(
                "Retrieval pipeline | query=%r | не удалось извлечь fulltext, fallback на сниппеты SearxNG",
                query,
            )

        await self._notify_progress(progress, "Делаю финальный реранк по полному тексту", 74, 100)
        stage2_ranked = await asyncio.to_thread(self._jina_rerank, query, stage2_input)

        if final_limit is None:
            top_k = settings.retrieval_stage2_top_k
        else:
            top_k = max(1, min(final_limit, settings.retrieval_stage2_top_k))

        final_results = stage2_ranked[:top_k]
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "Retrieval pipeline | query=%r | stage2=%d -> final=%d | %.0fms",
            query,
            len(stage2_ranked),
            len(final_results),
            total_ms,
        )
        return final_results

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

    def _init_flashrank(self):
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError as exc:
            raise RuntimeError(
                "Не найдена зависимость flashrank. Установите зависимости: pip install -r requirements.txt"
            ) from exc

        try:
            ranker = Ranker(model_name=self.FLASHRANK_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить FlashRank-модель {self.FLASHRANK_MODEL}: {exc}"
            ) from exc

        logger.info("FlashRank инициализирован: %s", self.FLASHRANK_MODEL)
        return ranker, RerankRequest

    def _init_jina_reranker(self):
        configure_runtime_environment()

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Не найдена зависимость sentence-transformers. "
                "Установите зависимости: pip install -r requirements.txt"
            ) from exc

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Для sentence-transformers требуется torch. "
                "Установите зависимости: pip install -r requirements.txt"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = CrossEncoder(
                self.JINA_MODEL,
                trust_remote_code=True,
                device=device,
                automodel_args={"dtype": torch.float32},
            )
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                inner_model.float()
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить Jina reranker {self.JINA_MODEL}: {exc}"
            ) from exc

        logger.info("Jina reranker инициализирован: %s | device=%s | dtype=float32", self.JINA_MODEL, device)
        return model

    def _flashrank_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        passages = []
        id_to_result: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(results):
            pid = str(index)
            text = self._compose_stage1_text(item)
            passages.append({"id": pid, "text": text})
            id_to_result[pid] = dict(item)

        request = self._flashrank_request_cls(query=query, passages=passages)
        reranked = self._flashrank.rerank(request)

        flashrank_scores: Dict[str, float] = {}
        for ranked_item in reranked:
            pid = str(ranked_item.get("id", ""))
            if pid not in id_to_result:
                continue
            flashrank_scores[pid] = self._to_float(ranked_item.get("score"))

        # Stage 1 formula (как согласовано):
        # combined = 0.8 * flashrank_score + 0.2 * score_searxng
        scored_docs = []
        for pid, base_doc in id_to_result.items():
            base = dict(base_doc)
            flashrank_score = flashrank_scores.get(pid, 0.0)
            searxng_score = self._to_float(base.get("score"))
            stage1_score = 0.8 * flashrank_score + 0.2 * searxng_score

            base["stage1_flashrank_score"] = flashrank_score
            base["stage1_searxng_score"] = searxng_score
            base["stage1_score"] = stage1_score
            scored_docs.append(base)

        scored_docs.sort(key=lambda item: item.get("stage1_score", 0.0), reverse=True)
        ordered: List[Dict[str, Any]] = scored_docs

        return ordered

    async def _download_and_extract_markdown(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = [asyncio.to_thread(self._download_single_page_markdown, item) for item in results]
        return await asyncio.gather(*tasks)

    def _download_single_page_markdown(self, result: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(result)
        url = (enriched.get("url") or "").strip()
        enriched["content_source"] = "searxng"
        if not url:
            return enriched

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Не найдена зависимость requests. Установите зависимости: pip install -r requirements.txt"
            ) from exc

        try:
            response = requests.get(
                url,
                timeout=settings.request_timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; searxng-mcp-orchestrator/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ru,en;q=0.9",
                },
            )
            response.raise_for_status()
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding
            html = response.text
        except requests.RequestException as exc:
            logger.warning("Не удалось скачать страницу %s: %s", url, exc)
            return enriched

        markdown, plain_text = self._extract_content_bundle(html, url)
        rank_text = (markdown or plain_text or "").strip()
        if rank_text:
            enriched["content"] = rank_text[: settings.retrieval_doc_max_chars]
            enriched["display_content"] = (
                plain_text[: settings.retrieval_doc_max_chars]
                if plain_text
                else rank_text[: settings.retrieval_doc_max_chars]
            )
            enriched["content_source"] = "trafilatura"
        return enriched

    def _extract_content_bundle(self, html: str, url: str) -> tuple[str, str]:
        try:
            import trafilatura
        except ImportError as exc:
            raise RuntimeError(
                "Не найдена зависимость trafilatura. Установите зависимости: pip install -r requirements.txt"
            ) from exc

        markdown = trafilatura.extract(
            html,
            output_format="markdown",
            include_formatting=True,
            include_links=True,
            url=url,
        )
        plain_text = trafilatura.extract(
            html,
            output_format="txt",
            url=url,
        )
        if not markdown:
            markdown = trafilatura.extract(html, url=url)
        if not plain_text and markdown:
            plain_text = markdown

        return (markdown or "").strip(), self._normalize_plain_text(plain_text or "")

    @staticmethod
    def _normalize_plain_text(text: str) -> str:
        # Сохраняем абзацы для читаемого терминального вывода, но убираем мусорные пробелы.
        lines = []
        for raw_line in text.replace("\r", "\n").split("\n"):
            line = " ".join(raw_line.split()).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _jina_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        pairs = []
        valid_indexes = []
        for idx, item in enumerate(results):
            text = self._compose_stage2_text(item)
            if not text:
                continue
            pairs.append([query, text])
            valid_indexes.append(idx)

        if not pairs:
            return results

        try:
            scores = self._jina_reranker.predict(pairs, show_progress_bar=False)
        except RuntimeError as exc:
            if "BFloat16" not in str(exc):
                raise

            logger.warning("Jina reranker вернул BFloat16 ошибку. Повтор с float32 на CPU.")
            self._force_jina_float32_cpu()
            scores = self._jina_reranker.predict(pairs, show_progress_bar=False)

        scored_results = []
        for idx, score in zip(valid_indexes, scores):
            enriched = dict(results[idx])
            enriched["stage2_score"] = float(score)
            scored_results.append(enriched)

        scored_results.sort(key=lambda item: item.get("stage2_score", 0.0), reverse=True)
        return scored_results

    @staticmethod
    def _compose_stage1_text(item: Dict[str, Any]) -> str:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        return f"{title}\n{content}".strip()

    @staticmethod
    def _compose_stage2_text(item: Dict[str, Any]) -> str:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        if not title and not content:
            return ""
        if title and content:
            return f"# {title}\n\n{content}"
        return title or content

    def _force_jina_float32_cpu(self) -> None:
        inner_model = getattr(self._jina_reranker, "model", None)
        if inner_model is None:
            return
        try:
            inner_model.to("cpu")
            inner_model.float()
        except Exception as exc:
            logger.warning("Не удалось принудительно переключить Jina reranker в float32 CPU: %s", exc)

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0
