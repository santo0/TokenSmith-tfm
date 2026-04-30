import json
import logging
import re
from math import sqrt
from typing import Any

from src.knowledge_graph.extractors.base_extractor import BaseExtractor
from src.knowledge_graph.models import Chunk, ExtractionResult
from src.knowledge_graph.openrouter_client import OpenRouterClient
from src.knowledge_graph.prompts import OPENROUTER_KEYWORD_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class OpenRouterExtractor(BaseExtractor):
    """Keyword extractor using OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        adaptive_top_n: bool = False,
        top_n: int = 10,
        retries: int = 1,
    ):
        super().__init__()
        self.model = model
        self.top_n = top_n
        self.adaptive_top_n = adaptive_top_n
        self._client = OpenRouterClient(api_key, retries=retries)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model": self.model,
                "top_n": self.top_n,
                "adaptive_top_n": self.adaptive_top_n,
            }
        )
        return config

    def _parse_keywords(self, content: str) -> list[str]:
        """Parse a JSON list of keywords from the LLM response string.

        Tries progressively more lenient strategies:
        1. Direct JSON parse
        2. Sanitize invalid backslash escapes, then parse
        3. Extract the first [...] substring, then parse (with sanitization)

        Raises:
            ValueError: When all strategies fail.
        """
        def _load(text: str) -> list | None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        def _sanitize(text: str) -> str:
            # Replace backslashes not part of a valid JSON escape sequence
            return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        result = _load(content) or _load(_sanitize(content))
        if result is None:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                extracted = match.group(0)
                result = _load(extracted) or _load(_sanitize(extracted))
        if result is None:
            raise ValueError(f"Cannot parse JSON list from response: {content!r}")
        if not isinstance(result, list):
            raise ValueError(f"Response is not a list: {result!r}")
        return result

    def _is_keyword_in_text(self, keyword: str, text: str) -> bool:
        return keyword.lower() in text.lower()

    def _build_result(self, chunk: Chunk, keywords_raw: list[str], requested_n: int) -> ExtractionResult:
        valid = [kw for kw in keywords_raw if self._is_keyword_in_text(kw, chunk.text)]
        invented = len(keywords_raw) - len(valid)
        if invented:
            logger.debug("Chunk %s: filtered %d invented keyword(s)", chunk.id, invented)
        return ExtractionResult(
            chunk_id=chunk.id,
            keywords=valid,
            stats={
                "requested_n": requested_n,
                "total_extracted": len(keywords_raw),
                "invented_count": invented,
            },
        )

    def extract(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        requests_: list[dict] = []
        top_ns: list[int] = []
        for chunk in chunks:
            top_n = int(sqrt(len(chunk.text))) if self.adaptive_top_n else self.top_n
            top_ns.append(top_n)
            requests_.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": OPENROUTER_KEYWORD_EXTRACTION_PROMPT.format(top_n=top_n),
                        },
                        {"role": "user", "content": f"Documents: {chunk.text}"},
                    ]
                }
            )

        outcomes = self._client.chat_many(requests_, model=self.model)

        results: list[ExtractionResult | None] = [None] * len(chunks)
        retry_indices: list[int] = []

        for i, (chunk, outcome) in enumerate(zip(chunks, outcomes)):
            if isinstance(outcome, Exception):
                logger.error("Chunk %s: all attempts failed — %s", chunk.id, outcome)
                results[i] = ExtractionResult(chunk_id=chunk.id, keywords=[])
            else:
                try:
                    results[i] = self._build_result(chunk, self._parse_keywords(outcome), top_ns[i])
                except Exception as e:
                    logger.warning("Chunk %s: parse error, will retry — %s", chunk.id, e)
                    retry_indices.append(i)

        for attempt in range(1, self._client.retries + 1):
            if not retry_indices:
                break
            logger.info(
                "Retrying %d parse-failed chunk(s) (attempt %d/%d)",
                len(retry_indices), attempt, self._client.retries,
            )
            retry_outcomes = self._client.chat_many(
                [requests_[i] for i in retry_indices], model=self.model
            )
            still_failing: list[int] = []
            for idx, outcome in zip(retry_indices, retry_outcomes):
                chunk = chunks[idx]
                if isinstance(outcome, Exception):
                    logger.error("Chunk %s: all attempts failed — %s", chunk.id, outcome)
                    results[idx] = ExtractionResult(chunk_id=chunk.id, keywords=[])
                else:
                    try:
                        results[idx] = self._build_result(chunk, self._parse_keywords(outcome), top_ns[idx])
                    except Exception as e:
                        logger.warning(
                            "Chunk %s: parse error on retry %d — %s", chunk.id, attempt, e
                        )
                        still_failing.append(idx)
            retry_indices = still_failing

        for idx in retry_indices:
            logger.error("Chunk %s: parse error after all retries", chunks[idx].id)
            results[idx] = ExtractionResult(chunk_id=chunks[idx].id, keywords=[])

        self.metadata["total_chunks"] = len(chunks)
        self.metadata["chunks_with_invented_keywords"] = sum(
            1 for r in results if r.stats.get("invented_count", 0) > 0  # type: ignore[union-attr]
        )
        self.metadata["total_invented_keywords"] = sum(
            r.stats.get("invented_count", 0) for r in results  # type: ignore[union-attr]
        )
        self.metadata["chunks_with_shortfall"] = sum(
            1 for r in results  # type: ignore[union-attr]
            if r.stats.get("total_extracted", r.stats.get("requested_n", 0)) < r.stats.get("requested_n", 0)  # type: ignore[union-attr]
        )
        self.metadata["total_shortfall"] = sum(
            max(0, r.stats.get("requested_n", 0) - r.stats.get("total_extracted", 0))  # type: ignore[union-attr]
            for r in results
        )

        return results  # type: ignore[return-value]
