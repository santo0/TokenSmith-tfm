import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Send chat-completion requests to OpenRouter with automatic retries.

    Args:
        api_key: OpenRouter API key.
        retries: Number of *extra* attempts after the first (default 2 → 3 total).
        backoff_base: Base delay in seconds for exponential backoff
            (default 1.0).
        max_workers: Thread pool size for concurrent calls (default 8).
    """

    def __init__(
        self,
        api_key: str,
        retries: int = 2,
        backoff_base: float = 1.0,
        max_workers: int = 8,
    ):
        self.api_key = api_key
        self.retries = retries
        self.backoff_base = backoff_base
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None,
        timeout: int,
    ) -> dict:
        """Execute one chat request with retries; return the raw response body."""
        payload: dict = {"model": model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    _URL, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_exc = e
                if attempt < self.retries:
                    delay = random.uniform(
                        0, self.backoff_base * (2**attempt)
                    )
                    logger.warning(
                        "OpenRouter attempt %d/%d failed: %s"
                        " — retrying in %.2fs…",
                        attempt + 1,
                        self.retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> str:
        """Send a chat request and return the assistant message content.

        Args:
            model: OpenRouter model identifier (e.g. ``"google/gemini-3-flash-preview"``).
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            response_format: Optional ``{"type": "json_object"}`` or similar.
            timeout: Per-request timeout in seconds.

        Returns:
            The stripped content string from the first choice.

        Raises:
            The last exception encountered when all attempts fail.
        """
        data = self._request(model, messages, response_format, timeout)
        return data["choices"][0]["message"]["content"].strip()

    def chat_with_meta(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> tuple[str, dict]:
        """Like ``chat`` but also returns response metadata.

        Returns:
            A ``(content, meta)`` tuple where *meta* contains:

            - ``id``         — OpenRouter generation ID
            - ``model_used`` — model actually served (may differ from requested)
            - ``usage``      — ``{"prompt_tokens": int, "completion_tokens": int,
              "total_tokens": int}``
        """
        data = self._request(model, messages, response_format, timeout)
        content = data["choices"][0]["message"]["content"].strip()
        meta: dict = {
            "id": data.get("id"),
            "model_used": data.get("model", model),
            "usage": data.get("usage", {}),
        }
        return content, meta

    def chat_many(
        self,
        requests_: list[dict],
        model: str,
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> list[str | Exception]:
        """Send multiple chat requests concurrently.

        Each item in ``requests_`` must be a dict with a ``"messages"`` key
        and an optional ``"model"`` / ``"response_format"`` key to override
        the defaults for that request.

        Results are returned in the same order as the input list.
        Individual failures are returned as ``Exception`` instances rather
        than raising, so callers can inspect partial successes.
        """
        return [
            content
            for content, _, _ in self.chat_many_with_meta(
                requests_, model, response_format, timeout
            )
        ]

    def chat_many_with_meta(
        self,
        requests_: list[dict],
        model: str,
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> list[tuple[str | Exception, dict | None, float]]:
        """Concurrent batch requests returning content, metadata, and latency.

        Returns:
            List (same order as input) of ``(content_or_exc, meta_or_none,
            latency_seconds)`` triples.  *meta* follows the same schema as
            ``chat_with_meta``; it is ``None`` when the request failed.
        """
        results: list[tuple[str | Exception, dict | None, float]] = [
            (Exception("not started"), None, 0.0)
        ] * len(requests_)

        def _call(
            index: int, req: dict
        ) -> tuple[int, str | Exception, dict | None, float]:
            t0 = time.perf_counter()
            try:
                content, meta = self.chat_with_meta(
                    model=req.get("model", model),
                    messages=req["messages"],
                    response_format=req.get("response_format", response_format),
                    timeout=timeout,
                )
                return index, content, meta, time.perf_counter() - t0
            except Exception as e:
                return index, e, None, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_call, i, req): i
                for i, req in enumerate(requests_)
            }
            for future in as_completed(futures):
                idx, content, meta, latency = future.result()
                results[idx] = (content, meta, latency)

        return results
