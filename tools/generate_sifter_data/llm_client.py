"""DeepSeek API client — 4-key pool with token buckets for sifter data generation.

Owns:           DeepSeekPool — thread-safe HTTP client with per-key rate limiting.
Depends on:     token_bucket (TokenBucket), config (API_KEYS, MODELS, MAX_CONCURRENT).
                requests (HTTP), concurrent.futures (parallelism).
Provides:       chat() → ChatResult, batch_generate() → Iterator[tuple[int, ChatResult]].
                total_cost property, stats property.
Invariants:     4 API keys loaded from env, never hardcoded.
                Buckets primed at full capacity on construction.
                thinking.type = "disabled" on every request.
                Exponential backoff on 429 (2^attempt seconds, capped at 16).
                MAX_CONCURRENT = 200 workers.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import requests

from tools.generate_sifter_data.config import (
    API_BASE,
    API_KEYS,
    MAX_CONCURRENT,
    MODELS,
)
from tools.generate_sifter_data.token_bucket import TokenBucket


# ── Types ───────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Successful API call result.

    Args:
        content: Text response from choices[0].message.content.
        model: Model name used (e.g. 'deepseek-v4-flash').
        key_index: Which API key (0-3) was used.
        tokens_used: Total tokens from usage.total_tokens.
        cost_yuan: Cost in yuan computed from pricing.
        latency_seconds: Wall clock time from request to response.
    """

    content: str
    model: str
    key_index: int
    tokens_used: int
    cost_yuan: float
    latency_seconds: float


@dataclass
class LLMError:
    """Failed API call result.

    Args:
        key_index: Which API key failed.
        status_code: HTTP status code (None for network errors).
        message: Human-readable error description.
    """

    key_index: int
    status_code: Optional[int]
    message: str


ChatResult = Union[LLMResponse, LLMError, None]


# ── Pool ────────────────────────────────────────────────────────────


class DeepSeekPool:
    """Thread-safe pool of 4 DeepSeek API keys with per-key token buckets.

    Purpose:         Rate-limited concurrent access to DeepSeek chat completion API.
    Dependencies:    TokenBucket per (key_index, model_id) pair.
                     requests.Session for HTTP.
    State Machine:   (key, model) → Bucket(primed) → try_consume → _call → refill.
                     Stats accumulated per-request under _lock.
    Invariants:      All 4 keys initialized from config.API_KEYS.
                     thinking.type = "disabled" on every request body.
                     429s trigger exponential backoff + bucket drain.
    """

    def __init__(self, keys: Optional[list[str]] = None):
        """Initialize pool with API keys.

        Args:
            keys: List of API key strings. If None, loads from DEEPSEEK_KEYS env var.

        Raises:
            ValueError: If no API keys available.
        """
        self.keys = keys if keys else API_KEYS
        if not self.keys:
            raise ValueError(
                "No API keys available. Set DEEPSEEK_KEYS environment variable "
                "as comma-separated list of DeepSeek API keys."
            )

        self._buckets: dict[tuple[int, str], TokenBucket] = {}
        self._lock = threading.Lock()
        self._stats = {
            "req": 0,
            "tok": 0,
            "cost": 0.0,
            "err": 0,
            "r429": 0,
        }

    def _get_bucket(self, key_idx: int, model_id: str) -> TokenBucket:
        """Get or create token bucket for (key_idx, model_id).

        New buckets are primed at full capacity (TokenBucket default).
        Thread-safe: caller holds self._lock.
        """
        key = (key_idx, model_id)
        if key not in self._buckets:
            self._buckets[key] = TokenBucket()
        return self._buckets[key]

    def _call(
        self,
        key_idx: int,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 300,
    ) -> ChatResult:
        """Execute one API call against a specific key.

        Does NOT check token bucket — caller must acquire capacity first.

        Args:
            key_idx: Index into self.keys (0-3).
            model_id: "fast" or "pro".
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature [0.0, 1.0].
            max_tokens: Maximum completion tokens.

        Returns:
            LLMResponse on success, LLMError on failure, None on unhandled exception.
        """
        key = self.keys[key_idx]
        model_name = MODELS.get(model_id, model_id)
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                },
                timeout=120,
            )
            elapsed = time.time() - t0

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                cost = (
                    usage.get("prompt_tokens", 0) * 1.0
                    + usage.get("completion_tokens", 0) * 2.0
                ) / 1_000_000
                with self._lock:
                    self._stats["req"] += 1
                    self._stats["tok"] += tokens
                    self._stats["cost"] += cost
                return LLMResponse(
                    content=content,
                    model=model_name,
                    key_index=key_idx,
                    tokens_used=tokens,
                    cost_yuan=cost,
                    latency_seconds=elapsed,
                )
            elif resp.status_code == 429:
                with self._lock:
                    self._stats["r429"] += 1
                    # Drain bucket on 429
                    key = (key_idx, model_id)
                    if key in self._buckets:
                        self._buckets[key].tpm = 0.0
                return LLMError(
                    key_index=key_idx,
                    status_code=429,
                    message="rate limited",
                )
            else:
                with self._lock:
                    self._stats["err"] += 1
                return LLMError(
                    key_index=key_idx,
                    status_code=resp.status_code,
                    message=resp.text[:200],
                )
        except Exception as exc:
            with self._lock:
                self._stats["err"] += 1
            return LLMError(
                key_index=key_idx,
                status_code=None,
                message=str(exc),
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "fast",
        temperature: float = 0.8,
        max_tokens: int = 300,
        retries_per_key: int = 1,
    ) -> ChatResult:
        """Send one chat completion, retrying across all 4 keys.

        Round-robin across keys: tries each key in sequence, retrying
        up to retries_per_key times. Uses exponential backoff on 429.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            model: "fast" (deepseek-v4-flash) or "pro" (deepseek-v4-pro).
            temperature: Sampling temperature [0.0, 1.0].
            max_tokens: Maximum tokens in completion.
            retries_per_key: Maximum attempts per key across all rounds.

        Returns:
            LLMResponse on success (content = text string).
            LLMError on failure with status_code and message.
            None if all keys exhausted and no calls succeeded.
        """
        tokens_est = 500
        last_error: Optional[LLMError] = None

        for attempt in range(retries_per_key):
            for ki in range(len(self.keys)):
                with self._lock:
                    bucket = self._get_bucket(ki, model)
                acquired = bucket.try_consume(tokens_est)
                if not acquired:
                    continue

                result = self._call(ki, model, messages, temperature, max_tokens)
                match result:
                    case LLMResponse():
                        return result
                    case LLMError(status_code=429):
                        last_error = result
                        # Exponential backoff: 2^attempt seconds, capped at 16
                        backoff = min(2**attempt, 16)
                        time.sleep(backoff)
                        break
                    case LLMError():
                        last_error = result
                        continue
                    case None:
                        continue

            # Small delay between retry rounds
            time.sleep(0.1)

        return last_error

    def batch_generate(
        self,
        prompts: list[list[dict[str, str]]],
        model: str = "fast",
        temperature: float = 0.8,
        max_tokens: int = 300,
    ) -> Iterator[tuple[int, ChatResult]]:
        """Generate responses for multiple prompts in parallel across keys.

        Uses ThreadPoolExecutor with MAX_CONCURRENT workers.
        Yields (prompt_index, result) in completion order.

        Args:
            prompts: List of message lists — one per prompt.
            model: "fast" or "pro".
            temperature: Sampling temperature [0.0, 1.0].
            max_tokens: Maximum tokens per completion.

        Yields:
            Tuple of (prompt_index: int, result: ChatResult).
            Index matches order in prompts list.
        """
        results: dict[int, ChatResult] = {}
        lock = threading.Lock()

        def worker(idx: int, msgs: list[dict[str, str]]) -> None:
            res = self.chat(
                msgs, model=model, temperature=temperature, max_tokens=max_tokens
            )
            with lock:
                results[idx] = res

        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(prompts))) as pool:
            futures = {pool.submit(worker, i, p): i for i, p in enumerate(prompts)}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

        for i in range(len(prompts)):
            yield (i, results.get(i))

    @property
    def total_cost(self) -> float:
        """Cumulative cost in yuan across all calls."""
        return float(self._stats["cost"])

    @property
    def stats(self) -> dict:
        """Aggregated statistics dictionary."""
        with self._lock:
            capacity_total = 0.0
            bucket_count = 0
            for key, bucket in self._buckets.items():
                capacity_total += bucket.capacity_remaining
                bucket_count += 1
            avg_capacity = capacity_total / max(1, bucket_count)

        return {
            "total_requests": self._stats["req"],
            "total_tokens": self._stats["tok"],
            "total_cost_yuan": self._stats["cost"],
            "errors": self._stats["err"],
            "rate_limits": self._stats["r429"],
            "capacity_remaining": round(avg_capacity, 4),
        }

    def print_stats(self) -> None:
        """Print current pool statistics to stdout."""
        s = self.stats
        print(
            f"[Pool] {s['total_requests']} req, {s['total_tokens']} tok, "
            f"¥{s['total_cost_yuan']:.4f} cost, {s['errors']} err, "
            f"{s['rate_limits']} 429s, cap={s['capacity_remaining']:.2f}"
        )


# ── Global pool singleton ────────────────────────────────────────────

_pool: Optional[DeepSeekPool] = None


def get_pool() -> DeepSeekPool:
    """Return global DeepSeekPool singleton, creating it on first access."""
    global _pool
    if _pool is None:
        _pool = DeepSeekPool()
    return _pool
