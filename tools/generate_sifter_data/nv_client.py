"""NVIDIA API pool — deepseek-v4-flash on NVIDIA infra, 4 keys, 10K+ TPM each."""

from __future__ import annotations

import threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Iterator

import requests

API_BASE = "https://integrate.api.nvidia.com/v1"
MODELS = [
    "mistralai/mistral-small-4-119b-2603",   # 238ms — fastest
    "mistralai/ministral-14b-instruct-2512", # 474ms — fallback
    "mistralai/mixtral-8x7b-instruct-v0.1",  # 564ms — fallback
]

KEYS = [
    "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",
    "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",
    "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",
    "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",
]


@dataclass
class NvResponse:
    content: str; key_index: int; tokens_used: int; latency: float


class NvidiaPool:
    def __init__(self):
        self.keys = KEYS
        self._lock = threading.Lock()
        self.req = 0; self.tok = 0; self.err = 0

    def chat(self, messages: list[dict], temperature=0.8, max_tokens=300) -> Optional[NvResponse]:
        for ki in range(len(self.keys)):
            key = self.keys[ki]
            model = MODELS[ki % len(MODELS)]  # round-robin models across keys
            t0 = time.time()
            try:
                resp = requests.post(f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": messages, "max_tokens": max_tokens,
                          "temperature": temperature, "top_p": 0.95},
                    timeout=45)
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    with self._lock: self.req += 1; self.tok += tokens
                    return NvResponse(content=content, key_index=ki, tokens_used=tokens, latency=elapsed)
            except Exception:
                with self._lock: self.err += 1
        return None

    def batch_chat(self, prompts: list[list[dict]], temperature=0.8, max_tokens=300) -> list[Optional[NvResponse]]:
        results: list[Optional[NvResponse]] = [None] * len(prompts)
        lock = threading.Lock()
        def worker(i, msgs):
            r = self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            with lock: results[i] = r
        with ThreadPoolExecutor(max_workers=min(20, len(prompts))) as ex:
            fs = [ex.submit(worker, i, p) for i, p in enumerate(prompts)]
            for f in as_completed(fs): f.result()
        return results

    @property
    def total_cost(self) -> float:
        return 0.0

    def print_stats(self):
        print(f"[Nvidia] {self.req} req, {self.tok} tok, {self.err} err")


_nvpool: Optional[NvidiaPool] = None


def get_nvpool() -> NvidiaPool:
    global _nvpool
    if _nvpool is None: _nvpool = NvidiaPool()
    return _nvpool
