"""Sekel constants — every threshold has a number, no adjectives.

Owns:           All magic numbers, product sizing, rate limits, thresholds.
Depends on:     os.environ (API keys, API base URL), pathlib (filesystem paths).
Provides:       ALL_CATEGORIES, TIER_LABELS, SD thresholds, API keys from env.
Invariants:     API_KEYS loaded from DEEPSEEK_KEYS env var (comma-separated).
                MAX_CONCURRENT = 200 (50 threads per key, I/O-bound).
                TPM_PER_KEY = 30_000, RPM_PER_KEY = 500 primed at init.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = ROOT / "tools"
DATA_DIR = TOOLS_DIR / "generate_sifter_data" / "data"

# ── Product sizing ──────────────────────────────────────────────────

PRODUCT_SIZES: dict[int, int] = {
    1: 5500,
    2: 11000,
    3: 1650,
    4: 2000,
}

SAMPLES_PER_CATEGORY = 200
SAMPLES_PER_TIER: dict[int, int] = {
    1: int(SAMPLES_PER_CATEGORY * 0.30),
    2: int(SAMPLES_PER_CATEGORY * 0.25),
    3: int(SAMPLES_PER_CATEGORY * 0.20),
    4: int(SAMPLES_PER_CATEGORY * 0.15),
    5: int(SAMPLES_PER_CATEGORY * 0.10),
}

TIER_LABELS: dict[int, str] = {
    1: "obvious",
    2: "subtle",
    3: "contextual",
    4: "adversarial",
    5: "false_positive",
}

# Deprecated alias maintained for backward compat in existing product code
DIFFICULTY_TIERS = TIER_LABELS

MANIPULATION_CATEGORIES: list[str] = [
    "authority_bias",
    "social_proof",
    "scarcity",
    "urgency",
    "emotional_appeal",
    "expertise_signaling",
    "semantic_traps",
    "template_fitting",
    "emphasis_typographic",
]

AUX_CATEGORIES: list[str] = [
    "multi_category",
    "clean_safe",
]

ALL_CATEGORIES: list[str] = MANIPULATION_CATEGORIES + AUX_CATEGORIES

# ── Rate limiting ───────────────────────────────────────────────────

TPM_PER_KEY: int = 30_000
RPM_PER_KEY: int = 500
MAX_CONCURRENT: int = 15

# ── DeepSeek API ────────────────────────────────────────────────────

API_BASE: str = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


def _load_api_keys() -> list[str]:
    raw = os.environ.get("DEEPSEEK_KEYS", "")
    if raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    return [
        "sk-26d0e090ed7b4d90803aae706d9b7247",
        "sk-325e26cc36474aad80822f5282ceffd7",
        "sk-8f26eca8b3b840f0b42d78fa539a4a52",
    ]


API_KEYS: list[str] = _load_api_keys()

MODELS: dict[str, str] = {
    "fast": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

# ── Signal Density thresholds ───────────────────────────────────────

SD_ACCEPT: float = 0.0  # accept all for collection; filter during retraining
SD_GOLD: float = 9.0
SD_ARCHIVE: float = 6.9

# Deprecated aliases for existing product code compatibility
SD_MIN_ACCEPT = SD_ACCEPT
SD_GOLD_THRESHOLD = SD_GOLD
SD_ARCHIVE_MAX = SD_ARCHIVE

# ── Gap thresholds ──────────────────────────────────────────────────

GAP_RETRY: float = 0.30
MAX_RETRIES: int = 3

# Deprecated aliases
GAP_RETRY_THRESHOLD = GAP_RETRY
MAX_REGENERATION_ATTEMPTS = MAX_RETRIES

# ── Batch ───────────────────────────────────────────────────────────

BATCH_SIZE: int = 8  # prompts per batch generate call

# ── Output paths ────────────────────────────────────────────────────

PRODUCT_OUTPUTS: dict[int, Path] = {
    1: DATA_DIR / "keyword_regression.jsonl",
    2: DATA_DIR / "classifier_training.jsonl",
    3: DATA_DIR / "adversarial_evasion.jsonl",
    4: DATA_DIR / "hard_negatives.jsonl",
}
