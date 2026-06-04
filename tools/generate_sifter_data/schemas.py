"""Pydantic ≥2.0 schemas for all 4 data products, validation results, and pool stats.

Owns:           BaseModel definitions for all serializable domain objects.
Depends on:     pydantic (BaseModel, Field), enum, datetime.
Provides:       KeywordRegressionSample, ClassifierTrainingSample, AdversarialSample,
                HardNegativePair, ValidationResult, PoolStats, GeneratorStats.
Invariants:     All models serialize via model_dump_json(). No manual to_jsonl methods.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Product 1: Keyword Regression ───────────────────────────────────


class KeywordRegressionSample(BaseModel):
    """Deterministic test case for keyword sifter regression testing.

    Each sample asserts that a specific text input produces an exact
    BiasBreakdown — expected_* fields are the ground truth.
    """

    text: str
    keyword: str
    category: str
    variant: str
    expected_authority: int = 0
    expected_social_proof: int = 0
    expected_scarcity: int = 0
    expected_urgency: int = 0
    expected_emotional_appeal: int = 0
    expected_expertise_signaling: int = 0
    expected_semantic_traps: int = 0
    expected_template_fitting: int = 0
    expected_emphasis: int = 0
    expected_total: int = 0


# ── Product 2: Classifier Training ──────────────────────────────────


class ClassifierTrainingSample(BaseModel):
    """LLM-generated labeled sample validated against sifter + classifier.

    Stores text, LLM label, classifier output, and Signal Density score.
    The gap between LLM label and classifier probability drives retry logic.
    """

    text: str
    category: str
    tier: int
    tier_label: str
    manipulation_score: int = Field(ge=0, le=100)
    is_manipulation: bool
    keywords_triggered: list[str] = Field(default_factory=list)
    classifier_prob: Optional[float] = None
    classifier_is_manipulation: Optional[bool] = None
    bias_breakdown: Optional[dict] = None
    signal_density: float = 0.0
    gap_manipulation_score: float = 0.0
    generation_attempt: int = 1


# ── Product 3: Adversarial Evasion ──────────────────────────────────


class AdversarialSample(BaseModel):
    """LLM-generated adversarial text that evades keyword detection.

    Text is labeled manipulative by LLM but keyword sifter reports safe —
    the gap IS the training signal for adversarial robustness.
    """

    text: str
    target_category: str
    evaded_keywords: list[str] = Field(default_factory=list)
    manipulation_score: int = Field(ge=0, le=100)
    is_manipulation: bool
    keyword_sifter_says_safe: bool
    keyword_sifter_total: int = 0
    classifier_prob: Optional[float] = None
    classifier_is_manipulation: Optional[bool] = None
    signal_density: float = 0.0
    generation_attempt: int = 1


# ── Product 4: Hard Negative Pairs ──────────────────────────────────


class HardNegativePair(BaseModel):
    """Contrastive pair — same surface keywords, opposite intent.

    Benign text uses keywords legitimately; manipulation text uses
    the same keywords manipulatively. The discriminative gap trains
    the classifier to distinguish keyword presence from intent.
    """

    category: str
    benign_text: str
    manipulation_text: str
    shared_keywords: list[str] = Field(default_factory=list)
    benign_score: int = Field(ge=0, le=100)
    manipulation_score: int = Field(ge=0, le=100)
    benign_classifier_prob: Optional[float] = None
    manipulation_classifier_prob: Optional[float] = None
    contrastive_gap: float = 0.0
    signal_density: float = 0.0


# ── Validation ──────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Combined keyword sifter + TF-IDF classifier output for one text sample."""

    text: str
    bias_breakdown: dict
    classifier_prob: float
    classifier_score: float
    classifier_is_manipulation: bool
    classifier_oov_ratio: float
    keywords_triggered: list[str] = Field(default_factory=list)
    total_bias_score: int = 0


# ── Pool Stats ──────────────────────────────────────────────────────


class PoolStats(BaseModel):
    """Aggregated API usage statistics across all keys."""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost_yuan: float = 0.0
    errors: int = 0
    rate_limits: int = 0
    capacity_remaining: float = 1.0


# ── Generator Stats ─────────────────────────────────────────────────


class GeneratorStats(BaseModel):
    """Output statistics for a product generation run."""

    product_id: int
    product_name: str
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    avg_signal_density: float = 0.0
    total_cost_yuan: float = 0.0
    runtime_seconds: float = 0.0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
