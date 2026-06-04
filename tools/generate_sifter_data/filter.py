"""Sample quality filter — Signal Density scoring, gap detection, retry decisions.

Owns:           compute_signal_density (standard, evasion, contrastive),
                evaluate_sample() → FilterDecision.
Depends on:     validator (ValidationResult), config (SD_ACCEPT, SD_GOLD, GAP_RETRY).
Provides:       FilterDecision dataclass, evaluate_sample() function.
Invariants:     SD_ACCEPT = 7.0, SD_GOLD = 9.0, GAP_RETRY = 0.30.
                3 distinct SD formulas for Products 2, 3, 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.generate_sifter_data.config import GAP_RETRY, SD_ACCEPT, SD_GOLD
from tools.generate_sifter_data.schemas import ValidationResult


@dataclass
class FilterDecision:
    """Decision on whether to accept a generated sample.

    Args:
        accepted: True if sample meets SD ≥ SD_ACCEPT threshold.
        signal_density: Computed SD score [0.0, 10.0].
        gap_score: Absolute difference between label and classifier probability.
        reason: Human-readable rejection reason (empty string if accepted).
        tier: "gold" (SD ≥ 9.0), "accepted" (7.0 ≤ SD < 9.0), "rejected" (SD < 7.0).
    """

    accepted: bool
    signal_density: float
    gap_score: float
    reason: str = ""
    tier: str = "rejected"

    @property
    def is_gold(self) -> bool:
        """SD ≥ 9.0 — promote to active training set."""
        return self.signal_density >= SD_GOLD


def _compute_sd_standard(
    label_score: int,
    label_is_manip: bool,
    classifier_prob: float,
    classifier_is_manip: bool,
    tier: int,
    novelty: float = 0.85,
) -> float:
    """Standard SD formula for Product 2.

    correctness = max(0, 1 - |label_score/100 - classifier_prob| * 2)
    Penalty for boolean disagreement: correctness *= 0.5.
    """
    prob_gap = abs(label_score / 100.0 - classifier_prob)
    correctness = max(0.0, 1.0 - prob_gap * 2.0)
    if label_is_manip != classifier_is_manip:
        correctness *= 0.5

    complexity = tier / 5.0

    if tier >= 4:
        diversity = 0.95
    elif tier >= 3:
        diversity = 0.85
    else:
        diversity = 0.70

    sd = (
        correctness * 0.35 + novelty * 0.25 + complexity * 0.20 + diversity * 0.20
    ) * 10.0
    return round(min(10.0, sd), 2)


def _compute_sd_evasion(
    label_score: int,
    classifier_prob: float,
    tier: int = 4,
    novelty: float = 0.85,
) -> float:
    """Evasion SD formula for Product 3.

    The gap IS the training signal:
        evasion_value = max(0, label_score/100 - classifier_prob)
    """
    evasion_value = max(0.0, label_score / 100.0 - classifier_prob)
    complexity = tier / 5.0

    sd_raw = (
        evasion_value * 0.40 + novelty * 0.30 + complexity * 0.15 + 0.95 * 0.15
    ) * 10.0
    sd = min(10.0, sd_raw * 1.10)
    return round(sd, 2)


def _compute_sd_contrastive(
    contrastive_gap: float,
    keyword_overlap: float = 1.0,
) -> float:
    """Contrastive SD formula for Product 4.

    Pair's discriminative value IS the score:
        discrimination = max(0, min(1, contrastive_gap / 0.7))
    """
    discrimination = max(0.0, min(1.0, contrastive_gap / 0.7))
    overlap_penalty = 0.8 + keyword_overlap * 0.2
    sd_raw = discrimination * overlap_penalty * 10.0 * 1.05
    return round(min(10.0, sd_raw), 2)


def compute_signal_density(
    label_score: int,
    label_is_manip: bool,
    classifier_prob: float,
    classifier_is_manip: bool,
    tier: int,
    novelty: float = 0.85,
    product: int = 2,
    **kwargs,
) -> float:
    """Compute signal density score for a generated sample.

    Routes to the appropriate formula based on product type.

    Args:
        label_score: LLM-assigned manipulation score [0, 100].
        label_is_manip: LLM boolean label.
        classifier_prob: TF-IDF classifier probability [0.0, 1.0].
        classifier_is_manip: TF-IDF boolean label.
        tier: Difficulty tier [1, 5].
        novelty: Edit-distance ratio [0.0, 1.0].
        product: 2=standard, 3=evasion, 4=contrastive.
        **kwargs: Additional formula-specific parameters.

    Returns:
        Signal density in [0.0, 10.0], rounded to 2 decimal places.
    """
    if product == 3:
        return _compute_sd_evasion(label_score, classifier_prob, tier, novelty)
    elif product == 4:
        contrastive_gap = kwargs.get("contrastive_gap", 0.0)
        keyword_overlap = kwargs.get("keyword_overlap", 1.0)
        return _compute_sd_contrastive(contrastive_gap, keyword_overlap)
    else:
        return _compute_sd_standard(
            label_score,
            label_is_manip,
            classifier_prob,
            classifier_is_manip,
            tier,
            novelty,
        )


def evaluate_sample(
    label_score: int,
    label_is_manip: bool,
    validation: ValidationResult,
    category: str,
    tier: int,
    product: Literal[2, 3, 4] = 2,
    novelty: float = 0.85,
    **kwargs,
) -> FilterDecision:
    """Evaluate a generated sample: SD scoring + gap check + acceptance decision.

    For Product 3 (adversarial evasion): gap IS the training signal.
    For Product 4 (contrastive pairs): evaluates discriminative value of pair.

    Args:
        label_score: LLM manipulation score [0, 100].
        label_is_manip: LLM boolean label.
        validation: ValidationResult from validate_text().
        category: Manipulation category name.
        tier: Difficulty tier [1, 5].
        product: 2, 3, or 4.
        novelty: Novelty score [0.0, 1.0].
        **kwargs: Additional parameters for SD formula.

    Returns:
        FilterDecision with accept/reject, SD score, gap, reason, tier.
    """
    gap = abs(label_score / 100.0 - validation.classifier_prob)

    sd = compute_signal_density(
        label_score=label_score,
        label_is_manip=label_is_manip,
        classifier_prob=validation.classifier_prob,
        classifier_is_manip=validation.classifier_is_manipulation,
        tier=tier,
        novelty=novelty,
        product=product,
        **kwargs,
    )

    if sd >= SD_GOLD:
        return FilterDecision(
            accepted=True,
            signal_density=sd,
            gap_score=gap,
            reason="",
            tier="gold",
        )
    elif sd >= SD_ACCEPT:
        return FilterDecision(
            accepted=True,
            signal_density=sd,
            gap_score=gap,
            reason="",
            tier="accepted",
        )
    elif gap > GAP_RETRY:
        return FilterDecision(
            accepted=False,
            signal_density=sd,
            gap_score=gap,
            reason=f"gap {gap:.2f} exceeds retry threshold {GAP_RETRY}",
            tier="rejected",
        )
    else:
        return FilterDecision(
            accepted=False,
            signal_density=sd,
            gap_score=gap,
            reason=f"SD {sd:.2f} below accept threshold {SD_ACCEPT}",
            tier="rejected",
        )
