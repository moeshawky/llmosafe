"""Keyword sifter + TF-IDF classifier — pure Python mirror of Rust llmosafe.

Owns:           Keyword lists (source of truth), compute_bias_breakdown(),
                TFIDFClassifier, validate_text().
Depends on:     config (ROOT, TOOLS_DIR for default model path).
Provides:       BiasBreakdown, ClassificationResult, ValidationResult.
                compute_bias_breakdown(text) → dict,
                validate_text(text) → ValidationResult,
                compute_signal_density() → float (3 formulas).
Invariants:     Keyword lists match src/llmosafe_sifter.rs exactly.
                Negation TTL = 6, keyword weight = 100, emphasis weight = 50.
                Multi-word phase 2 matches Rust window + negation logic.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.generate_sifter_data.schemas import ValidationResult

from tools.generate_sifter_data.config import ROOT

# ── Keyword lists — mirror src/llmosafe_sifter.rs exactly ───────────

NEGATION_WORDS: list[str] = [
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "hardly",
    "scarcely",
    "barely",
    "doesn't",
    "isn't",
    "wasn't",
    "shouldn't",
    "won't",
    "don't",
]

AUTHORITY_BIAS: list[str] = [
    "expert",
    "experts",
    "official",
    "officials",
    "government",
    "doctor",
    "doctors",
    "scientist",
    "scientists",
    "guaranteed",
    "certified",
    "proven",
]

SOCIAL_PROOF: list[str] = [
    "everyone",
    "thousands",
    "millions",
    "trending",
    "viral",
    "bestseller",
    "bestsellers",
    "testimonials",
    "consensus",
    "majority",
    "crowd",
]

SCARCITY: list[str] = [
    "limited",
    "rare",
    "exclusive",
    "handcrafted",
    "small-batch",
    "collectible",
    "once-in-a-lifetime",
    "restricted",
    "shortage",
    "vanishing",
    "low-stock",
    "while-supplies-last",
    "sold-out",
    "member-only",
]

URGENCY: list[str] = [
    "hurry",
    "rush",
    "deadline",
    "expiring",
    "immediately",
    "limited-time",
    "last-chance",
    "act-now",
    "don't-wait",
]

EMOTIONAL_APPEAL: list[str] = [
    "fear",
    "shocking",
    "miracle",
    "incredible",
    "tragic",
    "desperate",
    "heartwarming",
    "devastating",
    "thrilling",
    "terrifying",
]

EXPERTISE_SIGNALING: list[str] = [
    "sophisticated",
    "cutting-edge",
    "state-of-the-art",
    "revolutionary",
    "revolutionaries",
    "patented",
    "breakthrough",
    "breakthroughs",
    "proprietary",
    "paradigm",
    "holistic",
    "synergy",
]

SEMANTIC_TRAPS: list[str] = [
    "not but",
    "instead of",
    "rather than",
    "on the other hand",
]

TEMPLATE_FITTING: list[str] = [
    "as an ai",
    "my purpose is",
    "according to my instructions",
    "it is important to remember",
    "please note that",
    "i cannot",
    "i am programmed to",
]

# Map category names to keyword lists
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "authority_bias": AUTHORITY_BIAS,
    "social_proof": SOCIAL_PROOF,
    "scarcity": SCARCITY,
    "urgency": URGENCY,
    "emotional_appeal": EMOTIONAL_APPEAL,
    "expertise_signaling": EXPERTISE_SIGNALING,
    "semantic_traps": SEMANTIC_TRAPS,
    "template_fitting": TEMPLATE_FITTING,
}

CATEGORY_FIELD_MAP: dict[str, str] = {
    "authority_bias": "authority",
    "social_proof": "social_proof",
    "scarcity": "scarcity",
    "urgency": "urgency",
    "emotional_appeal": "emotional_appeal",
    "expertise_signaling": "expertise_signaling",
    "semantic_traps": "semantic_traps",
    "template_fitting": "template_fitting",
    "emphasis_typographic": "emphasis",
}

KEYWORD_WEIGHT: int = 100
EMPHASIS_WEIGHT: int = 50
NEGATION_TTL: int = 6

# ── Bias breakdown (mirrors BiasBreakdown struct) ────────────────────


@dataclass
class BiasBreakdown:
    """Mirror of Rust BiasBreakdown struct (llmosafe_sifter.rs:237).

    Each field stores the accumulated weight for a bias category.
    Emphasis tracks ALL CAPS typographic manipulation.
    """

    authority: int = 0
    social_proof: int = 0
    scarcity: int = 0
    urgency: int = 0
    emotional_appeal: int = 0
    expertise_signaling: int = 0
    semantic_traps: int = 0
    template_fitting: int = 0
    emphasis: int = 0

    def total(self) -> int:
        return (
            self.authority
            + self.social_proof
            + self.scarcity
            + self.urgency
            + self.emotional_appeal
            + self.expertise_signaling
            + self.semantic_traps
            + self.template_fitting
            + self.emphasis
        )

    def to_dict(self) -> dict:
        return {
            "authority": self.authority,
            "social_proof": self.social_proof,
            "scarcity": self.scarcity,
            "urgency": self.urgency,
            "emotional_appeal": self.emotional_appeal,
            "expertise_signaling": self.expertise_signaling,
            "semantic_traps": self.semantic_traps,
            "template_fitting": self.template_fitting,
            "emphasis": self.emphasis,
            "total": self.total(),
        }


# ── Keyword matching helpers ────────────────────────────────────────


def _word_in_list(word: str, lst: list[str]) -> bool:
    """Case-insensitive match of word against any entry in lst."""
    lower = word.casefold()
    return any(kw.casefold() == lower for kw in lst)


def _strip_punctuation(word: str) -> str:
    """Strip ASCII punctuation from both ends of a token.

    Mirrors Rust: raw_word.trim_matches(|c| c.is_ascii_punctuation()).
    ASCII punctuation = is_ascii && !is_alphanumeric.
    Covers: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
    """
    start = 0
    end = len(word)
    while start < end and word[start].isascii() and not word[start].isalnum():
        start += 1
    while end > start and word[end - 1].isascii() and not word[end - 1].isalnum():
        end -= 1
    return word[start:end]


# ── compute_bias_breakdown ──────────────────────────────────────────


def compute_bias_breakdown(text: str) -> BiasBreakdown:
    """Pure-Python reimplementation of get_bias_breakdown() from llmosafe_sifter.rs:288-393.

    Phase 1 (single-word): For each whitespace-split token:
        1. Strip ASCII punctuation from both ends.
        2. Check if token is a negation word → set negation_ttl = 6.
        3. If negation_ttl > 0, skip keyword matching, decrement TTL.
        4. Otherwise, check token against all 8 keyword lists (case-insensitive).
        5. Check ALL CAPS emphasis (len ≥ 2, all ASCII uppercase).

    Phase 2 (multi-word): For each multi-word phrase in SEMANTIC_TRAPS and
        TEMPLATE_FITTING:
        1. Tokenize text with punctuation stripped, track negation positions.
        2. Slide windows of phrase.word_count across tokens.
        3. If window matches phrase AND first token not negated → add weight.

    Args:
        text: Arbitrary input string.

    Returns:
        BiasBreakdown with 9 category scores. Each keyword match = +100,
        each emphasis hit = +50. Negation suppresses detection within TTL=6 tokens.

    Pre-conditions:  text is a str (empty string produces all-zero breakdown).
    Post-conditions: All scores are non-negative. Emphasis only fires on
                     ASCII-uppercase tokens of length ≥ 2.
    """
    breakdown = BiasBreakdown()

    # ── Phase 1: single-word detection ──
    negation_ttl = 0

    for raw_word in text.split():
        trimmed = _strip_punctuation(raw_word)
        if not trimmed:
            continue

        is_negation = _word_in_list(trimmed, NEGATION_WORDS)
        negated = negation_ttl > 0

        if is_negation:
            negation_ttl = NEGATION_TTL
        else:
            negation_ttl = max(0, negation_ttl - 1)

        if negated:
            continue

        if _word_in_list(trimmed, AUTHORITY_BIAS):
            breakdown.authority += KEYWORD_WEIGHT
        if _word_in_list(trimmed, SOCIAL_PROOF):
            breakdown.social_proof += KEYWORD_WEIGHT
        if _word_in_list(trimmed, SCARCITY):
            breakdown.scarcity += KEYWORD_WEIGHT
        if _word_in_list(trimmed, URGENCY):
            breakdown.urgency += KEYWORD_WEIGHT
        if _word_in_list(trimmed, EMOTIONAL_APPEAL):
            breakdown.emotional_appeal += KEYWORD_WEIGHT
        if _word_in_list(trimmed, EXPERTISE_SIGNALING):
            breakdown.expertise_signaling += KEYWORD_WEIGHT
        if _word_in_list(trimmed, SEMANTIC_TRAPS):
            breakdown.semantic_traps += KEYWORD_WEIGHT
        if _word_in_list(trimmed, TEMPLATE_FITTING):
            breakdown.template_fitting += KEYWORD_WEIGHT

        # Emphasis: ALL CAPS words (len >= 2) that are ASCII-only uppercase.
        # Excludes camelCase, PascalCase, mixed-case, emoji, Unicode scripts.
        if (
            not negated
            and len(trimmed) >= 2
            and trimmed.isascii()
            and trimmed.isupper()
        ):
            breakdown.emphasis += EMPHASIS_WEIGHT

    # ── Phase 2: multi-word phrase matching ──
    tokens: list[str] = [_strip_punctuation(t) for t in text.split()]
    tokens = [t for t in tokens if t]

    if not tokens:
        return breakdown

    # Compute negation positions for phase 2
    negated_positions: list[bool] = [False] * len(tokens)
    neg_ttl = 0
    for i, token in enumerate(tokens):
        is_neg = _word_in_list(token, NEGATION_WORDS)
        curr_negated = neg_ttl > 0
        if is_neg:
            neg_ttl = NEGATION_TTL
        else:
            neg_ttl = max(0, neg_ttl - 1)
        negated_positions[i] = curr_negated

    for phrase in SEMANTIC_TRAPS:
        if " " not in phrase:
            continue
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            if negated_positions[i]:
                continue
            window = tokens[i : i + n]
            if all(a.casefold() == b.casefold() for a, b in zip(window, phrase_tokens)):
                breakdown.semantic_traps += KEYWORD_WEIGHT
                break

    for phrase in TEMPLATE_FITTING:
        if " " not in phrase:
            continue
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            if negated_positions[i]:
                continue
            window = tokens[i : i + n]
            if all(a.casefold() == b.casefold() for a, b in zip(window, phrase_tokens)):
                breakdown.template_fitting += KEYWORD_WEIGHT
                break

    return breakdown


# ── FNV-1a 64-bit hash (matches Rust StreamingTokenizer) ────────────

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3


def fnv1a_64(s: str) -> int:
    """Compute FNV-1a 64-bit hash of an ASCII string."""
    h = FNV_OFFSET
    for b in s.encode("ascii", errors="ignore"):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# ── Sigmoid LUT (256 entries, exact copy from Rust) ──────────────────

SIGMOID_LUT_ENTRIES = [
    0.000335,
    0.000368,
    0.000404,
    0.000443,
    0.000486,
    0.000533,
    0.000585,
    0.000642,
    0.000704,
    0.000773,
    0.000848,
    0.000930,
    0.001021,
    0.001120,
    0.001228,
    0.001347,
    0.001478,
    0.001621,
    0.001778,
    0.001950,
    0.002139,
    0.002346,
    0.002573,
    0.002823,
    0.003096,
    0.003395,
    0.003724,
    0.004084,
    0.004479,
    0.004912,
    0.005387,
    0.005908,
    0.006479,
    0.007105,
    0.007791,
    0.008543,
    0.009368,
    0.010272,
    0.011263,
    0.012349,
    0.013539,
    0.014843,
    0.016271,
    0.017835,
    0.019548,
    0.021423,
    0.023476,
    0.025723,
    0.028180,
    0.030866,
    0.033802,
    0.037010,
    0.040514,
    0.044339,
    0.048513,
    0.053063,
    0.058020,
    0.063414,
    0.069277,
    0.075642,
    0.082541,
    0.090006,
    0.098070,
    0.106763,
    0.116113,
    0.126145,
    0.136882,
    0.148342,
    0.160540,
    0.173483,
    0.187174,
    0.201608,
    0.216771,
    0.232641,
    0.249187,
    0.266370,
    0.284144,
    0.302454,
    0.321240,
    0.340433,
    0.359965,
    0.379761,
    0.399746,
    0.419845,
    0.439984,
    0.460088,
    0.480088,
    0.499917,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
    0.500000,
] * 2
SIGMOID_LUT: list[float] = SIGMOID_LUT_ENTRIES[:256]
SIGMOID_LUT = SIGMOID_LUT + [0.5] * max(0, 256 - len(SIGMOID_LUT))
SIGMOID_LUT = SIGMOID_LUT[:256]


def sigmoid(x: float) -> float:
    """Sigmoid via lookup table — matches Rust implementation.

    Args:
        x: Input score (log-odds).

    Returns:
        Probability in (0.0, 1.0).
    """
    if x == 0.0:
        return 0.5
    if x > 0.0:
        return 1.0 - sigmoid(-x)
    if x <= -8.0:
        return 0.0
    idx = max(0, min(255, int((x + 8.0) * 31.875)))
    return SIGMOID_LUT[idx]


# ── Classification Result ────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Mirror of Rust ClassificationResult (llmosafe_classifier.rs:184).

    Args:
        score: Log-odds score (sum of intercept + Σ idf·coef for matched tokens).
        probability: Sigmoid-transformed score in (0.0, 1.0).
        is_manipulation: True if score > threshold.
        oov_ratio: Fraction of tokens not found in vocabulary.
        tokens_matched: Number of tokens found in vocabulary.
        tokens_total: Total number of hashed tokens (unigrams + bigrams).
    """

    score: float = 0.0
    probability: float = 0.5
    is_manipulation: bool = False
    oov_ratio: float = 1.0
    tokens_matched: int = 0
    tokens_total: int = 0


# ── TF-IDF Classifier ────────────────────────────────────────────────


class TFIDFClassifier:
    """Python mirror of Rust llmosafe_classifier. Loads vocab_model.bin.

    Purpose:         Classify text as manipulative or not using TF-IDF features
                     and logistic regression, matching the Rust implementation.
    Dependencies:    vocab_model.bin binary file (u32 vocab_size + f32 threshold
                     + f32 intercept + N×(u64 hash + f32 idf + f32 coef) LE).
                     FNV-1a hash for tokenization.

    State Machine:   Uninitialized → (load model) → Ready to classify.
                     classify() is idempotent and thread-safe (read-only).
    Invariants:      vocab sorted by hash for binary search.
                     Tokenization uses unigrams + bigrams with 0x1f separator.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """Load vocab_model.bin binary.

        Args:
            model_path: Path to vocab_model.bin. Defaults to tools/vocab_model.bin.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        if model_path is None:
            model_path = ROOT / "tools" / "vocab_model.bin"

        self.vocab: list[tuple[int, float, float]] = []
        self.threshold: float = 0.5
        self.intercept: float = -2.0
        self._vocab_size: int = 0

        if model_path.exists():
            self._load(model_path)

    def _load(self, path: Path) -> None:
        """Parse binary model format: u32 vocab_size, f32 threshold, f32 intercept,
        then N records of (u64 hash, f32 idf, f32 coef) in little-endian.
        """
        with open(path, "rb") as f:
            data = f.read()

        if len(data) < 12:
            return

        self._vocab_size = struct.unpack_from("<I", data, 0)[0]
        self.threshold = struct.unpack_from("<f", data, 4)[0]
        self.intercept = struct.unpack_from("<f", data, 8)[0]

        offset = 12
        for _ in range(self._vocab_size):
            h, idf_val, coef_val = struct.unpack_from("<Qff", data, offset)
            self.vocab.append((h, idf_val, coef_val))
            offset += 16

        self.vocab.sort(key=lambda x: x[0])

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text into FNV-1a hashes (unigrams + bigrams).

        Splits on non-alphanumeric characters, lowercases, and hashes
        each unigram. Bigrams use 0x1f separator byte — matches Rust tokenizer.
        """
        hashes: list[int] = []
        words: list[str] = []
        current: list[str] = []

        for ch in text:
            if ch.isascii() and ch.isalnum():
                current.append(ch.lower())
            else:
                if current:
                    token = "".join(current)
                    if len(token) <= 256:
                        words.append(token)
                    current = []
        if current:
            token = "".join(current)
            if len(token) <= 256:
                words.append(token)

        for w in words:
            hashes.append(fnv1a_64(w))

        for i in range(1, len(words)):
            bigram = f"{words[i - 1]}\x1f{words[i]}"
            hashes.append(fnv1a_64(bigram))

        return hashes

    def classify(self, text: str) -> ClassificationResult:
        """Classify text using TF-IDF features and logistic regression.

        Args:
            text: Arbitrary input string to classify.

        Returns:
            ClassificationResult with score, probability, is_manipulation,
            oov_ratio, tokens_matched, tokens_total.

        Pre-conditions:  model must be loaded (vocab non-empty).
        Post-conditions: probability ∈ (0.0, 1.0), oov_ratio ∈ [0.0, 1.0].
        """
        hashes = self._tokenize(text)
        total = len(hashes)

        score = self.intercept
        matched = 0

        for h in hashes:
            lo, hi = 0, len(self.vocab)
            while lo < hi:
                mid = (lo + hi) // 2
                if self.vocab[mid][0] < h:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(self.vocab) and self.vocab[lo][0] == h:
                _hash, idf_val, coef_val = self.vocab[lo]
                score += idf_val * coef_val
                matched += 1

        oov_ratio = 1.0 - (matched / total) if total > 0 else 1.0
        probability = sigmoid(score)

        return ClassificationResult(
            score=score,
            probability=probability,
            is_manipulation=score > self.threshold,
            oov_ratio=oov_ratio,
            tokens_matched=matched,
            tokens_total=total,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


# ── Global classifier singleton ──────────────────────────────────────

_classifier: Optional[TFIDFClassifier] = None


def get_classifier() -> TFIDFClassifier:
    """Return global TFIDFClassifier singleton, loading on first access."""
    global _classifier
    if _classifier is None:
        _classifier = TFIDFClassifier()
    return _classifier


# ── Unified validation ───────────────────────────────────────────────


def validate_text(
    text: str,
    classifier: Optional[TFIDFClassifier] = None,
) -> ValidationResult:
    """Run text through keyword sifter AND TF-IDF classifier.

    Uses global classifier singleton if none provided.

    Args:
        text: Text sample to validate.
        classifier: Optional pre-constructed classifier. If None, uses singleton.

    Returns:
        ValidationResult with bias_breakdown dict, classifier probabilty,
        keywords_triggered list, and total_bias_score.
    """
    from tools.generate_sifter_data.schemas import ValidationResult

    if classifier is None:
        classifier = get_classifier()

    breakdown = compute_bias_breakdown(text)
    class_result = classifier.classify(text)

    keywords_triggered = [
        name
        for name, field in CATEGORY_FIELD_MAP.items()
        if getattr(breakdown, field, 0) > 0
    ]

    return ValidationResult(
        text=text,
        bias_breakdown=breakdown.to_dict(),
        classifier_prob=class_result.probability,
        classifier_score=class_result.score,
        classifier_is_manipulation=class_result.is_manipulation,
        classifier_oov_ratio=class_result.oov_ratio,
        keywords_triggered=keywords_triggered,
        total_bias_score=breakdown.total(),
    )


# ── Signal Density scoring ──────────────────────────────────────────


def compute_signal_density(
    label_manipulation_score: int,
    label_is_manipulation: bool,
    classifier_prob: float,
    classifier_is_manipulation: bool,
    category: str,
    tier: int,
    novelty: float = 0.85,
) -> float:
    """Score a generated sample for training value (0.0–10.0).

    Standard formula (Product 2):
        correctness  = max(0, 1 - |label_score/100 - classifier_prob| * 2)
        complexity   = tier / 5
        diversity    = 0.95 if tier >= 4 else 0.85 if tier >= 3 else 0.70
        product_bonus = 1.00
        SD_raw = (correctness * 0.35 + novelty * 0.25 + complexity * 0.20 + diversity * 0.20) * 10

    Args:
        label_manipulation_score: LLM-assigned manipulation score [0, 100].
        label_is_manipulation: LLM boolean label.
        classifier_prob: TF-IDF classifier probability [0.0, 1.0].
        classifier_is_manipulation: TF-IDF boolean label.
        category: Manipulation category name.
        tier: Difficulty tier [1, 5].
        novelty: Edit-distance ratio from existing samples [0.0, 1.0].

    Returns:
        Signal density in [0.0, 10.0], rounded to 2 decimal places.
    """
    prob_gap = abs(label_manipulation_score / 100.0 - classifier_prob)
    correctness = max(0.0, 1.0 - prob_gap * 2.0)

    if label_is_manipulation != classifier_is_manipulation:
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


def compute_signal_density_evasion(
    label_manipulation_score: int,
    classifier_prob: float,
    tier: int = 4,
    novelty: float = 0.85,
) -> float:
    """Signal density for Product 3 (adversarial evasion).

    The gap IS the training signal. Weights adjusted:
        evasion_value = max(0, label_score/100 - classifier_prob)  [gap is value]
        SD_raw = (evasion_value * 0.40 + novelty * 0.30 + complexity * 0.15 + diversity * 0.15) * 10
        product_bonus = 1.10

    Args:
        label_manipulation_score: LLM-assigned manipulation score [0, 100].
        classifier_prob: TF-IDF classifier probability [0.0, 1.0].
        tier: Difficulty tier (default 4 for evasion).
        novelty: Novelty score [0.0, 1.0].

    Returns:
        Signal density in [0.0, 10.0], rounded to 2 decimal places.
    """
    evasion_value = max(0.0, label_manipulation_score / 100.0 - classifier_prob)
    complexity = tier / 5.0
    diversity = 0.95

    sd_raw = (
        evasion_value * 0.40 + novelty * 0.30 + complexity * 0.15 + diversity * 0.15
    ) * 10.0
    sd = min(10.0, sd_raw * 1.10)
    return round(sd, 2)


def compute_signal_density_contrastive(
    contrastive_gap: float,
    keyword_overlap: float = 1.0,
) -> float:
    """Signal density for Product 4 (contrastive pairs).

    The pair's discriminative value IS the score:
        discrimination = max(0, min(1, contrastive_gap / 0.7))
        overlap_penalty = 0.8 + keyword_overlap * 0.2
        SD_raw = discrimination * overlap_penalty * 10 * 1.05
        SD = min(10, SD_raw)

    Args:
        contrastive_gap: manipulation_prob - benign_prob.
        keyword_overlap: Fraction of keywords shared between texts [0.0, 1.0].

    Returns:
        Signal density in [0.0, 10.0], rounded to 2 decimal places.
    """
    discrimination = max(0.0, min(1.0, contrastive_gap / 0.7))
    overlap_penalty = 0.8 + keyword_overlap * 0.2
    sd_raw = discrimination * overlap_penalty * 10.0 * 1.05
    return round(min(10.0, sd_raw), 2)
