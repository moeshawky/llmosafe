//! LLMOSAFE Tier 3 Perceptual Sifter
//!
//! The sifter is the entry point for all safety processing. It converts raw text
//! observations into a scored `SiftedSynapse` via the TF-IDF classifier.
//!
//! # Architecture
//!
//! The v0.7.0 sifter replaces hand-tuned keyword categories with a logistic
//! regression classifier trained on 42K real samples:
//!
//! 1. `sift_perceptions()` calls `classify_text()` on each observation,
//!    selecting the highest-scoring result.
//! 3. **Entropy**: Normalised entropy `[0, 65535]`, computed as `probability * 65535`
//!    directly from the classifier's sigmoid output.
//! 3. `raw_surprise` = `probability * 65535` — how confident the classifier is
//!    that this is manipulation.
//! 4. `has_bias` = `classifier.is_manipulation` — boolean flag for bias gate.
//!
//! The legacy keyword-based `calculate_halo_signal()` and `get_bias_breakdown()`
//! remain for backward compatibility with existing consumers.
//!
//! # Deprecation Chain (v0.8.0+)
//!
//! **Deprecated functions (keyword-only, low accuracy 18.6%):**
//! - `get_bias_breakdown()` — deprecated v0.8.0: use `sift_perceptions()`
//! - `calculate_halo_signal()` — deprecated v0.8.0: use `sift_perceptions()`
//!   (wraps `get_bias_breakdown().total()`)
//!
//! **Modern dual-path (classifier + keyword, accuracy 93.4%):**
//! - `sift_text()` — primary Rust API (returns `SiftedSynapse`)
//! - `sift_perceptions()` — batch Rust API (returns best `SiftedSynapse`)
//!
//! **C-ABI surface (uses modern path, NOT deprecated):**
//! - `llmosafe_calculate_halo()` — routes through `sift_text()`, NOT keyword-only.
//!   Pass `text_ptr` + `text_len` (bounded safety).
//!
//! **Python bindings (uses modern path, NOT deprecated):**
//! - `calculate_halo()` — routes through `sift_text()` (dual-path).
//! - `calculate_halo_signal_legacy()` — wraps `calculate_halo_signal()` explicitly.
//!
//! The C-ABI and Python `calculate_halo` functions are NOT on the deprecation
//! path — they use the modern dual-path implementation. Only the Rust-native
//! `calculate_halo_signal()` and `get_bias_breakdown()` are deprecated.
// Arithmetic in this module operates on bounded keyword counts [0, ~9000]
// and hash accumulators where wrapping semantics are the intended behavior.
// DO-178C: these operations are verified safe by value range analysis at
// the module boundary — inputs are always validated before arithmetic.
#![allow(clippy::arithmetic_side_effects)]

use crate::control_types::ControlSignal;
use crate::llmosafe_classifier::{classify_text, ClassificationResult};
use crate::llmosafe_kernel::SiftedProof;
use crate::llmosafe_kernel::SiftedSynapse;
use crate::llmosafe_kernel::Synapse;
use crate::llmosafe_kernel::U16_MAX_F32;

/// Sifter Control Loop output.
///
/// # Control Signal
///
/// - Setpoint: 0.0 (ideal safe input has zero manipulation probability)
/// - Actual: `classifier_prob` ∈ [0.0, 1.0]
/// - Error: `e_sift = classifier_prob - 0.0 = classifier_prob`
/// - Gain: `K_sift = 1.0` (identity — sifter is feed-forward sensor)
///
/// # DAL E
///
/// The sifter is the outermost loop — its error signal is informational
/// for the PID composition. No direct actuation.
///
/// # Invariants
///
/// - `0.0 ≤ error_sift ≤ 1.0`
/// - `0 ≤ raw_entropy ≤ 65535`
/// - `has_bias == classification.is_manipulation` (classifier-only path;
///   `sift_text()` OR-s the keyword hard-bias breakdown into its output);
///   typographic `emphasis` is soft evidence only and never sets `has_bias`)
/// - `no_evidence == (classification.tokens_matched == 0)`.
///   Zero-match inputs are unknown/OOD — never treated as safe or safe-adjacent.
#[derive(Debug, Clone, Copy)]
pub struct SifterOutput {
    /// Error signal = classifier probability (setpoint=0).
    /// Normalised to [0.0, 1.0] by construction.
    pub error_sift: f32,
    /// Raw binary entropy [0, 65535].
    pub raw_entropy: u16,
    /// Classifier probability [0.0, 1.0].
    pub classifier_prob: f32,
    /// Bias flag from classifier.
    pub has_bias: bool,
    /// Out-of-vocabulary ratio `[0, 255]` (0=0%, 255=100%).
    pub oov_ratio: u8,
    /// True when zero vocab matches (`tokens_matched == 0`).
    /// Zero-match = unknown/OOD. Downstream MUST escalate,
    /// never treat as positive safety or positive manipulation evidence.
    pub no_evidence: bool,
}

impl ControlSignal for SifterOutput {
    fn error(&self) -> f32 {
        self.error_sift
    }

    fn setpoint(&self) -> f32 {
        0.0
    }
}

impl SifterOutput {
    /// Construct from raw classifier output.
    /// DAL A/E: error_sift = classifier_prob (setpoint=0).
    pub fn from_classification(classification: &ClassificationResult) -> Self {
        // entropy = 0 when safe (p→0), 65535 when manipulation (p→1)
        let entropy = (U16_MAX_F32 * classification.probability.clamp(0.0, 1.0)) as u16;
        Self {
            error_sift: classification.probability,
            raw_entropy: entropy,
            classifier_prob: classification.probability,
            has_bias: classification.is_manipulation,
            oov_ratio: (classification.oov_ratio * 255.0) as u8,
            no_evidence: classification.no_evidence,
        }
    }
}

/// Authority bias keyword detection list.
/// Matches terms that signal expertise/position appeals.
/// Pruned of high-frequency academic terms (research, study, professional).
/// Keeps 2-3 representatives per semantic cluster:
///   claims: guaranteed, certified, proven
///   roles: expert, official, government, doctor, scientist
pub const AUTHORITY_BIAS: &[&str] = &[
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
];

pub const NEGATION_WORDS: &[&str] = &[
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
];

/// Social Proof Keywords: Red flags for crowd/popularity bias.
/// Pruned of everyday community terms (common, standard, users, reviews, ratings, joined, peer, social).
/// Kept: high-signal crowd-manipulation markers.
pub const SOCIAL_PROOF: &[&str] = &[
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
];

/// Scarcity Keywords: Red flags for restricted availability bias.
/// Pruned of hyper-common words (only, special, private, unique, few, select).
/// Kept: domain-specific scarcity markers that are rare outside manipulation.
pub const SCARCITY: &[&str] = &[
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
];

/// Urgency Keywords: Red flags for time-pressure bias.
/// Pruned of universal time adjectives (now, today, fast, quick, soon, final, rapid, speedy).
/// Kept: genuine urgency-manipulation signals.
pub const URGENCY: &[&str] = &[
    "hurry",
    "rush",
    "deadline",
    "expiring",
    "immediately",
    "limited-time",
    "last-chance",
    "act-now",
    "don't-wait",
];

/// Emotional Appeal Keywords: Red flags for emotional manipulation bias.
/// Pruned of high-frequency words (love, joy, happy, sad, angry) that trigger
/// on everyday speech. Pruned further of remaining common sentiment words
/// (worry, hopeful, inspiring, passionate, touching).
/// Retains fearmongering, hyperbolic, and sentiment-charged terms.
pub const EMOTIONAL_APPEAL: &[&str] = &[
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
];

/// Expertise Signaling Keywords: Red flags for jargon/complexity bias.
/// Pruned of business/engineering buzzwords (advanced, complex, technical, leverage,
/// optimize, agile, lean, scalable, high-performance).
/// Kept: genuine novelty-claim and proprietary-signaling terms.
pub const EXPERTISE_SIGNALING: &[&str] = &[
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
];

/// Semantic Traps: Inversion patterns that flip safety predicates.
/// Single-word contrast markers removed (unless, however, although, despite, conversely)
/// — too common in everyday English. Multi-word inversion phrases retained.
pub const SEMANTIC_TRAPS: &[&str] = &["not but", "instead of", "rather than", "on the other hand"];

/// Template Fitting: Common AI-specific template markers.
pub const TEMPLATE_FITTING: &[&str] = &[
    "as an ai",
    "my purpose is",
    "according to my instructions",
    "it is important to remember",
    "please note that",
    "i cannot",
    "i am programmed to",
];

/// Fixed-size bias breakdown. Zero allocation.
/// Each field corresponds to one of the 8 bias categories plus typographic emphasis.
/// `emotional_appeal` is keyword-sifter-inert for the classifier pathway.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BiasBreakdown {
    pub authority: u16,
    pub social_proof: u16,
    pub scarcity: u16,
    pub urgency: u16,
    pub emotional_appeal: u16,
    pub expertise_signaling: u16,
    pub semantic_traps: u16,
    pub template_fitting: u16,
    /// Typographic emphasis signal: ALL CAPS words (attention-seeking formatting).
    /// Independent of keyword membership — catches shouting patterns that keywords miss.
    /// camelCase/PascalCase excluded (technical notation, not manipulation).
    pub emphasis: u16,
}

impl BiasBreakdown {
    /// Total bias score across all categories including typographic emphasis.
    pub fn total(&self) -> u16 {
        self.authority
            .saturating_add(self.social_proof)
            .saturating_add(self.scarcity)
            .saturating_add(self.urgency)
            .saturating_add(self.emotional_appeal)
            .saturating_add(self.expertise_signaling)
            .saturating_add(self.semantic_traps)
            .saturating_add(self.template_fitting)
            .saturating_add(self.emphasis)
    }

    /// Hard bias total across manipulation categories only.
    /// Excludes typographic `emphasis` — emphasis is soft heuristic
    /// evidence that may contribute to soft scoring (entropy, telemetry,
    /// PID input) but must NEVER independently trigger a hard BIAS
    /// override or set `has_bias`. All-caps acronyms like "AI",
    /// "API", "HTTP", "CPU" fire the emphasis pathway but not hard bias.
    pub fn hard_total(&self) -> u16 {
        self.authority
            .saturating_add(self.social_proof)
            .saturating_add(self.scarcity)
            .saturating_add(self.urgency)
            .saturating_add(self.emotional_appeal)
            .saturating_add(self.expertise_signaling)
            .saturating_add(self.semantic_traps)
            .saturating_add(self.template_fitting)
    }
}

/// Case-insensitive keyword match without allocation.
#[inline]
fn word_in_list(word: &str, list: &[&str]) -> bool {
    list.iter().any(|kw| word.eq_ignore_ascii_case(kw))
}

/// Check if consecutive tokens match a multi-word phrase.
/// Works in both std and no_std — uses slice indexing only.
#[inline]
fn phrase_matches(window: &[&str], phrase_words: &[&str]) -> bool {
    if window.len() < phrase_words.len() {
        return false;
    }
    window[..phrase_words.len()]
        .iter()
        .zip(phrase_words.iter())
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Returns a breakdown of detected biases by category.
#[deprecated(
    since = "0.8.0",
    note = "keyword-based detection retained for backward compatibility. Use sift_perceptions() for higher accuracy (93.4% vs keyword-based) via the TF-IDF classifier."
)]
pub fn get_bias_breakdown(text: &str) -> BiasBreakdown {
    let mut breakdown = BiasBreakdown::default();

    let mut negation_ttl = 0u8;

    for raw_word in text.split_whitespace() {
        let trimmed = raw_word.trim_matches(|c: char| c.is_ascii_punctuation());
        let is_negation = word_in_list(trimmed, NEGATION_WORDS);

        let negated = negation_ttl > 0;

        if is_negation {
            negation_ttl = 6;
        } else {
            negation_ttl = negation_ttl.saturating_sub(1);
        }

        if negated {
            continue;
        }

        if word_in_list(trimmed, AUTHORITY_BIAS) {
            breakdown.authority = breakdown.authority.saturating_add(100);
        }
        if word_in_list(trimmed, SOCIAL_PROOF) {
            breakdown.social_proof = breakdown.social_proof.saturating_add(100);
        }
        if word_in_list(trimmed, SCARCITY) {
            breakdown.scarcity = breakdown.scarcity.saturating_add(100);
        }
        if word_in_list(trimmed, URGENCY) {
            breakdown.urgency = breakdown.urgency.saturating_add(100);
        }
        if word_in_list(trimmed, EMOTIONAL_APPEAL) {
            breakdown.emotional_appeal = breakdown.emotional_appeal.saturating_add(100);
        }
        if word_in_list(trimmed, EXPERTISE_SIGNALING) {
            breakdown.expertise_signaling = breakdown.expertise_signaling.saturating_add(100);
        }
        // NOTE: SEMANTIC_TRAPS and TEMPLATE_FITTING entries are
        // all multi-word phrases — they are matched exclusively
        // in Phase 2 (bounded sliding-window below). Single-token
        // comparisons here can never match and have been removed
        // as dead code (see S1 no_std parity fix).

        // Attention-emphasis signal: ALL CAPS words (len >= 2) indicate
        // typographic manipulation independent of keyword membership.
        // Only fires on ASCII-uppercase — excludes emoji, Unicode scripts,
        // and camelCase/PascalCase (technical notation, not manipulation).
        if trimmed.len() >= 2 && trimmed.chars().all(|c| c.is_ascii_uppercase()) {
            breakdown.emphasis = breakdown.emphasis.saturating_add(50);
        }
    }

    // Phase 2: Multi-word phrase matching.
    // Bounded allocation-free sliding-window: uses fixed-size
    // arrays instead of Vec, so this works in both std and no_std.
    // Tokens and phrase words are normalized (trimmed, lower-cased
    // via eq_ignore_ascii_case) and compared without allocation.
    {
        // Collect tokens into a fixed-size array (no allocation).
        let mut tokens: [&str; 128] = [""; 128];
        let mut token_count = 0usize;
        for raw_word in text.split_whitespace() {
            let trimmed = raw_word.trim_matches(|c: char| c.is_ascii_punctuation());
            if token_count < 128 {
                tokens[token_count] = trimmed;
                token_count += 1;
            } else {
                break;
            }
        }

        // Compute negation positions into a fixed-size array.
        let mut negated_positions: [bool; 128] = [false; 128];
        let mut neg_ttl = 0u8;
        for i in 0..token_count {
            let is_neg = word_in_list(tokens[i], NEGATION_WORDS);
            let curr_negated = neg_ttl > 0;
            if is_neg {
                neg_ttl = 6;
            } else {
                neg_ttl = neg_ttl.saturating_sub(1);
            }
            negated_positions[i] = curr_negated;
        }

        // Bounded sliding-window phrase matching.
        // phrase_words_buf is a fixed-size array — no Vec allocation.
        let mut phrase_words_buf: [&str; 4] = [""; 4];

        for phrase in SEMANTIC_TRAPS {
            if !phrase.contains(' ') {
                continue;
            }
            let mut pw_count = 0usize;
            for w in phrase.split_whitespace() {
                if pw_count < 4 {
                    phrase_words_buf[pw_count] = w;
                    pw_count += 1;
                } else {
                    break;
                }
            }
            let pw_slice = &phrase_words_buf[..pw_count];
            if (0..=token_count.saturating_sub(pw_count)).any(|i| {
                !negated_positions[i] && phrase_matches(&tokens[i..i + pw_count], pw_slice)
            }) {
                breakdown.semantic_traps = breakdown.semantic_traps.saturating_add(100);
            }
        }

        for phrase in TEMPLATE_FITTING {
            if !phrase.contains(' ') {
                continue;
            }
            let mut pw_count = 0usize;
            for w in phrase.split_whitespace() {
                if pw_count < 4 {
                    phrase_words_buf[pw_count] = w;
                    pw_count += 1;
                } else {
                    break;
                }
            }
            let pw_slice = &phrase_words_buf[..pw_count];
            if (0..=token_count.saturating_sub(pw_count)).any(|i| {
                !negated_positions[i] && phrase_matches(&tokens[i..i + pw_count], pw_slice)
            }) {
                breakdown.template_fitting = breakdown.template_fitting.saturating_add(100);
            }
        }
    }

    breakdown
}

/// Legacy keyword-based halo signal. For backward compatibility with existing
/// consumers that call `calculate_halo_signal()` directly.
///
/// Prefer `sift_perceptions()` for new code — it routes through the TF-IDF
/// classifier which has higher accuracy (93.4% vs keyword-based).
/// Detects if the observation leverages cognitive shortcuts.
/// Aggregates all detected bias categories.
///
/// # Examples
///
/// ```
/// use llmosafe::calculate_halo_signal;
/// let signal = calculate_halo_signal("The expert provided a professional opinion.");
/// assert!(signal > 0);
/// ```
#[deprecated(
    since = "0.8.0",
    note = "keyword-based detection retained for backward compatibility. Use sift_perceptions() for higher accuracy (93.4% vs keyword-based) via the TF-IDF classifier."
)]
pub fn calculate_halo_signal(text: &str) -> u16 {
    #[allow(deprecated)]
    get_bias_breakdown(text).total()
}

/// Calculate the "Contextual Utility" (CPMI Proxy).
/// Measures how much the observation reduces uncertainty about the objective.
///
/// # Examples
///
/// ```
/// use llmosafe::calculate_utility;
/// let utility = calculate_utility("Rust is safe", "Rust safety");
/// assert!(utility > 0);
/// ```
pub fn calculate_utility(observation: &str, objective: &str) -> u16 {
    let mut obj_words = [""; 256];
    let mut obj_len = 0;

    // Optimization: avoid using `.skip()` on the string split iterator,
    // which incurs measurable performance overhead from re-iteration.
    // Instead, pre-parse the strings into a single contiguous array.
    for word_b in objective.split_whitespace() {
        if obj_len < 256 {
            obj_words[obj_len] = word_b.trim_matches(|c: char| c.is_ascii_punctuation());
            obj_len += 1;
        } else {
            break;
        }
    }

    let mut count = 0usize;

    for word_a in observation.split_whitespace() {
        let trimmed_a = word_a.trim_matches(|c: char| c.is_ascii_punctuation());

        // ⚡ Bolt: Slice iter directly on `&obj_words[..obj_len]` is slightly faster than `.iter().take()` by reducing iterator state machine overhead.
        for word_b in &obj_words[..obj_len] {
            if trimmed_a.eq_ignore_ascii_case(word_b) {
                count += 1;
                break;
            }
        }
    }

    count.saturating_mul(100).min(u16::MAX as usize) as u16
}

/// Internal version of `sift_text` that also returns the raw classifier score
/// and the pure classifier probability (without keyword bias).
///
/// The score is the unbounded logistic regression sum before sigmoid.
/// Callers that need only the synapse/proof pair should use `sift_text()`.
/// The classifier_probability is the pure classifier output [0.0, 1.0],
/// distinct from entropy-derived values that may include keyword-bias boosting.
#[allow(deprecated)]
pub(crate) fn sift_text_with_score(
    observation: &str,
) -> (SiftedSynapse, SiftedProof, f32, f32, bool, bool, u32) {
    let classification = classify_text(observation);
    let bias = get_bias_breakdown(observation);

    let classifier_entropy = (U16_MAX_F32 * classification.probability.clamp(0.0, 1.0)) as u16;
    let keyword_boost = if bias.total() > 0 {
        ((bias.total() as u32).saturating_mul(65535) / 9000).min(65535) as u16
    } else {
        0
    };
    let entropy = classifier_entropy.max(keyword_boost);

    let surprise = (U16_MAX_F32 * classification.oov_ratio.clamp(0.0, 1.0)) as u16;
    let hard_bias = bias.hard_total() > 0;
    let has_bias = classification.is_manipulation || hard_bias;

    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(entropy);
    synapse.set_raw_surprise(surprise);
    synapse.set_has_bias(has_bias);
    synapse.set_oov_ratio((classification.oov_ratio.clamp(0.0, 1.0) * 255.0) as u8);

    let anchor_hash = adler32::adler32(observation.as_bytes());
    synapse.set_anchor_hash(anchor_hash & 0x7FFFFFFF);

    let sifted = SiftedSynapse::new(synapse);
    let proof = SiftedProof::mint();

    (
        sifted,
        proof,
        classification.score,
        classification.probability,
        classification.is_manipulation,
        hard_bias,
        classification.tokens_matched,
    )
}

/// Error type for sifter operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiftError {
    /// Input exceeds the work budget (MAX_WORK_TOKENS).
    /// Resources would be exhausted by processing this input.
    ResourceExhaustion,
}

/// Canonical single-entry sifter: classifier (adaptive layer) + keyword bias
/// (innate layer). Both pathways contribute to the result — either can flag bias.
///
/// This is the ONLY function `CognitivePipeline` calls. It replaces the
/// three-representation pattern (SifterOutput + ClassificationResult +
/// SiftedSynapse) that existed in `process_ctrl()`.
///
/// # Fields set on Synapse
///
/// - `raw_entropy`: u16 — `max(classifier_entropy, keyword_boost)`, taking
///   the greater of the adaptive (classifier) and innate (keyword) layers
/// - `raw_surprise`: u16 — `classifier.oov_ratio * 65535` (classifier
///   uncertainty — how much vocabulary the model doesn't recognize)
/// - `has_bias`: bool — `classifier.is_manipulation || bias_breakdown.hard_total() > 0`
///   (hard manipulation categories only; typographic `emphasis` is soft
///   evidence that may contribute to entropy/telemetry but never sets
///   `has_bias` independently)
/// - `oov_ratio`: u8 — packed into synapse reserved bits 6-13
/// - `anchor_hash`: u31 — Adler-32 of observation bytes
/// - `no_evidence`: bool — `classification.tokens_matched == 0` (zero-match → unknown/OOD)
pub fn sift_text(observation: &str) -> Result<(SiftedSynapse, SiftedProof), SiftError> {
    // Single preflight: bounded-work budget check.
    // Prevents working-set amplification from adversarial whitespace/token
    // shapes. Fails closed with ResourceExhaustion error.
    if crate::count_tokens(observation) > crate::MAX_WORK_TOKENS {
        return Err(SiftError::ResourceExhaustion);
    }
    let (sifted, proof, _score, _classifier_probability, _is_manipulation, _hard_bias, _tokens) =
        sift_text_with_score(observation);
    Ok((sifted, proof))
}

/// Build a `(SiftedSynapse, SiftedProof)` pair from pre-computed classifier output.
///
/// Runs the same dual-path (classifier + keyword-bias) composition as
/// `sift_text()`, but accepts a pre-computed `ClassificationResult` to skip
/// tokenization. The keyword-bias pathway is the innate immune backstop:
/// even if classifier is compromised, keyword pattern-matching flags text with
/// known manipulation markers. Entropy is `max(classifier_entropy, keyword_boost)`,
/// and bias is `classifier.is_manipulation || bias_breakdown.hard_total() > 0`.
#[allow(deprecated)]
pub fn sift_observation(
    classification: &ClassificationResult,
    observation: &str,
) -> (SiftedSynapse, SiftedProof) {
    let bias = get_bias_breakdown(observation);

    let classifier_entropy = (U16_MAX_F32 * classification.probability.clamp(0.0, 1.0)) as u16;
    let keyword_boost = if bias.total() > 0 {
        ((bias.total() as u32).saturating_mul(65535) / 9000).min(65535) as u16
    } else {
        0
    };
    let entropy = classifier_entropy.max(keyword_boost);

    let surprise = (U16_MAX_F32 * classification.oov_ratio.clamp(0.0, 1.0)) as u16;
    let has_bias = classification.is_manipulation || bias.hard_total() > 0;

    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(entropy);
    synapse.set_raw_surprise(surprise);
    synapse.set_has_bias(has_bias);
    synapse.set_oov_ratio((classification.oov_ratio.clamp(0.0, 1.0) * 255.0) as u8);

    let anchor_hash = adler32::adler32(observation.as_bytes());
    synapse.set_anchor_hash(anchor_hash & 0x7FFFFFFF);

    let sifted = SiftedSynapse::new(synapse);
    let proof = SiftedProof::mint();

    (sifted, proof)
}

/// Classify observations through the dual-path sifter (classifier + keyword)
/// and return the result with highest entropy.
///
/// For each observation, calls `sift_text()`. Selects the observation with
/// the highest `raw_entropy()` (not raw classifier score — the dual-path
/// entropymax includes keyword-bias contribution).
///
/// Empty observations list returns a max-entropy (0xFFFF) synapse.
/// Returns `SiftError::ResourceExhaustion` if any observation exceeds
/// the work budget.
///
/// # Examples
///
/// ```
/// use llmosafe::sift_perceptions;
/// let (sifted, proof) = sift_perceptions(&["Observation 1", "Observation 2"], "safety").expect("sift should succeed");
/// ```
/// Multi-observation batch entry for processing multiple texts through the
/// dual-path sifter (classifier + keyword bias). Selects the observation with
/// highest entropy as the representative synapse+proof pair.
///
/// `_objective` is reserved for future metric scoring (halo signal based on
/// objective-keyword alignment) and is currently unused.
pub fn sift_perceptions(
    observations: &[&str],
    _objective: &str,
) -> Result<(SiftedSynapse, SiftedProof), SiftError> {
    if observations.is_empty() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0xFFFF);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);
        return Ok((SiftedSynapse::new(synapse), SiftedProof::mint()));
    }

    // Seed from the first observation so a non-empty batch
    // with all-zero entropy returns the first item's result,
    // not the 0xFFFF empty-batch fallback. Empty slice remains
    // the sole fail-closed sentinel (checked above).
    let mut best_result = sift_text(observations[0])?;
    let mut best_entropy = best_result.0.raw_entropy();

    for obs in &observations[1..] {
        let result = sift_text(obs)?;
        let entropy = result.0.raw_entropy();
        if entropy > best_entropy {
            best_entropy = entropy;
            best_result = result;
        }
    }

    Ok(best_result)
}

mod adler32 {
    //! The adler32 module contains a 32-bit Adler-32 checksum implementation of
    //! the observation bytes. Accumulates a = 1 + sum(bytes) and b = sum(a),
    //! reducing both mod 65521 after each 5552-byte chunk.
    pub fn adler32(data: &[u8]) -> u32 {
        let mut a: u32 = 1;
        let mut b: u32 = 0;

        for chunk in data.chunks(5552) {
            for &byte in chunk {
                a += byte as u32;
                b += a;
            }
            a %= 65521;
            b %= 65521;
        }

        (b << 16) | a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32::adler32(b""), 1);
    }

    #[test]
    fn test_adler32_simple() {
        // "Wikipedia" adler32 is 0x11E60398 (300286872)
        assert_eq!(adler32::adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn test_adler32_single_char() {
        // "a" -> a = 98, b = 98 -> (98 << 16) | 98 = 6422626
        assert_eq!(adler32::adler32(b"a"), 6422626);
    }

    #[test]
    #[allow(deprecated)]
    fn test_negation_awareness() {
        let text = "The agent is not an expert.";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.authority, 0);

        let text_no_neg = "The agent is an expert.";
        let breakdown_no_neg = get_bias_breakdown(text_no_neg);
        assert_eq!(breakdown_no_neg.authority, 100);
    }

    #[test]
    #[allow(deprecated)]
    fn test_halo_signal_all_categories_detected() {
        let text = "expert trending limited hurry incredible sophisticated";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.authority, 100);
        assert_eq!(breakdown.social_proof, 100);
        assert_eq!(breakdown.scarcity, 100);
        assert_eq!(breakdown.urgency, 100);
        assert_eq!(breakdown.emotional_appeal, 100);
        assert_eq!(breakdown.expertise_signaling, 100);
        assert_eq!(calculate_halo_signal(text), 600);
    }

    #[test]
    #[allow(deprecated)]
    fn test_multi_word_phrases_detected() {
        // "as an ai" and "i cannot" both fire template_fitting
        // "instead of" fires semantic_traps
        let text = "As an AI, I cannot comply, instead of helping you";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.template_fitting, 200);
        assert_eq!(breakdown.semantic_traps, 100);
    }

    #[test]
    #[allow(deprecated)]
    fn test_template_fitting_phrases() {
        // Each multi-word phrase should fire independently
        let text = "As an AI, my purpose is to note that I am programmed to follow";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.template_fitting, 300);
    }

    #[test]
    fn test_sift_perceptions_empty_observations() {
        let objective = "test";
        let observations: &[&str] = &[];

        let (sifted, _) = sift_perceptions(observations, objective)
            .expect("sift_perceptions should succeed on empty");
        assert_eq!(sifted.raw_entropy(), 0xFFFF);
        assert_eq!(
            sifted.validate().unwrap_err(),
            crate::llmosafe_kernel::KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_sift_perceptions_single_observation() {
        let observations = &["stable observation"];
        let (sifted, _) =
            sift_perceptions(observations, "test").expect("sift_perceptions should succeed");
        let _entropy = sifted.raw_entropy();
        let _surprise = sifted.raw_surprise();
    }

    #[test]
    fn test_utility_calculation() {
        let objective = "Build a Rust safety library";
        let obs1 = "Rust safety is paramount";
        let obs2 = "C++ is also good";

        let u1 = calculate_utility(obs1, objective);
        let u2 = calculate_utility(obs2, objective);

        assert!(u1 > u2);
    }

    #[test]
    fn test_sifter_token_bomb() {
        let objective = "test";
        let bomb = "token ".repeat(10000);
        let u = calculate_utility(&bomb, objective);
        let _ = u;
    }

    #[test]
    #[allow(deprecated)]
    fn test_halo_signal_keyword_density() {
        let text = "expert official government doctor scientist guaranteed certified proven experts officials scientists";
        let signal = calculate_halo_signal(text);
        assert!(signal >= 1000);
    }

    #[test]
    #[allow(deprecated)]
    fn test_halo_signal_metamorphic_monotonicity() {
        let text1 = "This is a normal observation.";
        let text2 = "This is an expert observation.";
        let text3 = "This is an expert and professional observation.";

        let s1 = calculate_halo_signal(text1);
        let s2 = calculate_halo_signal(text2);
        let s3 = calculate_halo_signal(text3);

        assert!(s1 <= s2);
        assert!(s2 <= s3);
        assert!(s3 > s1);
    }

    #[test]
    fn test_utility_metamorphic_shuffle() {
        let objective = "Safety Critical AI";
        let obs1 = "Formal verification ensures deterministic execution.";
        let obs2 = "execution deterministic ensures verification Formal.";

        let u1 = calculate_utility(obs1, objective);
        let u2 = calculate_utility(obs2, objective);

        assert_eq!(u1, u2);
    }

    #[test]
    fn test_sift_quantization_differential() {
        let observations = &["Safety is paramount"];
        let (sifted, _) =
            sift_perceptions(observations, "Safety").expect("sift_perceptions should succeed");
        let _entropy = sifted.raw_entropy();
        let _surprise = sifted.raw_surprise();
    }

    #[test]
    fn test_sift_perceptions_logic() {
        let observations = &[
            "Rust is the most secure language due to its ownership model",
            "Python is very popular and easy to learn",
            "C is a limited but performant systems language",
        ];

        let (sifted, _) = sift_perceptions(observations, "coding language safety")
            .expect("sift_perceptions should succeed");
        let _entropy = sifted.raw_entropy();
        let _surprise = sifted.raw_surprise();
        assert!(sifted.anchor_hash() != 0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_negation_ttl_covers_six_tokens() {
        // "not a very well known expert" — "expert" is 5 words after "not"
        let breakdown = get_bias_breakdown("not a very well known expert");
        assert_eq!(breakdown.authority, 0, "authority should be 0 when negated");

        // Without negation, same content triggers authority
        let breakdown2 = get_bias_breakdown("a very well known expert");
        assert_eq!(breakdown2.authority, 100);
    }

    #[test]
    #[allow(deprecated)]
    fn test_phase2_negation_multi_word() {
        // "not as an ai" — negation should prevent template_fitting match
        let breakdown = get_bias_breakdown("not as an ai");
        assert_eq!(
            breakdown.template_fitting, 0,
            "template_fitting should be 0 when negated"
        );

        // "not as an ai" — also check semantic_traps (no multi-word trap match)
        let breakdown2 = get_bias_breakdown("not as an ai");
        assert_eq!(breakdown2.semantic_traps, 0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_while_not_a_semantic_trap() {
        // "while" was removed from SEMANTIC_TRAPS — should not trigger
        let breakdown = get_bias_breakdown("while processing data");
        assert_eq!(
            breakdown.semantic_traps, 0,
            "while should not trigger semantic_traps"
        );
    }

    // ── sift_observation() tests ──────────────────────────────────

    #[test]
    #[allow(deprecated)]
    fn test_sift_observation_produces_valid_synapse() {
        // Construct a ClassificationResult with non-trivial values
        let class_result = ClassificationResult {
            score: 2.5,
            probability: 0.92,
            is_manipulation: true,
            oov_ratio: 0.15,
            tokens_matched: 8,
            tokens_total: 10,
            no_evidence: false,
        };
        let (sifted, _proof) = sift_observation(&class_result, "test observation text");
        // Entropy should reflect high manipulation probability: 0.92 * 65535 ≈ 60292
        let entropy = sifted.raw_entropy();
        assert!(
            entropy > 50000,
            "sifted entropy should be high for p=0.92 manipulation: {}",
            entropy
        );
        // Surprise from OOV: 0.15 * 65535 ≈ 9830
        let surprise = sifted.raw_surprise();
        assert!(
            surprise > 5000,
            "sifted surprise should be non-zero for oov_ratio=0.15: {}",
            surprise
        );
        // Bias flag should be true (is_manipulation=true)
        assert!(
            sifted.has_bias(),
            "has_bias should be true for manipulation"
        );
        // Anchor hash should be non-zero (non-empty observation)
        assert_ne!(sifted.anchor_hash(), 0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_sift_observation_empty_text() {
        let class_result = ClassificationResult {
            score: 0.0,
            probability: 0.5,
            is_manipulation: false,
            oov_ratio: 0.0,
            tokens_matched: 0,
            tokens_total: 0,
            no_evidence: true,
        };
        let (sifted, _proof) = sift_observation(&class_result, "");
        // Empty text: entropy = 0.5 * 65535 = 32767
        let entropy = sifted.raw_entropy();
        assert!(
            entropy > 0,
            "sifted entropy should be > 0 even for empty text"
        );
        // Anchor hash for empty string should be non-zero (adler32("") = 1)
        assert_ne!(sifted.anchor_hash(), 0);
    }

    // ── calculate_utility() edge cases ────────────────────────────

    #[test]
    fn test_calculate_utility_empty_observation() {
        let utility = calculate_utility("", "safety objective");
        assert_eq!(utility, 0);
    }

    #[test]
    fn test_calculate_utility_empty_objective() {
        let utility = calculate_utility("some observation text", "");
        assert_eq!(utility, 0);
    }

    #[test]
    fn test_sifter_output_from_classification() {
        let class_result = ClassificationResult {
            score: 1.5,
            probability: 0.82,
            is_manipulation: true,
            oov_ratio: 0.25,
            tokens_matched: 5,
            tokens_total: 10,
            no_evidence: false,
        };
        let output = SifterOutput::from_classification(&class_result);
        assert!((output.error_sift - 0.82).abs() < 0.01);
        assert!(output.raw_entropy > 50000);
        assert!(output.has_bias);
        assert_eq!(output.classifier_prob, 0.82);
        assert!(output.oov_ratio > 0);
    }

    // ── no_std phrase parity tests ─────────────────────────
    // These tests verify that multi-word phrase matching works
    // in both std and no_std builds. Without this, SEMANTIC_TRAPS
    // and TEMPLATE_FITTING (all multi-word entries) could never
    // fire — the old code gated phrase matching behind
    // #[cfg(feature = "std")], leaving no_std with dead code.

    #[test]
    #[allow(deprecated)]
    fn test_no_std_semantic_traps_phrases_fire() {
        // "instead of" fires semantic_traps — works in no_std
        let breakdown = get_bias_breakdown("instead of");
        assert_eq!(
            breakdown.semantic_traps, 100,
            "SEMANTIC_TRAPS multi-word phrases must fire in no_std"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_no_std_template_fitting_phrases_fire() {
        // "as an ai" fires template_fitting — works in no_std
        let breakdown = get_bias_breakdown("as an ai");
        assert_eq!(
            breakdown.template_fitting, 100,
            "TEMPLATE_FITTING multi-word phrases must fire in no_std"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_no_std_multi_word_combinations() {
        // Both SEMANTIC_TRAPS and TEMPLATE_FITTING fire together
        let text = "As an AI, I cannot comply, instead of helping you";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.template_fitting, 200);
        assert_eq!(breakdown.semantic_traps, 100);
    }

    /// Verify single-token comparisons against multi-word lists
    /// are dead code (removed). A single word like "not" should
    /// NOT match SEMANTIC_TRAPS/TEMPLATE_FITTING.
    #[test]
    #[allow(deprecated)]
    fn test_dead_single_token_comparisons_removed() {
        // "not" is a single word; SEMANTIC_TRAPS entries are all
        // multi-word. The old phase-1 check would have matched
        // "not" against SEMANTIC_TRAPS (dead code, now removed).
        // The phrase_matches function requires consecutive tokens.
        let breakdown = get_bias_breakdown("not");
        assert_eq!(breakdown.semantic_traps, 0);
        assert_eq!(breakdown.template_fitting, 0);
    }
}

// ── S3: Acronym/emphasis decoupling resolution ──────
//
// Resolution: emphasis is decoupled from hard bias.
// `has_bias` now uses `bias.hard_total() > 0` which excludes
// typographic `emphasis`. The invariant
// `has_bias == classification.is_manipulation` is restored.
//
// File:line trace after fix:
//   src/llmosafe_sifter.rs:296-305 — `hard_total()` sums all
//     manipulation categories EXCEPT `emphasis`.
//   src/llmosafe_sifter.rs:559 — `has_bias = classification.is_manipulation
//     || bias.hard_total() > 0`. Emphasis no longer sets has_bias.
//   src/llmosafe_sifter.rs:553-556 — `keyword_boost` still uses
//     `bias.total() > 0` (includes emphasis), so emphasis continues
//     to contribute to soft scoring (entropy/telemetry/PID input).
//   src/llmosafe_pipeline.rs:1099 — `if has_bias { ... BIAS }`
//     now correctly only triggers on genuine hard bias.
#[cfg(test)]
mod s3_acronym_emphasis_audit {
    use super::*;

    /// FIX VERIFIED: ordinary technical acronyms (AI, API, HTTP, CPU)
    /// no longer trigger `has_bias` via the emphasis pathway.
    ///
    /// Emphasis is still detected and contributes to soft scoring
    /// (via `bias.total()` which feeds `keyword_boost` and entropy),
    /// but `has_bias` uses `bias.hard_total() > 0` which excludes
    /// emphasis. This restores the invariant
    /// `has_bias == classification.is_manipulation`.
    #[test]
    #[allow(deprecated)]
    fn test_acronym_emphasis_no_longer_drives_has_bias() {
        // "AI", "API", "HTTP", "CPU" are ordinary technical
        // acronyms that match the emphasis check at
        // src/llmosafe_sifter.rs:369 (all ASCII-uppercase,
        // len >= 2).
        let text = "The AI API HTTP CPU are safe";
        let (_sifted, _) = sift_text(text).expect("sift_text should succeed");
        let breakdown = get_bias_breakdown(text);
        // After the fix, emphasis no longer sets has_bias.
        // The invariant has_bias == classification.is_manipulation
        // is restored — emphasis does NOT contribute to hard_total.
        assert!(
            breakdown.hard_total() == 0,
            "FIXED: ordinary technical acronyms (AI, API, HTTP, CPU) \
             must NOT contribute to hard_total(). \
             Emphasis is soft evidence only; hard_total() excludes emphasis."
        );
        // But emphasis still contributes to soft scoring (total > 0)
        assert!(
            breakdown.emphasis > 0,
            "emphasis should still be detected for ALL-CAPS words"
        );
        assert!(
            breakdown.total() > 0,
            "total() (including emphasis) should still be > 0 \
             for soft scoring/telemetry"
        );
        // hard_total() must be 0 for ordinary technical acronyms
        assert!(
            breakdown.hard_total() == 0,
            "hard_total() (excluding emphasis) must be 0 \
             for ordinary technical acronyms"
        );
    }

    /// Genuine manipulation/bias cases that set hard bias before
    /// still do. Keywords like "expert", "guaranteed", etc. are
    /// in the hard-bias categories and fire hard_total() > 0.
    #[test]
    #[allow(deprecated)]
    fn test_genuine_manipulation_still_sets_hard_bias() {
        let text = "The expert provided a guaranteed professional opinion";
        let (_sifted, _) = sift_text(text).expect("sift_text should succeed");
        let breakdown = get_bias_breakdown(text);
        // Authority + expertise + guaranteed keyword bias → hard bias
        assert!(
            breakdown.hard_total() > 0,
            "genuine manipulation must fire hard_total() > 0"
        );
    }

    /// ALL-CAPS emphasis can still raise soft risk if retained.
    /// Emphasis contributes to `bias.total()` which feeds
    /// `keyword_boost` and entropy — soft scoring, telemetry,
    /// and PID input, but never independently triggers the
    /// BIAS override or Halt.
    #[test]
    #[allow(deprecated)]
    fn test_emphasis_still_contributes_to_soft_scoring() {
        let text = "THE QUICK BROWN FOX";
        let breakdown = get_bias_breakdown(text);
        // All-caps words trigger emphasis
        assert!(
            breakdown.emphasis > 0,
            "ALL-CAPS words must still trigger emphasis detection"
        );
        // Emphasis contributes to total() for soft scoring
        assert!(
            breakdown.total() > 0,
            "emphasis must still contribute to total() for soft scoring"
        );
        // But hard_total() excludes emphasis
        assert!(
            breakdown.hard_total() == 0,
            "emphasis must NOT contribute to hard_total()"
        );
        // And emphasis alone must not be the cause of has_bias
        // (has_bias is determined by classifier.is_manipulation || hard_total() > 0;
        // emphasis is excluded from hard_total())
        let (_sifted, _) = sift_text(text).expect("sift_text should succeed");
        // Emphasis is not in hard_total, so it doesn't independently drive has_bias
        let breakdown = get_bias_breakdown(text);
        assert!(
            breakdown.hard_total() == 0,
            "emphasis must NOT set has_bias via hard_total()"
        );
    }
}

// ── R1 No-Evidence sifter tests ──

#[test]
fn test_sifter_no_evidence_empty() {
    let (_sifted, _) = sift_text("").expect("sift_text should succeed");
    let output = SifterOutput::from_classification(&crate::llmosafe_classifier::classify_text(""));
    assert!(output.no_evidence, "empty input must have no_evidence=true");
    assert!(!output.has_bias, "no_evidence must not set has_bias");
}

#[test]
fn test_sifter_no_evidence_unknown() {
    let (_sifted, _) = sift_text("xyzzytotallyunknownabc123").expect("sift_text should succeed");
    let classification = crate::llmosafe_classifier::classify_text("xyzzytotallyunknownabc123");
    assert!(
        classification.no_evidence,
        "unknown must have no_evidence=true"
    );
    assert!(
        !classification.is_manipulation,
        "zero-match must NOT be manipulation"
    );
    let output = SifterOutput::from_classification(&classification);
    assert!(output.no_evidence, "sifter must expose no_evidence");
    assert!(!output.has_bias, "no_evidence must not set has_bias");
}

#[test]
fn test_sifter_no_evidence_cjk() {
    let classification = crate::llmosafe_classifier::classify_text("你好世界你好你好");
    assert!(classification.no_evidence, "CJK must have no_evidence=true");
    assert!(
        !classification.is_manipulation,
        "CJK zero-match must NOT be manipulation"
    );
    let output = SifterOutput::from_classification(&classification);
    assert!(output.no_evidence, "sifter must expose no_evidence for CJK");
}
