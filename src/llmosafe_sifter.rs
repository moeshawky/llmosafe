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
///   `sift_text()` OR-s the keyword-bias breakdown into its output)
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
        let entropy = (65535.0_f32 * classification.probability.clamp(0.0, 1.0)) as u16;
        Self {
            error_sift: classification.probability,
            raw_entropy: entropy,
            classifier_prob: classification.probability,
            has_bias: classification.is_manipulation,
            oov_ratio: (classification.oov_ratio * 255.0_f32) as u8,
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
    /// Total bias score across all categories.
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
}

/// Case-insensitive keyword match without allocation.
#[inline]
fn word_in_list(word: &str, list: &[&str]) -> bool {
    list.iter().any(|kw| word.eq_ignore_ascii_case(kw))
}

/// Check if consecutive tokens match a multi-word phrase.
#[cfg(feature = "std")]
#[inline]
fn phrase_matches(window: &[&str], phrase: &str) -> bool {
    let phrase_len = phrase.split_whitespace().count();
    if window.len() < phrase_len {
        return false;
    }
    window[..phrase_len]
        .iter()
        .zip(phrase.split_whitespace())
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Returns a breakdown of detected biases by category.
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
        if word_in_list(trimmed, SEMANTIC_TRAPS) {
            breakdown.semantic_traps = breakdown.semantic_traps.saturating_add(100);
        }
        if word_in_list(trimmed, TEMPLATE_FITTING) {
            breakdown.template_fitting = breakdown.template_fitting.saturating_add(100);
        }

        // Attention-emphasis signal: ALL CAPS words (len >= 2) indicate
        // typographic manipulation independent of keyword membership.
        // Only fires on ASCII-uppercase — excludes emoji, Unicode scripts,
        // and camelCase/PascalCase (technical notation, not manipulation).
        if trimmed.len() >= 2 && trimmed.chars().all(|c| c.is_ascii_uppercase()) {
            breakdown.emphasis = breakdown.emphasis.saturating_add(50);
        }
    }

    // Phase 2: Multi-word phrase matching (for entries containing spaces).
    // Requires `std` for Vec allocation. no_std users get single-word detection only.
    #[cfg(feature = "std")]
    {
        let tokens: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
            .collect();

        let mut negated_positions = vec![false; tokens.len()];
        let mut neg_ttl = 0u8;
        for (i, token) in tokens.iter().enumerate() {
            let is_neg = word_in_list(token, NEGATION_WORDS);
            let curr_negated = neg_ttl > 0;
            if is_neg {
                neg_ttl = 6;
            } else {
                neg_ttl = neg_ttl.saturating_sub(1);
            }
            negated_positions[i] = curr_negated;
        }

        for phrase in SEMANTIC_TRAPS {
            if !phrase.contains(' ') {
                continue;
            }
            if tokens
                .windows(phrase.split_whitespace().count())
                .enumerate()
                .any(|(i, w)| !negated_positions[i] && phrase_matches(w, phrase))
            {
                breakdown.semantic_traps = breakdown.semantic_traps.saturating_add(100);
            }
        }

        for phrase in TEMPLATE_FITTING {
            if !phrase.contains(' ') {
                continue;
            }
            if tokens
                .windows(phrase.split_whitespace().count())
                .enumerate()
                .any(|(i, w)| !negated_positions[i] && phrase_matches(w, phrase))
            {
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
pub fn calculate_halo_signal(text: &str) -> u16 {
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
    let mut obj_words = [""; 128];
    let mut obj_len = 0;

    for word_b in objective.split_whitespace() {
        if obj_len < 128 {
            obj_words[obj_len] = word_b.trim_matches(|c: char| c.is_ascii_punctuation());
            obj_len += 1;
        } else {
            break;
        }
    }

    calculate_utility_with_cache(observation, objective, &obj_words, obj_len)
}

fn calculate_utility_with_cache(
    observation: &str,
    objective: &str,
    obj_words: &[&str],
    obj_len: usize,
) -> u16 {
    let mut excess_words = [""; 128];
    let mut excess_len = 0;
    for word_b in objective.split_whitespace().skip(obj_len) {
        if excess_len < 128 {
            excess_words[excess_len] =
                word_b.trim_matches(|c: char| c.is_ascii_punctuation());
            excess_len += 1;
        } else {
            break;
        }
    }

    let mut count = 0usize;

    for word_a in observation.split_whitespace() {
        let trimmed_a = word_a.trim_matches(|c: char| c.is_ascii_punctuation());

        let mut found = false;
        for word_b in obj_words.iter().take(obj_len) {
            if trimmed_a.eq_ignore_ascii_case(word_b) {
                count += 1;
                found = true;
                break;
            }
        }

        if !found && excess_len > 0 {
            for word_b in excess_words.iter().take(excess_len) {
                if trimmed_a.eq_ignore_ascii_case(word_b) {
                    count += 1;
                    break;
                }
            }
        }
    }

    count.saturating_mul(100).min(u16::MAX as usize) as u16
}

/// Internal version of `sift_text` that also returns the raw classifier score.
///
/// The score is the unbounded logistic regression sum before sigmoid.
/// Callers that need only the synapse/proof pair should use `sift_text()`.
pub(crate) fn sift_text_with_score(
    observation: &str,
) -> (SiftedSynapse, SiftedProof, f32) {
    let classification = classify_text(observation);
    let bias = get_bias_breakdown(observation);

    let classifier_entropy =
        (65535.0_f32 * classification.probability.clamp(0.0, 1.0)) as u16;
    let keyword_boost = if bias.total() > 0 {
        ((bias.total() as u32).saturating_mul(65535) / 9000).min(65535) as u16
    } else {
        0
    };
    let entropy = classifier_entropy.max(keyword_boost);

    let surprise = (65535.0_f32 * classification.oov_ratio.clamp(0.0, 1.0)) as u16;
    let has_bias = classification.is_manipulation || bias.total() > 0;

    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(entropy);
    synapse.set_raw_surprise(surprise);
    synapse.set_has_bias(has_bias);
    synapse.set_oov_ratio((classification.oov_ratio.clamp(0.0, 1.0) * 255.0) as u8);

    let anchor_hash = adler32::adler32(observation.as_bytes());
    synapse.set_anchor_hash(anchor_hash & 0x7FFFFFFF);

    let sifted = SiftedSynapse::new(synapse);
    let proof = SiftedProof::mint();

    (sifted, proof, classification.score)
}

/// Canonical single-entry sifter: classifier (adaptive layer) + keyword bias
/// (innate layer). Both pathways contribute to the result — either can flag bias.
///
/// This is the ONLY function `CognitivePipeline` calls. It replaces the
/// three-representation pattern (SifterOutput + ClassificationResult +
/// SiftedSynapse) that existed in `process_ctrl()`.
///
/// The keyword-bias pathway is the innate immune layer: fast pattern-matching
/// against known manipulation markers. It runs on every input and OR-s into
/// the bias flag. It stays as a separately-auditable module — if the classifier
/// is ever compromised by adversarial ML, the keyword path provides a
/// backstop.
///
/// # Fields set on Synapse
///
/// - `raw_entropy`: u16 — `max(classifier_entropy, keyword_boost)`, taking
///   the greater of the adaptive (classifier) and innate (keyword) layers
/// - `raw_surprise`: u16 — `classifier.oov_ratio * 65535` (classifier
///   uncertainty — how much vocabulary the model doesn't recognize)
/// - `has_bias`: bool — `classifier.is_manipulation || bias_breakdown.total() > 0`
/// - `oov_ratio`: u8 — packed into synapse reserved bits 6-13
/// - `anchor_hash`: u31 — Adler-32 of observation bytes
pub fn sift_text(observation: &str) -> (SiftedSynapse, SiftedProof) {
    let (sifted, proof, _score) = sift_text_with_score(observation);
    (sifted, proof)
}

/// Build a `(SiftedSynapse, SiftedProof)` pair from pre-computed classifier output.
///
/// Runs the same dual-path (classifier + keyword-bias) composition as
/// `sift_text()`, but accepts a pre-computed `ClassificationResult` to skip
/// tokenization. The keyword-bias pathway is the innate immune backstop:
/// even if classifier is compromised, keyword pattern-matching flags text with
/// known manipulation markers. Entropy is `max(classifier_entropy, keyword_boost)`,
/// and bias is `classifier.is_manipulation || bias_breakdown.total() > 0`.
pub fn sift_observation(
    classification: &ClassificationResult,
    observation: &str,
) -> (SiftedSynapse, SiftedProof) {
    let bias = get_bias_breakdown(observation);

    let classifier_entropy =
        (65535.0_f32 * classification.probability.clamp(0.0, 1.0)) as u16;
    let keyword_boost = if bias.total() > 0 {
        ((bias.total() as u32).saturating_mul(65535) / 9000).min(65535) as u16
    } else {
        0
    };
    let entropy = classifier_entropy.max(keyword_boost);

    let surprise = (65535.0_f32 * classification.oov_ratio.clamp(0.0, 1.0)) as u16;
    let has_bias = classification.is_manipulation || bias.total() > 0;

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
///
/// # Examples
///
/// ```
/// use llmosafe::sift_perceptions;
/// let (sifted, proof) = sift_perceptions(&["Observation 1", "Observation 2"], "safety");
/// ```
/// Multi-observation batch entry for processing multiple texts through the
/// dual-path sifter (classifier + keyword bias). Selects the observation with
/// highest entropy as the representative synapse+proof pair.
pub fn sift_perceptions(observations: &[&str], _objective: &str) -> (SiftedSynapse, SiftedProof) {
    if observations.is_empty() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0xFFFF);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);
        return (SiftedSynapse::new(synapse), SiftedProof::mint());
    }

    let mut best_entropy: u16 = 0;
    let mut best_result: Option<(SiftedSynapse, SiftedProof)> = None;

    for obs in observations {
        let result = sift_text(obs);
        let entropy = result.0.raw_entropy();
        if entropy > best_entropy {
            best_entropy = entropy;
            best_result = Some(result);
        }
    }

    best_result.unwrap_or_else(|| {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0xFFFF);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);
        (SiftedSynapse::new(synapse), SiftedProof::mint())
    })
}

mod adler32 {
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
    fn test_negation_awareness() {
        let text = "The agent is not an expert.";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.authority, 0);

        let text_no_neg = "The agent is an expert.";
        let breakdown_no_neg = get_bias_breakdown(text_no_neg);
        assert_eq!(breakdown_no_neg.authority, 100);
    }

    #[test]
    fn test_halo_signal() {
        assert_eq!(
            calculate_halo_signal("The lead expert is certified and official."),
            300
        );
        assert_eq!(
            calculate_halo_signal("This is a random sentence without flags."),
            0
        );
        assert_eq!(calculate_halo_signal("limited and exclusive rare"), 300);
    }

    #[test]
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
    fn test_multi_word_phrases_detected() {
        // "as an ai" and "i cannot" both fire template_fitting
        // "instead of" fires semantic_traps
        let text = "As an AI, I cannot comply, instead of helping you";
        let breakdown = get_bias_breakdown(text);
        assert_eq!(breakdown.template_fitting, 200);
        assert_eq!(breakdown.semantic_traps, 100);
    }

    #[test]
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

        let (sifted, _) = sift_perceptions(observations, objective);
        assert_eq!(sifted.raw_entropy(), 0xFFFF);
        assert_eq!(
            sifted.validate().unwrap_err(),
            crate::llmosafe_kernel::KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_sift_perceptions_single_observation() {
        let observations = &["stable observation"];
        let (sifted, _) = sift_perceptions(observations, "test");
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
    fn test_halo_signal_keyword_density() {
        let text = "expert official government doctor scientist guaranteed certified proven experts officials scientists";
        let signal = calculate_halo_signal(text);
        assert!(signal >= 1000);
    }

    #[test]
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
        let (sifted, _) = sift_perceptions(observations, "Safety");
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

        let (sifted, _) = sift_perceptions(observations, "coding language safety");
        let _entropy = sifted.raw_entropy();
        let _surprise = sifted.raw_surprise();
        assert!(sifted.anchor_hash() != 0);
    }

    #[test]
    fn test_negation_ttl_covers_six_tokens() {
        // "not a very well known expert" — "expert" is 5 words after "not"
        let breakdown = get_bias_breakdown("not a very well known expert");
        assert_eq!(breakdown.authority, 0, "authority should be 0 when negated");

        // Without negation, same content triggers authority
        let breakdown2 = get_bias_breakdown("a very well known expert");
        assert_eq!(breakdown2.authority, 100);
    }

    #[test]
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
    fn test_while_not_a_semantic_trap() {
        // "while" was removed from SEMANTIC_TRAPS — should not trigger
        let breakdown = get_bias_breakdown("while processing data");
        assert_eq!(
            breakdown.semantic_traps, 0,
            "while should not trigger semantic_traps"
        );
    }
}
