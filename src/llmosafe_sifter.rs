//! LLMOSAFE Tier 3 Perceptual Sifter (Rust Implementation)
//!
//! This module implements the "Perceptual Decoupling" Axiom.
//! It acts as a 'Thinking Proxy' (MemSifter pattern) that sifts external
//! reality into high-signal anchors for the Tier 1/2 safety core.
//!
//! Research Grounds:
//! - MemSifter: Think-and-Rank proxy reasoning.
//! - Selective Attention: Bias/Halo screening (SCS).
//! - GDWM: Task-Contextual Utility (CPMI).

use crate::llmosafe_kernel::SiftedSynapse;
use crate::llmosafe_kernel::Synapse;

/// Authority Bias Keywords: Red flags for expertise/position bias.
pub const AUTHORITY_BIAS: &[&str] = &[
    "expert",
    "official",
    "government",
    "doctor",
    "professor",
    "scientist",
    "research",
    "study",
    "guaranteed",
    "certified",
    "authorized",
    "proven",
    "reliable",
    "trusted",
    "professional",
    "specialist",
    "veteran",
    "master",
    "guru",
    "authority",
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
pub const SOCIAL_PROOF: &[&str] = &[
    "popular",
    "everyone",
    "thousands",
    "millions",
    "trending",
    "bestseller",
    "viral",
    "community",
    "joined",
    "users",
    "testimonials",
    "reviews",
    "ratings",
    "common",
    "standard",
    "consensus",
    "majority",
    "crowd",
    "peer",
    "social",
];

/// Scarcity Keywords: Red flags for restricted availability bias.
pub const SCARCITY: &[&str] = &[
    "limited",
    "rare",
    "exclusive",
    "only",
    "few",
    "unique",
    "special",
    "handcrafted",
    "small-batch",
    "collectible",
    "once-in-a-lifetime",
    "select",
    "restricted",
    "shortage",
    "vanishing",
    "low-stock",
    "while-supplies-last",
    "sold-out",
    "member-only",
    "private",
];

/// Urgency Keywords: Red flags for time-pressure bias.
pub const URGENCY: &[&str] = &[
    "now",
    "today",
    "fast",
    "quick",
    "instant",
    "hurry",
    "rush",
    "deadline",
    "expiring",
    "ending",
    "soon",
    "immediately",
    "pronto",
    "rapid",
    "speedy",
    "limited-time",
    "last-chance",
    "act-now",
    "don't-wait",
    "final",
];

/// Emotional Appeal Keywords: Red flags for emotional manipulation bias.
pub const EMOTIONAL_APPEAL: &[&str] = &[
    "love",
    "joy",
    "fear",
    "worry",
    "happy",
    "sad",
    "angry",
    "exciting",
    "amazing",
    "beautiful",
    "heartwarming",
    "shocking",
    "miracle",
    "incredible",
    "tragic",
    "desperate",
    "hopeful",
    "passionate",
    "inspiring",
    "emotional",
    "appealing",
];

/// Expertise Signaling Keywords: Red flags for jargon/complexity bias.
pub const EXPERTISE_SIGNALING: &[&str] = &[
    "sophisticated",
    "advanced",
    "cutting-edge",
    "state-of-the-art",
    "revolutionary",
    "innovative",
    "patented",
    "breakthrough",
    "proprietary",
    "complex",
    "technical",
    "paradigm",
    "holistic",
    "synergy",
    "leverage",
    "optimize",
    "agile",
    "lean",
    "scalable",
    "high-performance",
];

/// Semantic Traps: Inversion patterns that flip safety predicates.
pub const SEMANTIC_TRAPS: &[&str] = &[
    "not but",
    "instead of",
    "rather than",
    "unless",
    "however",
    "conversely",
    "on the other hand",
    "despite",
    "although",
    "while",
];

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
/// Each field corresponds to one of the 8 bias categories.
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
    }
}

/// Case-insensitive keyword match without allocation.
#[inline]
fn word_in_list(word: &str, list: &[&str]) -> bool {
    list.iter().any(|kw| word.eq_ignore_ascii_case(kw))
}

/// Returns a breakdown of detected biases by category.
pub fn get_bias_breakdown(text: &str) -> BiasBreakdown {
    let mut breakdown = BiasBreakdown::default();

    let mut negation_ttl = 0;

    for raw_word in text.split_whitespace() {
        let trimmed = raw_word.trim_matches(|c: char| c.is_ascii_punctuation());

        let is_negated = negation_ttl > 0;

        if word_in_list(trimmed, NEGATION_WORDS) {
            negation_ttl = 3;
        } else if negation_ttl > 0 {
            negation_ttl -= 1;
        }

        if is_negated {
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
    }

    breakdown
}

/// Calculate the "Halo Signal" (SCS Proxy).
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
    let mut count = 0usize;

    for word_a in observation.split_whitespace() {
        let trimmed_a = word_a.trim_matches(|c: char| c.is_ascii_punctuation());
        for word_b in objective.split_whitespace() {
            let trimmed_b = word_b.trim_matches(|c: char| c.is_ascii_punctuation());
            if trimmed_a.eq_ignore_ascii_case(trimmed_b) {
                count += 1;
                break;
            }
        }
    }

    count.saturating_mul(100).min(u16::MAX as usize) as u16
}

/// Sift Perceptions (Think-and-Rank).
/// Converts a list of raw observations into a single Synapse spike.
/// Returns a high-entropy Synapse if no observations are provided.
///
/// # Examples
///
/// ```
/// use llmosafe::sift_perceptions;
/// let objective = "Safety";
/// let observations = &["Observation 1", "Observation 2"];
/// let sifted = sift_perceptions(observations, objective);
/// ```
pub fn sift_perceptions(observations: &[&str], objective: &str) -> SiftedSynapse {
    if observations.is_empty() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0xFFFF);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);
        return SiftedSynapse::new(synapse);
    }

    let mut best_obs: &str = "";
    let mut best_score: i32 = i32::MIN;
    let mut total_score: i64 = 0;

    for obs in observations {
        let utility = calculate_utility(obs, objective);
        let halo = calculate_halo_signal(obs);
        let score = (utility as i32) - (halo as i32);
        total_score += score as i64;
        if score > best_score {
            best_score = score;
            best_obs = obs;
        }
    }

    let mean_score = total_score / (observations.len() as i64);

    let entropy = 1000i32.saturating_sub(best_score);
    let surprise = (best_score as i64 - mean_score).unsigned_abs() as u16;
    let has_bias = calculate_halo_signal(best_obs) > 0;

    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(entropy.clamp(0, 0xFFFF) as u16);
    synapse.set_raw_surprise(surprise);
    synapse.set_has_bias(has_bias);

    let anchor_hash = adler32::adler32(best_obs.as_bytes());
    synapse.set_anchor_hash(anchor_hash & 0x7FFFFFFF);

    SiftedSynapse::new(synapse)
}

mod adler32 {
    pub fn adler32(data: &[u8]) -> u32 {
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
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
            calculate_halo_signal("The lead expert is professional and official."),
            300
        );
        assert_eq!(
            calculate_halo_signal("This is a random sentence without flags."),
            0
        );
        assert_eq!(calculate_halo_signal("limited and exclusive special"), 300);
    }

    #[test]
    fn test_halo_signal_all_categories_detected() {
        let text = "expert popular limited now love sophisticated";
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
    fn test_sift_perceptions_empty_observations() {
        let objective = "test";
        let observations: &[&str] = &[];

        let sifted = sift_perceptions(observations, objective);
        assert_eq!(sifted.raw_entropy(), 0xFFFF);
        assert_eq!(
            sifted.validate().unwrap_err(),
            crate::llmosafe_kernel::KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_sift_perceptions_single_observation() {
        let objective = "test";
        let observations = &["stable observation"];
        let sifted = sift_perceptions(observations, objective);
        assert!(sifted.validate().is_ok());
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
        let text = "expert professional authorized reliable trusted specialist veteran master guru authority";
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
        let objective = "Safety";
        let observations = &["Safety is paramount"];

        let sifted = sift_perceptions(observations, objective);
        assert_eq!(sifted.raw_entropy(), 900);
    }

    #[test]
    fn test_sift_perceptions_logic() {
        let objective = "Identify the best coding language for safety";
        let observations = &[
            "Rust is the most secure language due to its ownership model",
            "Python is very popular and easy to learn",
            "C is exclusive because it is limited but unsafe",
        ];

        let sifted = sift_perceptions(observations, objective);

        assert!(sifted.validate().is_ok());
        assert!(sifted.anchor_hash() != 0);
    }
}
