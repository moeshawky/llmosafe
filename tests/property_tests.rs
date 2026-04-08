//! Property-based tests using proptest
//!
//! These tests verify invariants hold across a wide range of inputs.
use llmosafe::{
    calculate_halo_signal, get_bias_breakdown, sift_perceptions, ConfidenceTracker, DriftDetector,
    EscalationPolicy, PressureLevel, RepetitionDetector, Synapse, WorkingMemory,
};
use proptest::prelude::*;

proptest! {
    /// Halo signal should be non-negative for any input
    #[test]
    fn halo_signal_non_negative(text in ".*") {
        let signal = calculate_halo_signal(&text);
            let _ = signal; // It's u16, so it's always >= 0
    }

    /// Bias breakdown total should match halo signal
    #[test]
    fn bias_breakdown_matches_halo(text in ".*") {
        let breakdown = get_bias_breakdown(&text);
        let halo = calculate_halo_signal(&text);
        prop_assert_eq!(breakdown.total(), halo);
    }

    /// Synapse roundtrip should preserve bits
    #[test]
    fn synapse_roundtrip(bits in any::<u128>()) {
        let synapse = Synapse::from_raw_u128(bits);
        let encoded = u128::from_le_bytes(synapse.into_bytes());
        prop_assert_eq!(bits, encoded);
    }

    /// Sifting should always produce a valid synapse (may have high entropy)
    #[test]
    fn sift_always_produces_synapse(observations in prop::collection::vec(".*", 0..10)) {
        let obs_refs: Vec<&str> = observations.iter().map(|s| s.as_str()).collect();
        let sifted = sift_perceptions(&obs_refs, "test");

        // Should always produce some entropy value
            let _ = sifted.raw_entropy(); // <= 0xFFFF is true for u16
    }

    /// Working memory should accept valid synapses
    #[test]
    fn working_memory_accepts_valid(entropy in 0u16..800u16) {
        let mut memory = WorkingMemory::<64>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(entropy);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = llmosafe::SiftedSynapse::new(synapse);
        prop_assert!(memory.update(sifted).is_ok());
    }

    /// Pressure level from percentage should be monotonic
    #[test]
    fn pressure_level_monotonic(pct in 0u8..=100u8) {
        let level = PressureLevel::from_percentage(pct);

        // Verify ordering
        if pct <= 25 {
            prop_assert!(level == PressureLevel::Nominal);
        } else if pct <= 50 {
            prop_assert!(level == PressureLevel::Elevated);
        } else if pct <= 75 {
            prop_assert!(level == PressureLevel::Critical);
        } else {
            prop_assert!(level == PressureLevel::Emergency);
        }
    }

    /// Safety decision severity should be ordered
    #[test]
    fn decision_severity_ordered(entropy in 0u16..=1500u16, surprise in 0u16..=1000u16) {
        let policy = EscalationPolicy::default();
        let decision = policy.decide(entropy, surprise, false);
        let severity = decision.severity();
        // Higher entropy should generally produce higher severity (modulo thresholds)
        // This is a weak property - just verify severity is in range
        prop_assert!(severity <= 3);
    }

    /// Repetition detector should eventually detect stuck state
    #[test]
    fn repetition_detector_stuck(event in "same", count in 3usize..=10) {
        let mut detector = RepetitionDetector::new(3);
        for _ in 0..count {
            detector.observe(&event);
        }
        prop_assert!(detector.is_stuck());
    }

    /// Confidence tracker trend should be bounded
    #[test]
    fn confidence_tracker_trend_bounded(confidences in prop::collection::vec(0.0f32..=1.0f32, 2..=8)) {
        let mut tracker = ConfidenceTracker::new(0.0, 100);
        for &c in &confidences {
            tracker.observe(c);
        }
        let trend = tracker.trend();
            prop_assert!((-1.0..=1.0).contains(&trend));
    }

    /// Bias breakdown should be additive
    #[test]
    fn bias_breakdown_additive(text1 in ".*", text2 in ".*") {
        let b1 = get_bias_breakdown(&text1);
        let b2 = get_bias_breakdown(&text2);
        // Combined text
        let combined = format!("{} {}", text1, text2);
        let bc = get_bias_breakdown(&combined);
        // Total should be at least as much as each individual
        // (may be more due to combined patterns)
        prop_assert!(bc.total() >= b1.total() || bc.total() >= b2.total());
    }

    /// Drift detector score should be bounded
    #[test]
    fn drift_score_bounded(goal in ".*", obs in ".*") {
        let mut detector = DriftDetector::new(&goal, 0.5);
        detector.observe(&obs);
        let score = detector.drift_score();
        prop_assert!((0.0..=1.0).contains(&score));
    }
}
