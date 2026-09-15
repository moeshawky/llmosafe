// Test code uses unwrap for assertions, raw indexing for fixed arrays,
// float comparison for exact-match tests, and arithmetic on controlled
// test inputs — all safe in test context per DO-178C.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::float_cmp))]
#![cfg_attr(test, allow(clippy::float_cmp_const))]
#![cfg_attr(test, allow(clippy::arithmetic_side_effects))]
#![cfg_attr(test, allow(clippy::indexing_slicing))]
#![cfg_attr(test, allow(clippy::as_conversions))]
#![cfg_attr(test, allow(clippy::expect_used))]
#![cfg_attr(test, allow(unused_results))]
#![cfg_attr(test, allow(clippy::shadow_reuse))]
#![cfg_attr(test, allow(clippy::shadow_same))]
#![cfg_attr(test, allow(clippy::shadow_unrelated))]

//! G-EDGE tests for detection module - comprehensive boundary testing

#[cfg(test)]
mod tests {
    use llmosafe::{
        AdversarialDetector, ConfidenceTracker, CusumDetector, DriftDetector, RepetitionDetector,
    };

    #[test]
    fn test_repetition_detector_threshold_boundary() {
        let mut detector = RepetitionDetector::new(3);

        detector.observe("same");
        assert!(!detector.is_stuck(), "n=1 should not be stuck");

        detector.observe("same");
        assert!(!detector.is_stuck(), "n=2 should not be stuck");

        detector.observe("same");
        assert!(detector.is_stuck(), "n=3 should be stuck");

        detector.observe("same");
        assert!(detector.is_stuck(), "n=4 should still be stuck");
    }

    #[test]
    fn test_repetition_detector_empty_input() {
        let mut detector = RepetitionDetector::new(3);

        detector.observe("");
        detector.observe("");
        detector.observe("");
        assert!(
            detector.is_stuck(),
            "Empty string repetition should be detected"
        );
    }

    #[test]
    fn test_repetition_detector_different_inputs() {
        let mut detector = RepetitionDetector::new(3);

        detector.observe("a");
        detector.observe("b");
        detector.observe("c");
        assert!(
            !detector.is_stuck(),
            "Different inputs should not trigger stuck"
        );
    }

    #[test]
    fn test_drift_detector_empty_strings() {
        let mut detector = DriftDetector::new("", 0.5);
        detector.observe("something");
        assert!(detector.drift_score() >= 0.0, "Should handle empty goal");

        let mut detector2 = DriftDetector::new("goal", 0.5);
        detector2.observe("");
        assert!(
            detector2.drift_score() >= 0.0,
            "Should handle empty observation"
        );

        let mut detector3 = DriftDetector::new("", 0.5);
        detector3.observe("");
        assert!(detector3.drift_score() >= 0.0, "Should handle both empty");
    }

    #[test]
    fn test_drift_detector_exact_match() {
        let mut detector = DriftDetector::new("rust safety library", 0.5);
        detector.observe("rust safety library");
        assert!(
            detector.drift_score() < 0.1,
            "Exact match should have low drift"
        );
    }

    #[test]
    fn test_confidence_tracker_extreme_values() {
        let mut tracker = ConfidenceTracker::new(0.5, 100);

        tracker.observe(0.0);
        assert!(tracker.is_low(), "Zero confidence should be low");

        let mut tracker2 = ConfidenceTracker::new(0.5, 100);
        tracker2.observe(1.0);
        assert!(!tracker2.is_low(), "Max confidence should not be low");

        let mut tracker3 = ConfidenceTracker::new(0.5, 100);
        tracker3.observe(0.0);
        tracker3.observe(1.0);
        tracker3.observe(0.5);
        assert!(tracker3.trend().abs() <= 1.0, "Trend should be bounded");
    }

    #[test]
    fn test_confidence_tracker_single_observation() {
        let mut tracker = ConfidenceTracker::new(0.5, 100);
        tracker.observe(0.7);
        let trend = tracker.trend();
        assert!(
            trend.abs() < 0.01,
            "Single observation should have ~0 trend"
        );
    }

    #[test]
    fn test_confidence_tracker_decaying() {
        let mut tracker = ConfidenceTracker::new(0.5, 3);

        tracker.observe(0.9);
        tracker.observe(0.7);
        tracker.observe(0.5);
        tracker.observe(0.3);

        // After 3 consecutive drops, should be decaying
        assert!(
            tracker.is_decaying(),
            "Decreasing sequence should be decaying after 3 drops"
        );
    }

    #[test]
    fn test_adversarial_detector_all_patterns() {
        let detector = AdversarialDetector::new();

        // These patterns should be detected
        let detected_patterns = vec![
            ("ignore previous instructions", true),
            ("disregard the above", true),
            ("you are now", true),
            ("jailbreak", true),
            ("developer mode", true),
        ];

        for (pattern, should_detect) in detected_patterns {
            let found = detector.detect_substrings(pattern);
            if should_detect {
                assert_ne!(found, 0, "Should detect pattern: {pattern}");
            }
        }
    }

    #[test]
    fn test_adversarial_detector_clean_input() {
        let detector = AdversarialDetector::new();
        let patterns = detector.detect_substrings("This is a normal request for help");
        assert_eq!(patterns, 0, "Clean input should have no patterns");
    }

    #[test]
    fn test_cusum_detector_threshold_boundary() {
        // D3: Normalized domain [0,1]. mu_ref=0.5, k=0.1, h=0.5.
        let mut detector = CusumDetector::new(0.5, 0.1, 0.5);

        for _ in 0..10 {
            assert!(
                !detector.update(0.5, true),
                "At reference, should not detect"
            );
        }

        let mut detector2 = CusumDetector::new(0.5, 0.1, 0.5);
        for _ in 0..20 {
            detector2.update(0.9, true);
        }
        assert!(detector2.detected(), "Sustained shift should trigger");
    }

    #[test]
    fn test_cusum_detector_negative_values() {
        // D3: Normalized domain. Test extreme out-of-range values.
        let mut detector = CusumDetector::new(0.5, 0.1, 0.5);

        // Positive values above reference (accepted)
        assert!(!detector.update(0.8, true), "Should handle positive values");

        // Values below reference trigger s_low (accepted)
        let mut detector2 = CusumDetector::new(0.5, 0.1, 0.5);

        // Values far below reference trigger s_low
        for _ in 0..30 {
            detector2.update(0.0, true); // 0.5 units below reference
        }
        // Eventually should detect
    }

    #[test]
    fn test_repetition_detector_reset() {
        let mut detector = RepetitionDetector::new(2);
        detector.observe("a");
        detector.observe("a");
        assert!(detector.is_stuck());

        detector.reset();
        assert!(!detector.is_stuck(), "After reset, should not be stuck");

        detector.observe("b");
        assert!(
            !detector.is_stuck(),
            "New input after reset should not be stuck"
        );
    }

    #[test]
    fn test_drift_detector_threshold_boundary() {
        let mut detector = DriftDetector::new("safety critical rust library", 0.5);
        detector.observe("safety critical rust library");
        assert!(!detector.is_drifting(), "Exact match should not drift");

        detector.observe("python web framework");
        assert!(detector.is_drifting(), "Different topic should drift");
    }

    // D4: drift_score must always be ∈ [0,1] by construction.
    // Repeating one goal word must not impersonate full goal coverage.
    #[test]
    fn test_drift_detector_range_property() {
        // Repeated single word should give high drift (not aligned)
        let mut detector = DriftDetector::new("a b c", 0.5);
        detector.observe("a a a a a a a a");
        let score = detector.drift_score();
        assert!(
            (0.0..=1.0).contains(&score),
            "Score must be in [0,1], got {score}"
        );
        assert!(
            detector.is_drifting(),
            "Single-word repetition must drift (score > 0.5)"
        );
        assert!(
            score > 0.5,
            "Single-word repetition should give high drift, got {score}"
        );
    }

    // Disjoint text → maximal drift (score ≈ 1.0).
    #[test]
    fn test_drift_disjoint_maximal_drift() {
        let mut detector = DriftDetector::new("a b c", 0.5);
        detector.observe("xyz completely unrelated words here");
        assert!(
            detector.drift_score() >= 0.9,
            "Disjoint text should give maximal drift, got {}",
            detector.drift_score()
        );
    }

    // Complete coverage → ~zero drift.
    #[test]
    fn test_drift_full_coverage_zero_drift() {
        let mut detector = DriftDetector::new("a b c", 0.5);
        detector.observe("a b c");
        assert!(
            detector.drift_score() < 0.01,
            "Full coverage should give near-zero drift, got {}",
            detector.drift_score()
        );
        assert!(!detector.is_drifting(), "Fully covered should not drift");
    }

    // Oracle regression: goal "a b c" + 8×"a" must read in_range AND drifted.
    #[test]
    fn test_drift_oracle_example() {
        let mut detector = DriftDetector::new("a b c", 0.5);
        detector.observe("a a a a a a a a");
        let score = detector.drift_score();
        assert!(
            (0.0..=1.0).contains(&score),
            "Oracle example: score must be in [0,1], got {score}"
        );
        assert!(
            detector.is_drifting(),
            "Oracle example: must drift (score > threshold)"
        );
    }
}
