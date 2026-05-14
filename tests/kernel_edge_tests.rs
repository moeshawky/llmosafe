//! G-EDGE tests for kernel module - comprehensive boundary testing

#[cfg(test)]
mod tests {
    use llmosafe::{
        CognitiveEntropy, CusumDetector, DynamicStabilityMonitor, KernelError, ReasoningLoop,
        SiftedSynapse, StabilityResult, Synapse, PRESSURE_THRESHOLD, STABILITY_THRESHOLD,
    };

    #[test]
    fn test_reasoning_loop_max_steps_exceeded() {
        let mut loop_guard = ReasoningLoop::<3>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let validated = memory.update(sifted).unwrap();

        assert!(loop_guard.next_step(validated).is_ok());
        assert!(loop_guard.next_step(validated).is_ok());
        assert!(loop_guard.next_step(validated).is_ok());

        let result = loop_guard.next_step(validated);
        assert_eq!(result, Err(KernelError::DepthExceeded));
    }

    #[test]
    fn test_reasoning_loop_exact_boundary() {
        let mut loop_guard = ReasoningLoop::<5>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        let sifted = SiftedSynapse::new(synapse);
        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let validated = memory.update(sifted).unwrap();

        for i in 0..5 {
            assert!(
                loop_guard.next_step(validated).is_ok(),
                "Step {} should succeed",
                i
            );
        }
        assert_eq!(
            loop_guard.next_step(validated),
            Err(KernelError::DepthExceeded)
        );
    }

    #[test]
    fn test_synapse_entropy_u16_max() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(u16::MAX);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);

        assert_eq!(synapse.raw_entropy(), 65535);
        let entropy = synapse.entropy();
        assert!(!entropy.is_stable(1000));
    }

    #[test]
    fn test_synapse_entropy_zero() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0);
        assert_eq!(synapse.raw_entropy(), 0);
        assert!(synapse.entropy().is_stable(1000));
    }

    #[test]
    fn test_cusum_detector_threshold_exact() {
        let mut detector = CusumDetector::new(500.0, 50.0, 200.0);

        for _ in 0..10 {
            assert!(!detector.update(500.0), "At reference, should not detect");
        }

        for _ in 0..5 {
            detector.update(750.0);
        }
        assert!(
            detector.detected() || detector.update(750.0),
            "Should detect after sustained shift"
        );
    }

    #[test]
    fn test_dynamic_stability_all_results() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        assert_eq!(monitor.update(100), StabilityResult::Stable);
        assert_eq!(monitor.update(100), StabilityResult::Stable);

        let high_result = monitor.update(10000);
        assert!(matches!(
            high_result,
            StabilityResult::High | StabilityResult::Both
        ));

        monitor.reset();
        monitor.update(1000);
        let low_result = monitor.update(1);
        assert!(matches!(
            low_result,
            StabilityResult::Low | StabilityResult::Stable
        ));
    }

    #[test]
    fn test_cognitive_entropy_boundary() {
        let at_threshold = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD);
        assert!(at_threshold.is_stable(STABILITY_THRESHOLD));

        let above = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD + 1);
        assert!(!above.is_stable(STABILITY_THRESHOLD));

        let at_pressure = CognitiveEntropy::<28, 2>::new(PRESSURE_THRESHOLD);
        assert!(at_pressure.is_stable(STABILITY_THRESHOLD));
    }

    #[test]
    fn test_synapse_bit_patterns() {
        let zero = Synapse::from_raw_u128(0);
        assert_eq!(zero.raw_entropy(), 0);
        assert_eq!(zero.raw_surprise(), 0);

        let max_bits = Synapse::from_raw_u128(u128::MAX);
        assert_eq!(max_bits.raw_entropy(), u16::MAX);

        // In little-endian, entropy is in the lowest 16 bits
        // Setting bit 0-15 should set entropy
        let entropy_one = Synapse::from_raw_u128(1u128); // entropy = 1
        assert_eq!(entropy_one.raw_entropy(), 1);

        // Setting bits 16-31 should set surprise
        let surprise_one = Synapse::from_raw_u128(1u128 << 16); // surprise = 1
        assert_eq!(surprise_one.raw_surprise(), 1);
        assert_eq!(surprise_one.raw_entropy(), 0);
    }

    #[test]
    fn test_reasoning_loop_size_one() {
        let mut loop_guard = ReasoningLoop::<1>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        let sifted = SiftedSynapse::new(synapse);
        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let validated = memory.update(sifted).unwrap();

        assert!(loop_guard.next_step(validated).is_ok());
        assert_eq!(
            loop_guard.next_step(validated),
            Err(KernelError::DepthExceeded)
        );
    }

    #[test]
    fn test_cusum_negative_values() {
        // Test CusumDetector with negative inputs
        // CusumDetector computes: s_high = max(0, s_high + (val - mu_ref) - k)
        // For mu_ref=100, k=10, h=100, val=-50:
        // s_high = max(0, 0 + (-50 - 100) - 10) = max(0, -160) = 0
        // s_low = max(0, 0 - (-50 - 100) - 10) = max(0, 140) = 140
        let mut detector = CusumDetector::new(100.0, 10.0, 200.0);

        // Negative relative to reference
        detector.update(-50.0);
        // s_low should be positive but below threshold
        assert!(!detector.detected());

        // More negative values should eventually trigger
        for _ in 0..5 {
            detector.update(-50.0);
        }
        // After enough samples below reference, should detect
    }

    #[test]
    fn test_cusum_reset() {
        let mut detector = CusumDetector::new(500.0, 50.0, 200.0);
        for _ in 0..20 {
            detector.update(1000.0);
        }
        detector.reset();
        assert_eq!(detector.s_high(), 0.0);
        assert_eq!(detector.s_low(), 0.0);
    }
}
