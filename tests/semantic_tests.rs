//! G-SEM semantic correctness tests - math verification

#[cfg(test)]
mod tests {
    use llmosafe::{
        calculate_halo_signal, get_bias_breakdown, CusumDetector, DynamicStabilityMonitor,
        SiftedSynapse, StabilityResult, Synapse, WorkingMemory,
    };

    #[test]
    fn test_cusum_math_correctness() {
        // Known values test: mu_ref=100, k=10, h=50
        // After 5 values of 150 (shift of +50 from reference):
        // s_high = max(0, 0 + 50 - 10) = 40
        // s_high = max(0, 40 + 50 - 10) = 80 -> exceeds h=50

        let mut detector = CusumDetector::new(100.0, 10.0, 50.0);

        assert!(!detector.update(100.0), "At reference, no detection");
        assert!(!detector.update(150.0), "First deviation, s_high=40");

        let detected_after_second = detector.update(150.0);
        assert!(
            detected_after_second,
            "Second deviation should trigger (s_high=80 > 50)"
        );
    }

    #[test]
    fn test_cusum_symmetric_detection() {
        // Test both directions
        let mut detector_high = CusumDetector::new(100.0, 10.0, 100.0);
        let mut detector_low = CusumDetector::new(100.0, 10.0, 100.0);

        // High shift
        for _ in 0..10 {
            detector_high.update(200.0);
        }
        assert!(detector_high.detected(), "High shift should be detected");

        // Low shift
        for _ in 0..10 {
            detector_low.update(0.0);
        }
        assert!(detector_low.detected(), "Low shift should be detected");
    }

    #[test]
    fn test_dynamic_stability_math_correctness() {
        // MSB index calculation: floor(log2(x))
        // log2(100) ≈ 6.64, so MSB = 6
        // log2(1000) ≈ 9.97, so MSB = 9
        // log2(10000) ≈ 13.29, so MSB = 13

        let mut monitor = DynamicStabilityMonitor::new(2);

        // First value sets baseline
        let result1 = monitor.update(100);
        assert_eq!(
            result1,
            StabilityResult::Stable,
            "First value initializes baseline"
        );

        // Similar value should be stable
        let result2 = monitor.update(100);
        assert_eq!(result2, StabilityResult::Stable, "Similar value is stable");

        // Much higher value (100x) should be unstable (MSB diff > k)
        let result3 = monitor.update(10000);
        assert!(
            matches!(result3, StabilityResult::High | StabilityResult::Both),
            "Large increase should be detected"
        );
    }

    #[test]
    fn test_surprise_gating_formula() {
        // Surprise threshold gates acceptance
        // If surprise > threshold, reject with HallucinationDetected

        let mut memory = WorkingMemory::<64>::new(500);

        // Exactly at threshold
        let mut synapse1 = Synapse::new();
        synapse1.set_raw_entropy(100);
        synapse1.set_raw_surprise(500);
        let sifted1 = SiftedSynapse::new(synapse1);
        assert!(
            memory.update(sifted1).is_ok(),
            "surprise == threshold should pass"
        );

        // Just above threshold
        let mut synapse2 = Synapse::new();
        synapse2.set_raw_entropy(100);
        synapse2.set_raw_surprise(501);
        let sifted2 = SiftedSynapse::new(synapse2);
        assert!(
            memory.update(sifted2).is_err(),
            "surprise > threshold should fail"
        );
    }

    #[test]
    fn test_entropy_accumulation_correctness() {
        // Mean entropy should be arithmetic mean
        let mut memory = WorkingMemory::<4>::new(1000);

        let values = [100u16, 200, 300, 400];
        for &v in &values {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(v);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }

        let mean = memory.mean_entropy();
        let expected_mean = (100.0 + 200.0 + 300.0 + 400.0) / 4.0;
        assert!(
            (mean - expected_mean).abs() < 1.0,
            "Mean should be {}, got {}",
            expected_mean,
            mean
        );
    }

    #[test]
    fn test_variance_calculation_correctness() {
        // Variance = E[X^2] - E[X]^2
        let mut memory = WorkingMemory::<2>::new(1000);

        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        memory.update(SiftedSynapse::new(s1)).unwrap();

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        memory.update(SiftedSynapse::new(s2)).unwrap();

        // Variance of [100, 200] = ((100-150)^2 + (200-150)^2) / 2 = 2500
        let variance = memory.entropy_variance();
        assert!(
            (variance - 2500.0).abs() < 1.0,
            "Variance should be 2500, got {}",
            variance
        );
    }

    #[test]
    fn test_trend_calculation_correctness() {
        // Linear regression slope
        // For [1, 2, 3, 4] mapped to [100, 200, 300, 400]
        // Slope should be exactly 100 (perfect linear)

        let mut memory = WorkingMemory::<4>::new(1000);

        for i in 1..=4u16 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(i * 100);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }

        let trend = memory.trend();
        assert!(
            (trend - 100.0).abs() < 1.0,
            "Trend should be 100, got {}",
            trend
        );
    }

    #[test]
    fn test_bias_breakdown_sum_correctness() {
        // BiasBreakdown.total() should equal calculate_halo_signal()
        let test_texts = vec![
            "normal text",
            "expert recommendation",
            "popular limited offer",
            "expert says this is popular",
        ];

        for text in test_texts {
            let breakdown = get_bias_breakdown(text);
            let halo = calculate_halo_signal(text);
            assert_eq!(
                breakdown.total(),
                halo,
                "Breakdown total should equal halo signal for: {}",
                text
            );
        }
    }

    #[test]
    fn test_msb_index_computation() {
        // Test MSB (most significant bit) calculation
        // 1 -> MSB 0 (2^0)
        // 2 -> MSB 1 (2^1)
        // 4 -> MSB 2 (2^2)
        // 255 -> MSB 7 (2^7 < 255 < 2^8)
        // 256 -> MSB 8 (2^8)

        let test_cases: Vec<(u32, u8)> = vec![
            (1, 0),
            (2, 1),
            (4, 2),
            (8, 3),
            (255, 7),
            (256, 8),
            (1024, 10),
            (65535, 15),
        ];

        for (value, expected_msb) in test_cases {
            let msb = 31u8.wrapping_sub(value.leading_zeros() as u8);
            assert_eq!(
                msb, expected_msb,
                "MSB of {} should be {}, got {}",
                value, expected_msb, msb
            );
        }
    }
}
