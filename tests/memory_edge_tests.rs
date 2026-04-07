//! G-EDGE tests for memory module - comprehensive boundary testing

#[cfg(test)]
mod tests {
    use llmosafe::{KernelError, SiftedSynapse, Synapse, WorkingMemory};

    #[test]
    fn test_memory_size_one() {
        let mut memory = WorkingMemory::<1>::new(1000);

        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        s1.set_raw_surprise(50);
        let sifted1 = SiftedSynapse::new(s1);
        memory.update(sifted1).unwrap();

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        s2.set_raw_surprise(50);
        let sifted2 = SiftedSynapse::new(s2);
        memory.update(sifted2).unwrap();

        assert_eq!(memory.mean_entropy(), 200.0);
    }

    #[test]
    fn test_memory_surprise_at_threshold() {
        let mut memory = WorkingMemory::<64>::new(500);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(500);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        assert!(
            memory.update(sifted).is_ok(),
            "Surprise == threshold should succeed"
        );
    }

    #[test]
    fn test_memory_surprise_above_threshold() {
        let mut memory = WorkingMemory::<64>::new(500);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(501);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        assert_eq!(
            memory.update(sifted),
            Err(KernelError::HallucinationDetected),
            "Surprise > threshold should fail"
        );
    }

    #[test]
    fn test_memory_entropy_at_stability_threshold() {
        let mut memory = WorkingMemory::<64>::new(1000);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1000);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        assert!(
            memory.update(sifted).is_ok(),
            "Entropy == STABILITY_THRESHOLD should succeed"
        );
    }

    #[test]
    fn test_memory_entropy_above_stability() {
        let mut memory = WorkingMemory::<64>::new(1000);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1001);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        assert_eq!(
            memory.update(sifted),
            Err(KernelError::CognitiveInstability),
            "Entropy > STABILITY_THRESHOLD should fail"
        );
    }

    #[test]
    fn test_memory_bias_detection() {
        let mut memory = WorkingMemory::<64>::new(1000);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);

        assert_eq!(
            memory.update(sifted),
            Err(KernelError::BiasHaloDetected),
            "has_bias=true should fail"
        );
    }

    #[test]
    fn test_memory_mean_empty() {
        let memory = WorkingMemory::<4>::new(1000);
        let mean = memory.mean_entropy();
        assert_eq!(mean, 0.0, "Empty memory should have mean 0");
    }

    #[test]
    fn test_memory_trend_flat() {
        let mut memory = WorkingMemory::<4>::new(1000);

        for _ in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(100);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }

        let trend = memory.trend();
        assert!(
            trend.abs() < 0.1,
            "Flat values should have ~0 trend, got {}",
            trend
        );
    }

    #[test]
    fn test_memory_drift_detection() {
        let mut memory = WorkingMemory::<4>::new(1000);

        for i in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(100 * (i + 1) as u16);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }

        assert!(
            memory.is_drifting(10.0),
            "Increasing values should be detected as drift"
        );
    }

    #[test]
    fn test_memory_variance_calculation() {
        let mut memory = WorkingMemory::<2>::new(1000);

        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        memory.update(SiftedSynapse::new(s1)).unwrap();

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        memory.update(SiftedSynapse::new(s2)).unwrap();

        let variance = memory.entropy_variance();
        assert!(
            (variance - 2500.0).abs() < 1.0,
            "Variance of [100,200] should be ~2500, got {}",
            variance
        );
    }

    #[test]
    fn test_memory_wraparound() {
        let mut memory = WorkingMemory::<2>::new(1000);

        for i in 1..=3 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(i * 100);
            memory.update(SiftedSynapse::new(synapse)).unwrap();
        }

        assert!(
            (memory.mean_entropy() - 250.0).abs() < 1.0,
            "After wraparound, mean should be (200+300)/2"
        );
    }
}
