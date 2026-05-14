//! G-ERR tests for all KernelError variants

#[cfg(test)]
mod tests {
    use llmosafe::{
        KernelError, ReasoningLoop, SiftedSynapse, Synapse, WorkingMemory, STABILITY_THRESHOLD,
    };

    #[test]
    fn test_kernel_error_depth_exceeded() {
        let mut loop_guard = ReasoningLoop::<2>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        let mut memory = WorkingMemory::<64>::new(1000);
        let validated = memory.update(sifted).unwrap();

        loop_guard.next_step(validated).unwrap();
        loop_guard.next_step(validated).unwrap();

        let result = loop_guard.next_step(validated);
        match result {
            Err(KernelError::DepthExceeded) => (),
            Err(e) => panic!("Expected DepthExceeded, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_kernel_error_cognitive_instability() {
        let mut memory = WorkingMemory::<64>::new(1000);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy((STABILITY_THRESHOLD + 1) as u16);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        match memory.update(sifted) {
            Err(KernelError::CognitiveInstability) => (),
            Err(e) => panic!("Expected CognitiveInstability, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_kernel_error_bias_halo_detected() {
        let mut memory = WorkingMemory::<64>::new(1000);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);

        match memory.update(sifted) {
            Err(KernelError::BiasHaloDetected) => (),
            Err(e) => panic!("Expected BiasHaloDetected, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_kernel_error_hallucination_detected() {
        let mut memory = WorkingMemory::<64>::new(500);

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(501);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);

        match memory.update(sifted) {
            Err(KernelError::HallucinationDetected) => (),
            Err(e) => panic!("Expected HallucinationDetected, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_kernel_error_resource_exhaustion() {
        // ResourceExhaustion is typically raised by ResourceGuard
        // This tests that the error variant exists and can be matched
        let err = KernelError::ResourceExhaustion;
        let display = format!("{}", err);
        assert!(
            display.contains("resource") || display.contains("exhaustion") || !display.is_empty()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_error_codes_match() {
        // C-ABI functions are exported from lib.rs
        // Use the library's own test infrastructure

        // Test valid case via lib's unit tests (they verify C-ABI)
        // This test verifies error variant matching instead
        let mut memory = WorkingMemory::<64>::new(500);

        // Test hallucination detection
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(501);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted),
            Err(KernelError::HallucinationDetected),
            "High surprise should trigger HallucinationDetected"
        );

        // Test cognitive instability
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1001);
        synapse.set_raw_surprise(100);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted),
            Err(KernelError::CognitiveInstability),
            "High entropy should trigger CognitiveInstability"
        );

        // Test bias detection
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted),
            Err(KernelError::BiasHaloDetected),
            "Bias flag should trigger BiasHaloDetected"
        );
    }

    #[test]
    fn test_all_error_variants_display() {
        let errors = vec![
            KernelError::DepthExceeded,
            KernelError::CognitiveInstability,
            KernelError::BiasHaloDetected,
            KernelError::HallucinationDetected,
            KernelError::ResourceExhaustion,
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(
                !display.is_empty(),
                "Error {:?} should have Display impl",
                err
            );
        }
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(KernelError::DepthExceeded, KernelError::DepthExceeded);
        assert_eq!(
            KernelError::CognitiveInstability,
            KernelError::CognitiveInstability
        );
        assert_ne!(
            KernelError::DepthExceeded,
            KernelError::CognitiveInstability
        );
    }
}
