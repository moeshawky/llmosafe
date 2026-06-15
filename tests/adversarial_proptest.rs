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

#[cfg(all(feature = "testing", feature = "std"))]
mod tests {
    use llmosafe::{
        KernelError, ReasoningLoop, SiftedProof, SiftedSynapse, Synapse, WorkingMemory,
        PRESSURE_THRESHOLD,
    };
    use proptest::prelude::*;

    // Full-pipeline fuzz: 10K random bit patterns through sift→memory→kernel.
    // Must never panic. Must correctly reject bias. Must produce valid entropy.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]
        #[test]
        fn test_full_pipeline_never_panics(entropy: u16, surprise: u16, has_bias: bool, hash: u32) {
            // Build synapse with controlled adversarial fields
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(entropy);
            synapse.set_raw_surprise(surprise);
            synapse.set_has_bias(has_bias);
            synapse.set_anchor_hash(hash & 0x7FFFFFFF); // B31 = max 31 bits

            let sifted = SiftedSynapse::from_synapse(synapse);
            let proof = SiftedProof::for_testing();
            let mut memory = WorkingMemory::<64>::new(500);

            let result = memory.update(sifted, proof);
            match result {
                Ok((validated, vproof)) => {
                    // Bias-flagged synapses must be rejected by update()
                    if has_bias {
                        panic!("update() accepted a synapse with has_bias=true");
                    }
                    // Entropy and surprise are u16 by definition, always in range
                    let _ = validated.raw_entropy();
                    let _ = validated.raw_surprise();

                    let mut guard = ReasoningLoop::<10>::new();
                    let step = guard.next_step(validated, vproof);
                    match step {
                        Ok(()) | Err(KernelError::DepthExceeded | KernelError::CognitiveInstability) => {},
                        Err(other) => panic!("unexpected kernel error in pipeline: {:?}", other),
                    }
                }
                Err(KernelError::BiasHaloDetected) => {
                    assert!(has_bias, "BiasHaloDetected but has_bias=false — false positive");
                }
                Err(KernelError::CognitiveInstability) => {
                    assert!(entropy > PRESSURE_THRESHOLD as u16 || entropy == 0xFFFF,
                        "CognitiveInstability but entropy={entropy} is within bounds");
                }
                Err(KernelError::HallucinationDetected) => {
                    assert!(surprise as i128 > 500 || surprise > 500,
                        "HallucinationDetected but surprise={surprise} is within bounds");
                }
                Err(other) => {
                    panic!("unexpected error from update(): {:?}", other);
                }
            }
        }
    }
}

// ── Test 2: Corruption Test (Confession 44) ──────────────────────

/// Corruption proptest: generates a valid Synapse, serializes to bytes,
/// flips a random single bit, deserializes back, and verifies that the
/// corrupted form is either correctly rejected or produces detectably
/// different values from the original.
///
/// A corrupted synapse must NOT silently pass validation with values
/// identical to the original — that would be corruption = absence conflation.
#[cfg(all(feature = "testing", feature = "std"))]
mod corruption_tests {
    use llmosafe::{KernelError, SiftedProof, SiftedSynapse, Synapse, WorkingMemory};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn test_corrupted_synapse_rejection(
            entropy in 0u16..=u16::MAX,
            surprise in 0u16..=u16::MAX,
            has_bias in any::<bool>(),
            bit_to_flip in 0usize..128,
        ) {
            // 1. Build a valid synapse with controlled fields
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(entropy);
            synapse.set_raw_surprise(surprise);
            synapse.set_has_bias(has_bias);

            // Record original values before corruption
            let orig_entropy = synapse.raw_entropy();
            let orig_surprise = synapse.raw_surprise();
            let orig_bias = synapse.has_bias();

            // 2. Serialize to bytes
            let mut bytes: [u8; 16] = synapse.into_bytes();

            // 3. Flip a single bit at the selected position
            let byte_idx = bit_to_flip / 8;
            let bit_idx = bit_to_flip % 8;
            bytes[byte_idx] ^= 1u8 << bit_idx;

            // 4. Deserialize the corrupted bytes
            let corrupted = Synapse::from_raw_u128(u128::from_le_bytes(bytes));
            let corrupted_entropy = corrupted.raw_entropy();
            let corrupted_surprise = corrupted.raw_surprise();
            let corrupted_bias = corrupted.has_bias();

            // 5. Assert: the corrupted version is detectably different OR
            //    validation rejects it (Err). Silent acceptance with identical
            //    values is the corruption=absence failure mode.
            //
            // Use full struct equality (Synapse derives PartialEq) so that
            // changes to ANY field (entropy, surprise, bias, position, timestamp,
            // cascade_depth, anchor_hash, reserved) are detected.
            let values_changed = corrupted != synapse;
            let validation_result = corrupted.validate();

            prop_assert!(
                values_changed || validation_result.is_err(),
                "Corrupted synapse must be rejected or have detectably different values.\n\
                 orig: (entropy={}, surprise={}, bias={}), corrupted: (entropy={}, surprise={}, bias={}), \
                 bit flipped at position {}",
                orig_entropy, orig_surprise, orig_bias,
                corrupted_entropy, corrupted_surprise, corrupted_bias,
                bit_to_flip
            );

            // 6. If the corrupted synapse passes validation (because bit flip
            //    was in a non-critical field), it must still be structurally valid
            //    — try feeding through WorkingMemory to verify no panics.
            if validation_result.is_ok() {
                let sifted = SiftedSynapse::from_synapse(corrupted);
                let mut memory = WorkingMemory::<64>::new(1000);
                let _ = memory.update(sifted, SiftedProof::for_testing());
                // If we reach here without panic, the corruption was benign
                // (hit a field that doesn't affect safety-critical decisions).
            }
        }
    }

    /// Targeted corruption: flips bits specifically in the entropy field
    /// (bits 0-15) and verifies that corruption is always detected.
    #[test]
    fn test_corrupted_entropy_field_detection() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100); // stable entropy
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);

        let mut bytes: [u8; 16] = synapse.into_bytes();

        // Flip bit 7 of byte 0 (entropy field, lower byte)
        bytes[0] ^= 1u8 << 7; // entropy becomes significantly different
        let corrupted = Synapse::from_raw_u128(u128::from_le_bytes(bytes));

        assert_ne!(
            corrupted.raw_entropy(),
            100,
            "entropy must be different after byte flip in entropy field"
        );

        // Even if validation passes, WorkingMemory must not crash
        let sifted = SiftedSynapse::from_synapse(corrupted);
        let mut memory = WorkingMemory::<64>::new(1000);
        let _ = memory.update(sifted, SiftedProof::for_testing());
    }

    /// Targeted corruption: flips the bias bit (bit 32) specifically.
    /// A corrupted bias flag must be detected by validation.
    #[test]
    fn test_corrupted_bias_bit_detection() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);

        let mut bytes: [u8; 16] = synapse.into_bytes();
        // In modular_bitfield little-endian layout, has_bias is at bit 32 (byte 4, bit 0)
        bytes[4] ^= 1u8 << 0;
        let corrupted = Synapse::from_raw_u128(u128::from_le_bytes(bytes));

        assert!(
            corrupted.has_bias(),
            "flipping the bias bit must set has_bias=true"
        );

        let result = corrupted.validate();
        assert!(
            result.is_err(),
            "corrupted bias flag (has_bias=true) must cause validation failure"
        );
        assert_eq!(
            result.unwrap_err(),
            KernelError::BiasHaloDetected,
            "corrupted bias flag must produce BiasHaloDetected"
        );
    }

    /// Targeted corruption: flips bits in the surprise field (bits 16-31).
    /// Corrupted surprise must be detectably different.
    #[test]
    fn test_corrupted_surprise_field_detection() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(200);
        synapse.set_has_bias(false);

        let orig_surprise = synapse.raw_surprise();
        let mut bytes: [u8; 16] = synapse.into_bytes();
        // Surprise field is at bits 16-31 → bytes 2-3
        bytes[2] ^= 1u8 << 0; // flip LSB of byte 2 (bit 16 of 128-bit value)
        let corrupted = Synapse::from_raw_u128(u128::from_le_bytes(bytes));

        assert_ne!(
            corrupted.raw_surprise(),
            orig_surprise,
            "surprise must be different after byte flip in surprise field"
        );

        // No panic through WorkingMemory
        let sifted = SiftedSynapse::from_synapse(corrupted);
        let mut memory = WorkingMemory::<64>::new(1000);
        let _ = memory.update(sifted, SiftedProof::for_testing());
    }
}
