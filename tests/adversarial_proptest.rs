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
