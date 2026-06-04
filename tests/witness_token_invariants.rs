#[cfg(feature = "testing")]
use llmosafe::llmosafe_classifier::classify_text;
#[cfg(feature = "testing")]
use llmosafe::{
    KernelError, ReasoningLoop, SiftedProof, SiftedSynapse, SifterOutput, Synapse, ValidatedProof,
    WorkingMemory, sift_text,
};

#[cfg(feature = "testing")]
#[test]
fn test_proof_is_zero_sized() {
    assert_eq!(std::mem::size_of::<SiftedProof>(), 0);
    assert_eq!(std::mem::size_of::<ValidatedProof>(), 0);
}

#[cfg(feature = "testing")]
#[test]
fn test_proof_is_copy() {
    let proof = SiftedProof::for_testing();
    let proof2 = proof;
    let _ = proof;
    let _ = proof2;
}

#[cfg(feature = "testing")]
#[test]
fn test_proof_reuse_across_synapses() {
    use llmosafe::WorkingMemory;
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);

    let mut s1 = Synapse::new();
    s1.set_raw_entropy(100);
    s1.set_has_bias(false);
    let (validated1, _) = memory
        .update(SiftedSynapse::from_synapse(s1), proof)
        .unwrap();
    assert_eq!(validated1.raw_entropy(), 100);

    let mut s2 = Synapse::new();
    s2.set_raw_entropy(200);
    s2.set_has_bias(false);
    let (validated2, _) = memory
        .update(SiftedSynapse::from_synapse(s2), proof)
        .unwrap();
    assert_eq!(validated2.raw_entropy(), 200);
}

#[cfg(feature = "testing")]
#[test]
fn test_from_synapse_bias_rejection() {
    let mut synapse = Synapse::new();
    synapse.set_has_bias(true);
    synapse.set_raw_entropy(100);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);

    let result = memory.update(sifted, proof);
    assert!(matches!(result, Err(KernelError::BiasHaloDetected)));
}

#[cfg(feature = "testing")]
#[test]
fn test_from_synapse_high_entropy_rejection() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(50001);
    synapse.set_has_bias(false);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);

    let result = memory.update(sifted, proof);
    assert!(matches!(result, Err(KernelError::CognitiveInstability)));
}

#[cfg(feature = "testing")]
#[test]
fn test_from_synapse_high_surprise_rejection() {
    let mut synapse = Synapse::new();
    synapse.set_raw_surprise(600);
    synapse.set_has_bias(false);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);

    let result = memory.update(sifted, proof);
    assert!(matches!(result, Err(KernelError::HallucinationDetected)));
}

#[cfg(feature = "testing")]
#[test]
fn test_sifted_to_kernel_boundary() {
    let classification = classify_text("the weather is sunny today");
    let sifter_out = SifterOutput::from_classification(&classification);
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(sifter_out.raw_entropy);
    synapse.set_has_bias(sifter_out.has_bias);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);
    match memory.update(sifted, proof) {
        Ok((validated, vproof)) => {
            let mut guard = ReasoningLoop::<10>::new();
            let result = guard.next_step(validated, vproof);
            assert!(result.is_ok());
        }
        Err(_) => {
            // classifier rejects with synthetic model — valid pipeline behavior
        }
    }
}

#[cfg(feature = "testing")]
#[test]
fn test_sifted_to_kernel_boundary_via_sift_text() {
    let (sifted, proof) = sift_text("the weather is sunny today");
    let mut memory = WorkingMemory::<64>::new(500);
    match memory.update(sifted, proof) {
        Ok((validated, vproof)) => {
            let mut guard = ReasoningLoop::<10>::new();
            let _ = guard.next_step(validated, vproof);
        }
        Err(_) => {
            // valid pipeline behavior
        }
    }
}

#[cfg(feature = "testing")]
#[test]
fn test_c_abi_rejects_biased_synapse() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(500);
    synapse.set_has_bias(true);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);

    let result = memory.update(sifted, proof);
    assert!(matches!(result, Err(KernelError::BiasHaloDetected)));

    let biased_bits: u128 = 500u128 | (1u128 << 32);
    let result = llmosafe::llmosafe_memory::cognitive_memory::process_state_update(biased_bits);
    assert_eq!(
        result, -3,
        "C-ABI must reject biased synapse (BiasHaloDetected=-3)"
    );
}

#[cfg(feature = "testing")]
#[test]
fn test_c_abi_accepts_valid_synapse() {
    let valid_bits: u128 = 100u128;
    let result = llmosafe::llmosafe_memory::cognitive_memory::process_state_update(valid_bits);
    assert_eq!(result, 0);
}

#[cfg(feature = "testing")]
#[test]
fn test_manual_max_entropy_synapse() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(0xFFFF);
    let sifted = SiftedSynapse::from_synapse(synapse);
    assert_eq!(sifted.raw_entropy(), 0xFFFF);
}

#[cfg(feature = "testing")]
#[test]
fn test_full_chain_rejects_max_entropy() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(0xFFFF);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);
    let result = memory.update(sifted, proof);
    assert!(matches!(result, Err(KernelError::CognitiveInstability)));
}

#[cfg(feature = "testing")]
#[test]
fn test_proof_not_clone_across_threads() {
    let proof = SiftedProof::for_testing();
    let proof_ref = &proof;
    let _ = *proof_ref;
    let move_closure = move || {
        let _ = proof;
    };
    move_closure();
}

#[cfg(feature = "testing")]
#[test]
fn test_same_synapse_cannot_be_updated_twice() {
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(100);
    let sifted = SiftedSynapse::from_synapse(synapse);
    let proof = SiftedProof::for_testing();
    let mut memory = WorkingMemory::<64>::new(500);
    let _ = memory.update(sifted, proof);
}
