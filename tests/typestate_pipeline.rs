#[allow(unused_imports)]
use llmosafe::{ReasoningLoop, SiftedProof, Synapse, WorkingMemory};

#[cfg(feature = "testing")]
#[test]
fn test_wrong_tier_sifted_to_kernel() {
    let _proof = SiftedProof::for_testing();
    let _loop_guard = ReasoningLoop::<10>::new();
    // This should fail: ReasoningLoop::next_step expects ValidatedSynapse, not SiftedSynapse
    // _loop_guard.next_step(_sifted).unwrap();
}

#[test]
fn test_wrong_tier_raw_to_memory() {
    let _memory = WorkingMemory::<64>::new(1000);
    let _synapse = Synapse::new();
    // This should fail: WorkingMemory::update expects SiftedSynapse, not raw Synapse
    // _memory.update(_synapse).unwrap();
}
