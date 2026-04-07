use llmosafe::{sift_perceptions, ReasoningLoop, Synapse, WorkingMemory};

#[test]
fn test_wrong_tier_sifted_to_kernel() {
    let _sifted = sift_perceptions(&["test"], "test");
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
