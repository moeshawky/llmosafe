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

#[cfg(feature = "testing")]
use llmosafe::{ReasoningLoop, SiftedProof};
use llmosafe::{Synapse, WorkingMemory};

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
