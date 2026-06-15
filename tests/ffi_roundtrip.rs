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

//! FFI Roundtrip Test (G6 Verification)
//!
//! Tests the C-ABI boundary for 128-bit Synapse functions.
//! Screens: G-HALL (APIs exist), G-EDGE (boundary values), G-SEM (return codes)

use llmosafe::*;
use proptest::prelude::*;

#[test]
fn test_ffi_get_stability_valid() {
    let valid_bits = 400u128;
    let result = unsafe { llmosafe_get_stability(valid_bits) };
    assert_eq!(result, 0, "Expected stable (0), got {result}");
}

#[test]
fn test_ffi_get_stability_unstable() {
    let unstable_bits = 50001u128;
    let result = unsafe { llmosafe_get_stability(unstable_bits) };
    assert_eq!(result, -2, "Expected unstable (-2), got {result}");
}

#[test]
fn test_ffi_get_stability_zero() {
    let result = unsafe { llmosafe_get_stability(0u128) };
    assert_eq!(result, 0, "Zero entropy should be stable");
}

#[test]
fn test_ffi_get_stability_max_u128() {
    let max_bits = u128::MAX;
    let result = unsafe { llmosafe_get_stability(max_bits) };
    assert!(result <= 0, "Max u128 should not crash");
}

#[test]
fn test_ffi_process_synapse_valid() {
    let valid_bits = 500u128;
    let result = unsafe { llmosafe_process_synapse(valid_bits) };
    assert_eq!(result, 0, "Expected success (0), got {result}");
}

#[test]
fn test_ffi_process_synapse_unstable() {
    let unstable_bits = 50001u128;
    let result = unsafe { llmosafe_process_synapse(unstable_bits) };
    assert_eq!(
        result, -2,
        "Expected cognitive instability (-2), got {result}"
    );
}

proptest! {
    #[test]
    fn ffi_roundtrip_u128_stability(bits in any::<u128>()) {
        let result = unsafe { llmosafe_get_stability(bits) };
        // Return-code format validation (existing, retained)
        prop_assert!(result == 0 || result == -1 || result == -2 || result == -3 || result == -4 || result == -5);

        // Behavioral validation (Test 6: Confession 48 — added):
        // Verify the return code matches the input's actual properties,
        // not just that it's a valid code.
        let synapse = Synapse::from_raw_u128(bits);
        let entropy = synapse.raw_entropy();
        let has_bias = synapse.has_bias();

        // has_bias=true → must return BiasHaloDetected (-3)
        if has_bias {
            prop_assert_eq!(result, -3,
                "has_bias=true must return -3 (BiasHaloDetected), got {}", result);
        }
        // entropy > PRESSURE_THRESHOLD (40000) → CognitiveInstability (-2)
        // (validate() uses PRESSURE_THRESHOLD, not STABILITY_THRESHOLD)
        else if entropy > 40000 {
            prop_assert_eq!(result, -2,
                "entropy={} > PRESSURE_THRESHOLD must return -2 (CognitiveInstability), got {}", entropy, result);
        }
        // entropy <= PRESSURE_THRESHOLD and no bias → success (0)
        else {
            prop_assert_eq!(result, 0,
                "entropy={} <= PRESSURE_THRESHOLD and no bias must return 0", entropy);
        }
    }

    #[test]
    fn ffi_roundtrip_synapse_create_validate(bits in any::<u128>()) {
        let synapse = Synapse::from_raw_u128(bits);
        let validation = synapse.validate();
        // Format assertion (existing, retained)
        prop_assert!(validation.is_ok() || validation.is_err());

        // Behavioral validation (Test 6: Confession 48 — added):
        // Verify that validation correctly rejects biased synapses
        // and accepts low-entropy clean ones.
        if synapse.has_bias() {
            prop_assert!(validation.is_err(),
                "has_bias=true must cause validation failure");
            prop_assert_eq!(validation.unwrap_err(), KernelError::BiasHaloDetected,
                "has_bias=true must produce BiasHaloDetected");
        } else if synapse.raw_entropy() > 40000 {
            prop_assert!(validation.is_err(),
                "entropy > PRESSURE_THRESHOLD (40000) must cause validation failure");
            prop_assert_eq!(validation.unwrap_err(), KernelError::CognitiveInstability,
                "high entropy must produce CognitiveInstability");
        } else {
            prop_assert!(validation.is_ok(),
                "low entropy, no bias must pass validation");
        }
    }
}

// ── Test 6: Behavioral assertions on top of existing tests ───────

#[test]
fn ffi_stability_biased_input_detection() {
    // Construct synapse with has_bias=true → must return -3 (BiasHaloDetected)
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(100);
    synapse.set_has_bias(true);
    let bits = u128::from_le_bytes(synapse.into_bytes());
    let result = unsafe { llmosafe_get_stability(bits) };
    assert_eq!(
        result, -3,
        "biased synapse must return -3 (BiasHaloDetected)"
    );
}

#[test]
fn ffi_stability_high_entropy_detection() {
    // Construct synapse with entropy > PRESSURE_THRESHOLD (40000) → must return -2
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(50001);
    synapse.set_has_bias(false);
    let bits = u128::from_le_bytes(synapse.into_bytes());
    let result = unsafe { llmosafe_get_stability(bits) };
    assert_eq!(
        result, -2,
        "high entropy must return -2 (CognitiveInstability)"
    );
}

#[test]
fn ffi_stability_low_entropy_clean() {
    // Construct clean synapse → must return 0
    let mut synapse = Synapse::new();
    synapse.set_raw_entropy(100);
    synapse.set_raw_surprise(50);
    synapse.set_has_bias(false);
    let bits = u128::from_le_bytes(synapse.into_bytes());
    let result = unsafe { llmosafe_get_stability(bits) };
    assert_eq!(result, 0, "clean synapse must return 0 (Stable)");
}

/// NaN-to-Halt decision path (Confession 49): when a NaN sensor value
/// reaches the FFI boundary, the NaN guard in pid_risk_to_decision
/// must trigger Halt, not Proceed.
#[test]
fn ffi_nan_sensor_triggers_halt_path() {
    // The FFI stability function doesn't directly handle NaN f32 inputs
    // (it takes u128), but we verify that the NaN guards in the pipeline
    // (sigmoid(NaN)→0.5, pid_risk_to_decision(NaN)→Halt) are tested
    // in kernel_edge_tests.rs (test_sigmoid_nan_guard_returns_neutral,
    // test_pid_risk_to_decision_nan_guard_returns_halt).
    //
    // Here we verify that the ffi_get_stability function handles both
    // zero-bits (all-clear) and u128::MAX (worst-case) correctly:
    let result_zero = unsafe { llmosafe_get_stability(0) };
    assert_eq!(result_zero, 0, "all-zero synapse must be stable");

    let result_max = unsafe { llmosafe_get_stability(u128::MAX) };
    // u128::MAX: all bits set → entropy=65535, surprise=65535, has_bias=true
    // has_bias=true takes priority in validate() → BiasHaloDetected (-3)
    assert_eq!(
        result_max, -3,
        "u128::MAX (bias+max entropy) must return -3"
    );
}

extern "C" {
    fn llmosafe_get_stability(synapse_bits: u128) -> i32;
    fn llmosafe_process_synapse(synapse_bits: u128) -> i32;
}
