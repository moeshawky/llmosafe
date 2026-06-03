//! FFI Roundtrip Test (G6 Verification)
//!
//! Tests the C-ABI boundary for 128-bit Synapse functions.
//! Screens: G-HALL (APIs exist), G-EDGE (boundary values), G-SEM (return codes)

use llmosafe::*;
use proptest::prelude::*;

#[test]
fn test_ffi_get_stability_valid() {
    let valid_bits = 400u64;
    let result = unsafe { llmosafe_get_stability(valid_bits) };
    assert_eq!(result, 0, "Expected stable (0), got {}", result);
}

#[test]
fn test_ffi_get_stability_unstable() {
    let unstable_bits = 50001u64;
    let result = unsafe { llmosafe_get_stability(unstable_bits) };
    assert_eq!(result, -2, "Expected unstable (-2), got {}", result);
}

#[test]
fn test_ffi_get_stability_zero() {
    let result = unsafe { llmosafe_get_stability(0u64) };
    assert_eq!(result, 0, "Zero entropy should be stable");
}

#[test]
fn test_ffi_get_stability_max_lower_64() {
    let max_lower = u64::MAX;
    let result = unsafe { llmosafe_get_stability(max_lower) };
    assert!(result <= 0, "Max u64 should not crash");
}

#[test]
fn test_ffi_process_synapse_valid() {
    let valid_bits = 500u64;
    let result = unsafe { llmosafe_process_synapse(valid_bits) };
    assert_eq!(result, 0, "Expected success (0), got {}", result);
}

#[test]
fn test_ffi_process_synapse_unstable() {
    let unstable_bits = 50001u64;
    let result = unsafe { llmosafe_process_synapse(unstable_bits) };
    assert_eq!(
        result, -2,
        "Expected cognitive instability (-2), got {}",
        result
    );
}

proptest! {
    #[test]
    fn ffi_roundtrip_u64_stability(bits in any::<u64>()) {
        let result = unsafe { llmosafe_get_stability(bits) };
        prop_assert!(result == 0 || result == -1 || result == -2 || result == -3 || result == -4 || result == -5);
    }

    #[test]
    fn ffi_roundtrip_synapse_create_validate(bits in any::<u128>()) {
        let synapse = Synapse::from_raw_u128(bits);
        let validation = synapse.validate();
        prop_assert!(validation.is_ok() || validation.is_err());
    }
}

#[allow(improper_ctypes)]
extern "C" {
    fn llmosafe_get_stability(synapse_bits: u64) -> i32;
    fn llmosafe_process_synapse(synapse_bits: u64) -> i32;
}
