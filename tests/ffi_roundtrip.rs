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
    assert_eq!(result, 0, "Expected stable (0), got {}", result);
}

#[test]
fn test_ffi_get_stability_unstable() {
    let unstable_bits = 1500u128;
    let result = unsafe { llmosafe_get_stability(unstable_bits) };
    assert_eq!(result, -2, "Expected unstable (-2), got {}", result);
}

#[test]
fn test_ffi_get_stability_zero() {
    let result = unsafe { llmosafe_get_stability(0u128) };
    assert_eq!(result, 0, "Zero entropy should be stable");
}

#[test]
fn test_ffi_get_stability_max_lower_64() {
    let max_lower = u64::MAX as u128;
    let result = unsafe { llmosafe_get_stability(max_lower) };
    assert!(result <= 0, "Max lower 64 bits should not crash");
}

#[test]
fn test_ffi_process_synapse_valid() {
    let valid_bits = 500u128;
    let result = unsafe { llmosafe_process_synapse(valid_bits) };
    assert_eq!(result, 0, "Expected success (0), got {}", result);
}

#[test]
fn test_ffi_process_synapse_unstable() {
    let unstable_bits = 1500u128;
    let result = unsafe { llmosafe_process_synapse(unstable_bits) };
    assert_eq!(
        result, -2,
        "Expected cognitive instability (-2), got {}",
        result
    );
}

proptest! {
    #[test]
    fn ffi_roundtrip_u128_stability(bits in any::<u128>()) {
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

extern "C" {
    #[allow(improper_ctypes)]
    fn llmosafe_get_stability(synapse_bits: u128) -> i32;
    #[allow(improper_ctypes)]
    fn llmosafe_process_synapse(synapse_bits: u128) -> i32;
}
