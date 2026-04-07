#![cfg_attr(not(feature = "std"), no_std)]

//! LLMOSAFE: A Safety-Critical AI Agent Library
//!
//! This library provides the formal primitives for building safety-critical AI agents.
//! It implements a 4-tier safety architecture:
//! - Tier 0: Resource Body (Physical safety - requires `std`)
//! - Tier 1: Deterministic Kernel (Formal Law)
//! - Tier 2: Cognitive Working Memory (Stateful Safety)
//! - Tier 3: Perceptual Sifter (Boundary Safety)

pub mod llmosafe_kernel;
pub mod llmosafe_memory;
pub mod llmosafe_sifter;
pub mod llmosafe_integration;
pub mod llmosafe_detection;

#[cfg(feature = "std")]
pub mod llmosafe_body;

#[cfg(feature = "std")]
pub use llmosafe_body::ResourceGuard;
pub use llmosafe_kernel::{
    CognitiveEntropy, DynamicStabilityMonitor, KernelError, ReasoningLoop, SiftedSynapse,
    StabilityResult, Synapse, ValidatedSynapse, PRESSURE_THRESHOLD, STABILITY_THRESHOLD,
};
pub use llmosafe_memory::WorkingMemory;
pub use llmosafe_sifter::{calculate_halo_signal, calculate_utility, sift_perceptions, get_bias_breakdown};
pub use llmosafe_integration::{SafetyDecision, EscalationPolicy, PressureLevel, EscalationReason, SafetyContext};
pub use llmosafe_detection::{RepetitionDetector, DriftDetector, ConfidenceTracker, AdversarialDetector, CusumDetector, DetectionResult};

#[cfg(feature = "std")]
mod c_abi {
    use crate::llmosafe_body::ResourceGuard;
    use crate::llmosafe_kernel::KernelError;
    use crate::llmosafe_kernel::Synapse;
    use crate::llmosafe_memory;

    #[no_mangle]
    pub extern "C" fn llmosafe_process_synapse(synapse_bits: u64) -> i32 {
        llmosafe_memory::cognitive_memory::process_state_update(synapse_bits as u128)
    }

    #[no_mangle]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub extern "C" fn llmosafe_calculate_halo(text_ptr: *const core::ffi::c_char) -> u16 {
        if text_ptr.is_null() {
            return 0;
        }
        let c_str = unsafe { core::ffi::CStr::from_ptr(text_ptr) };
        let text = c_str.to_string_lossy();
        crate::llmosafe_sifter::calculate_halo_signal(&text)
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_check_resources(ceiling_mb: u32) -> i32 {
        let ceiling_bytes = (ceiling_mb as usize) * 1024 * 1024;
        let guard = ResourceGuard::new(ceiling_bytes);

        match guard.check() {
            Ok(_) => 0,
            Err(KernelError::ResourceExhaustion) => -5,
            Err(KernelError::DepthExceeded) => -1,
            Err(KernelError::CognitiveInstability) => -2,
            Err(KernelError::BiasHaloDetected) => -3,
            Err(KernelError::HallucinationDetected) => -4,
        }
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_get_resource_pressure(ceiling_mb: u32) -> u8 {
        let ceiling_bytes = (ceiling_mb as usize) * 1024 * 1024;
        if ceiling_bytes == 0 {
            return 100;
        }
        let guard = ResourceGuard::new(ceiling_bytes);
        guard.pressure()
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_get_stability(synapse_bits: u64) -> i32 {
        let synapse = Synapse::from_raw_u64(synapse_bits);
        match synapse.validate() {
            Ok(()) => 0,
            Err(KernelError::CognitiveInstability) => -2,
            Err(KernelError::BiasHaloDetected) => -3,
            Err(KernelError::DepthExceeded) => -1,
            Err(KernelError::HallucinationDetected) => -4,
            Err(KernelError::ResourceExhaustion) => -5,
        }
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_get_system_cpu_load() -> u8 {
        ResourceGuard::system_cpu_load()
    }

    // We don't redefine it here because it's already defined with #[no_mangle] in llmosafe_body.rs
    // But we need it to be visible to cbindgen in this crate.
    // Re-exporting it without #[no_mangle] here won't work for C-ABI if it's already there.
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_process_synapse_valid_bits() {
        let bits = 400u64;
        let result = crate::c_abi::llmosafe_process_synapse(bits);
        assert_eq!(result, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_calculate_halo_null_pointer() {
        let result = crate::c_abi::llmosafe_calculate_halo(std::ptr::null());
        assert_eq!(result, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_check_resources_ceiling_zero() {
        let result = crate::c_abi::llmosafe_check_resources(0);
        assert_eq!(result, -5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_invalid_utf8() {
        let invalid_data = b"Hello\xFFWorld\0";
        let result = crate::c_abi::llmosafe_calculate_halo(
            invalid_data.as_ptr() as *const core::ffi::c_char
        );
        let _ = result;
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_very_long_string() {
        let mut long_string = std::vec![b'a'; 1024 * 1024];
        long_string.push(0);
        let result =
            crate::c_abi::llmosafe_calculate_halo(long_string.as_ptr() as *const core::ffi::c_char);
        let _ = result;
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_get_resource_pressure() {
        let pressure = crate::c_abi::llmosafe_get_resource_pressure(1024);
        assert!(pressure <= 100);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_get_stability_valid() {
        let valid_bits = 400u64;
        let result = crate::c_abi::llmosafe_get_stability(valid_bits);
        assert_eq!(result, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_get_stability_unstable() {
        let unstable_bits = 1100u64;
        let result = crate::c_abi::llmosafe_get_stability(unstable_bits);
        assert_eq!(result, -2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_get_system_cpu_load() {
        let load = crate::c_abi::llmosafe_get_system_cpu_load();
        assert!(load <= 100);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_get_environmental_entropy() {
        let entropy = crate::llmosafe_body::llmosafe_get_environmental_entropy();
        let _ = entropy;
    }
}
