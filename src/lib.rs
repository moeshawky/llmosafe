#![cfg_attr(not(feature = "std"), no_std)]
// modular-bitfield's `#[bitfield]` proc macro generates field definitions
// with parenthesized types (e.g., `pub raw_entropy: B16`), which triggers
// the `unused_parens` lint on all 8 fields. Struct-level #[allow] does not
// suppress proc-macro-originated warnings — the crate-level attribute is
// required by the upstream crate's code generation.
#![allow(unused_parens)]

//! LLMOSAFE: Runtime Safety Guardrails
//!
//! A 4-tier safety architecture for systems processing untrusted inputs.
//! Provides three gauges — bias, surprise, entropy — that answer "should I stop?"
//!
//! # Architecture
//!
//! - **Tier 3: Perceptual Sifter** — TF-IDF classifier trained on 42K real samples
//!   detects manipulation patterns. Streaming FNV-1a tokenizer, binary search in
//!   sorted vocab, zero-alloc, `no_std` compatible.
//! - **Tier 2: Working Memory** — Surprise-gated ring buffer with mean, variance,
//!   and trend statistics. Fixed size, no heap.
//! - **Tier 1: Cognitive Kernel** — Binary entropy-based stability check.
//!   Bounded `ReasoningLoop<MAX_STEPS>`. Self-calibrating `DynamicStabilityMonitor`.
//! - **Tier 0: Resource Body** (requires `std`) — RSS memory monitoring,
//!   CPU load tracking, pressure-based escalation.
//!
//! # Quick Usage
//!
//! ```ignore
//! use llmosafe::{sift_perceptions, WorkingMemory, ReasoningLoop};
//!
//! let (sifted, proof) = sift_perceptions(&["observation"], "safety");
//! let mut memory = WorkingMemory::<64>::new(58000);
//! let (validated, proof) = memory.update(sifted, proof)?;
//! let mut loop_guard = ReasoningLoop::<10>::new();
//! loop_guard.next_step(validated, proof)?;
//! ```

#[cfg(not(feature = "std"))]
use core::panic::PanicInfo;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub mod llmosafe_classifier;
pub mod llmosafe_detection;
pub mod llmosafe_integration;
pub mod llmosafe_kernel;
pub mod llmosafe_memory;
pub mod llmosafe_sifter;

#[cfg(feature = "std")]
pub mod llmosafe_body;

#[cfg(feature = "std")]
pub use llmosafe_body::ResourceGuard;
#[cfg(feature = "std")]
pub use llmosafe_detection::DetectionResult;
pub use llmosafe_detection::{
    AdversarialDetector, ConfidenceTracker, CusumDetector, DriftDetector, RepetitionDetector,
};
#[cfg(feature = "std")]
pub use llmosafe_integration::SafetyContext;
pub use llmosafe_integration::{EscalationPolicy, EscalationReason, PressureLevel, SafetyDecision};
pub use llmosafe_kernel::{
    CognitiveEntropy, DynamicStabilityMonitor, KernelError, ReasoningLoop, SiftedProof,
    SiftedSynapse, StabilityResult, Synapse, ValidatedProof, ValidatedSynapse, PRESSURE_THRESHOLD,
    STABILITY_THRESHOLD,
};
pub use llmosafe_memory::WorkingMemory;
pub use llmosafe_sifter::{
    calculate_halo_signal, calculate_utility, get_bias_breakdown, sift_perceptions,
};

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
    // SAFETY: This is a C-ABI entry point. Raw pointer safety is the
    // caller's responsibility; validated below on lines 97-103 before
    // dereference. The function cannot be annotated `unsafe` because
    // that conveys Rust-side UB responsibility, which C callers cannot
    // honor. The #[allow] is function-scoped (narrowest possible) because
    // clippy::not_unsafe_ptr_arg_deref triggers on the extern "C" fn
    // signature, not on the unsafe block within.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub extern "C" fn llmosafe_calculate_halo(text_ptr: *const u8, text_len: usize) -> u16 {
        let max_text_len = 10 * 1024 * 1024;
        if text_ptr.is_null()
            || text_len == 0
            || text_len > isize::MAX as usize
            || text_len > max_text_len
        {
            return 0;
        }
        // SAFETY: text_ptr is validated non-null and text_len is bounded to
        // [1, 10 MiB] on lines 97-103 above. The slice lives only for the duration of
        // from_utf8_lossy below.
        let slice = unsafe { core::slice::from_raw_parts(text_ptr, text_len) };
        let text = String::from_utf8_lossy(slice);
        crate::llmosafe_sifter::calculate_halo_signal(&text)
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_check_resources(ceiling_mb: u32) -> i32 {
        let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
        let guard = ResourceGuard::new(ceiling_bytes);

        match guard.check() {
            Ok(_) => 0,
            Err(KernelError::ResourceExhaustion) => -5,
            Err(KernelError::DepthExceeded) => -1,
            Err(KernelError::CognitiveInstability) => -2,
            Err(KernelError::BiasHaloDetected) => -3,
            Err(KernelError::HallucinationDetected) => -4,
            Err(KernelError::SelfMemoryExceeded) => -6,
            Err(KernelError::DeadlineExceeded) => -7,
        }
    }

    #[no_mangle]
    pub extern "C" fn llmosafe_get_resource_pressure(ceiling_mb: u32) -> u8 {
        let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
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
            Err(KernelError::SelfMemoryExceeded) => -6,
            Err(KernelError::DeadlineExceeded) => -7,
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
        let result = crate::c_abi::llmosafe_calculate_halo(std::ptr::null(), 10);
        assert_eq!(result, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_calculate_halo_zero_length() {
        let data = b"Hello";
        let result = crate::c_abi::llmosafe_calculate_halo(data.as_ptr(), 0);
        assert_eq!(result, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_c_abi_calculate_halo_large_length() {
        let data = b"Hello";
        let result =
            crate::c_abi::llmosafe_calculate_halo(data.as_ptr(), (isize::MAX as usize) + 1);
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
        let invalid_data = b"Hello\\xFFWorld\\0";
        let result =
            crate::c_abi::llmosafe_calculate_halo(invalid_data.as_ptr(), invalid_data.len());
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
        let unstable_bits = 50001u64;
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
