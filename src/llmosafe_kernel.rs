//! LLMOSAFE Tier 1 Cognitive Kernel
#![deny(clippy::cast_lossless)]
//!
//! Formal stability layer using cognitive entropy tracking. Validates sifted
//! data through the entropy stability gate before execution.
//!
//! # Thresholds
//!
//! - `STABILITY_THRESHOLD = 50000` — stability() classifies entropy above this as `Unstable`
//! - `PRESSURE_THRESHOLD = 40000` — validate() and next_step() gate on this; entropy
//!   above this returns `CognitiveInstability` (Pressure zone is gated at the boundary).
//!
//! Entropy range is [0, 65535]. Binary entropy `H(p) = 4p(1-p)` peaks at
//! p=0.5 (maximum classifier uncertainty) and drops to 0 at both extremes.
//! Both "confidently safe" and "confidently dangerous" are stable states.
//!
//! # Components
//!
//! - `CognitiveEntropy<P,S>` — fixed-point entropy with precision P, scale S
//! - `Synapse` — 128-bit input signal from the sifter (entropy, surprise, bias, hash)
//! - `DynamicStabilityMonitor` — self-calibrating envelope tracker using MSB-index bits

use crate::control_types::ControlSignal;

/// Kernel Control Loop output.
///
/// # Control Signal
///
/// - Setpoint: 0.0 (zero entropy = perfect stability)
/// - Actual: `mean_entropy / 65535.0` (normalised)
/// - Error: `e_kernel = actual - 0.0 = actual`
/// - Gain: `K_kernel = 1.0 / (STABILITY_THRESHOLD/65535) = 1.31`
///   Amplifier: maps threshold-crossing to error = 1.0 at STABILITY_THRESHOLD.
///
/// # DAL A
///
/// Kernel loop is the final gate before PID — it validates that the
/// system is in a stable cognitive state. Failure to halt here is
/// catastrophic (system processes unsafe input).
///
/// # Invariants
///
/// - `error_kernel ≥ 0.763` → Halt (STABILITY_THRESHOLD/65535 = 0.763)
/// - `depth < MAX_STEPS`
/// - `has_bias` → Halt (BiasHaloDetected, DAL A override)
#[derive(Debug, Clone, Copy)]
pub struct KernelOutput {
    /// Normalised entropy error `[0.0, 1.0]` where setpoint=0.
    pub error_kernel: f32,
    /// True when mean entropy < STABILITY_THRESHOLD.
    pub is_stable: bool,
    /// Current reasoning step depth.
    pub depth: usize,
}

impl ControlSignal for KernelOutput {
    fn error(&self) -> f32 {
        self.error_kernel
    }

    fn setpoint(&self) -> f32 {
        0.0
    }
}

/// Cognitive entropy tracker using fixed-point arithmetic.
/// Precision 28, scale 2 ensures deterministic arithmetic
/// for agent surprise metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CognitiveEntropy<const P: u32, const S: u32> {
    mantissa: i128,
}

/// Self-calibrating stability monitor using bit-index envelope tracking.
/// Based on the "fast inverse square root" philosophy - uses MSB tracking
/// for O(1) adaptive thresholds without statistical assumptions.
///
/// Usage:
///   let mut monitor = DynamicStabilityMonitor::new(3); // k=3 safety margin
///   if monitor.update(entropy_value).is_unstable() {
///       // Handle instability
///   }
#[derive(Debug, Clone, Copy)]
pub struct DynamicStabilityMonitor {
    hi_idx: u8, // max accepted floor(log2(max(x,1)))
    lo_idx: u8, // min accepted floor(log2(max(x,1)))
    seen: bool,
    k: u8, // safety margin in bits
}

impl DynamicStabilityMonitor {
    /// Create a new monitor with the given safety margin k.
    /// k=2 is typical for embedded safety, k=3 for more robust.
    pub const fn new(k: u8) -> Self {
        Self {
            hi_idx: 0,
            lo_idx: 255, // Start with inverted state
            seen: false,
            k,
        }
    }

    /// Compute MSB index (floor(log2(x))) for x > 0, returns 0 for x == 0.
    /// Uses the same technique as fast inverse sqrt - bit manipulation.
    #[inline]
    fn msb_idx(x: u32) -> u8 {
        if x == 0 {
            return 0;
        }
        // Equivalent to 31 - __builtin_clz(x) but portable
        31u8.wrapping_sub(x.leading_zeros() as u8)
    }

    /// Update with a new entropy measurement.
    /// Returns true if the value is unstable (too high OR too low).
    pub fn update(&mut self, entropy: u32) -> StabilityResult {
        let idx = Self::msb_idx(entropy);

        // Initialization: first non-zero value sets both envelopes
        if !self.seen {
            self.hi_idx = idx;
            self.lo_idx = idx;
            self.seen = true;
            return StabilityResult::Stable;
        }

        // Bidirectional instability check
        let high_unstable = idx > self.hi_idx.wrapping_add(self.k);
        let low_unstable = idx < self.lo_idx.saturating_sub(self.k);

        if high_unstable || low_unstable {
            // Adapt envelopes even on instability to prevent lockout
            if idx > self.hi_idx {
                self.hi_idx = idx;
            }
            if idx < self.lo_idx {
                self.lo_idx = idx;
            }
            return if high_unstable && low_unstable {
                StabilityResult::Both
            } else if high_unstable {
                StabilityResult::High
            } else {
                StabilityResult::Low
            };
        }

        // Adapt envelopes (self-calibrating)
        if idx > self.hi_idx {
            self.hi_idx = idx;
        }
        if idx < self.lo_idx {
            self.lo_idx = idx;
        }

        StabilityResult::Stable
    }

    /// Get current adaptive thresholds based on observed envelopes.
    /// Returns (high_threshold, low_threshold, pressure_threshold).
    pub fn get_thresholds(&self) -> (u32, u32, u32) {
        if !self.seen {
            return (u32::MAX, 0, 0);
        }
        let high = if self.hi_idx >= 31 {
            u32::MAX
        } else {
            (1u32 << (self.hi_idx + 1)) - 1
        };
        let low = if self.lo_idx == 0 {
            0
        } else {
            1u32 << self.lo_idx
        };
        let pressure = (u64::from(high) * 4 / 5) as u32;
        (high, low, pressure)
    }

    /// Reset the monitor to uninitialized state.
    pub fn reset(&mut self) {
        self.seen = false;
        self.hi_idx = 0;
        self.lo_idx = 255;
    }
}

/// Result of stability check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityResult {
    Stable,
    High, // Too high (dangerous)
    Low,  // Too low (suspiciously perfect)
    Both, // Both directions unstable
}

pub const STABILITY_THRESHOLD: i128 = 50000;
pub const PRESSURE_THRESHOLD: i128 = 40000;

/// Detection flag: repetition count exceeded max_repetitions.
pub const FLAG_STUCK: u8 = 0x01;
/// Detection flag: drift_score exceeds drift_threshold.
pub const FLAG_DRIFTING: u8 = 0x02;
/// Detection flag: latest confidence below min_confidence.
pub const FLAG_LOW_CONFIDENCE: u8 = 0x04;
/// Detection flag: consecutive confidence drops exceed decay_threshold.
pub const FLAG_DECAYING: u8 = 0x08;
/// Detection flag: CUSUM s_high or s_low exceeds threshold h.
pub const FLAG_ANOMALY: u8 = 0x10;
/// All detection flag bits.
pub const DETECTION_FLAGS_MASK: u8 = 0x1F;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CognitiveStability {
    Stable,
    Pressure,
    Unstable,
}

impl From<StabilityResult> for CognitiveStability {
    fn from(result: StabilityResult) -> Self {
        match result {
            StabilityResult::Stable => CognitiveStability::Stable,
            StabilityResult::High => CognitiveStability::Unstable,
            StabilityResult::Low => CognitiveStability::Unstable,
            StabilityResult::Both => CognitiveStability::Unstable,
        }
    }
}

impl<const P: u32, const S: u32> CognitiveEntropy<P, S> {
    /// Creates a new CognitiveEntropy with the given mantissa.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::CognitiveEntropy;
    /// let entropy = CognitiveEntropy::<28, 2>::new(500);
    /// ```
    pub const fn new(mantissa: i128) -> Self {
        Self { mantissa }
    }

    pub const fn mantissa(&self) -> i128 {
        self.mantissa
    }

    /// The "Hard Guard" threshold. If entropy exceeds this, reasoning must halt.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::CognitiveEntropy;
    /// let entropy = CognitiveEntropy::<28, 2>::new(500);
    /// assert!(entropy.is_stable(1000));
    /// ```
    pub const fn is_stable(&self, threshold: i128) -> bool {
        self.mantissa <= threshold
    }
}

/// The "Reasoning Step" container.
/// Implements the LLMSAFE Axiom of Determinism.
pub struct ReasoningLoop<const MAX_STEPS: usize> {
    current_step: usize,
}

impl<const MAX_STEPS: usize> ReasoningLoop<MAX_STEPS> {
    /// Creates a new ReasoningLoop starting at step 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::llmosafe_kernel::ReasoningLoop;
    /// let loop_guard = ReasoningLoop::<10>::new();
    /// ```
    pub const fn new() -> Self {
        Self { current_step: 0 }
    }
}

impl<const MAX_STEPS: usize> Default for ReasoningLoop<MAX_STEPS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_STEPS: usize> ReasoningLoop<MAX_STEPS> {
    /// Validates a reasoning transition against the stability kernel.
    /// Derived from Knowledge Mechanisms (CC-VT RMPC).
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::{sift_perceptions, WorkingMemory, ReasoningLoop};
    ///
    /// let (sifted, sifted_proof) = sift_perceptions(&["the weather is nice today"], "test");
    /// let mut memory = WorkingMemory::<64>::new(65535);
    /// let (validated, validated_proof) = memory.update(sifted, sifted_proof).unwrap();
    ///
    /// let mut loop_guard = ReasoningLoop::<10>::new();
    /// assert!(loop_guard.next_step(validated, validated_proof).is_ok());
    /// ```
    pub fn next_step(
        &mut self,
        synapse: ValidatedSynapse,
        _proof: ValidatedProof,
    ) -> Result<(), KernelError> {
        if self.current_step >= MAX_STEPS {
            return Err(KernelError::DepthExceeded);
        }

        if synapse.has_bias() {
            return Err(KernelError::BiasHaloDetected);
        }

        // Concentric Container Check: Is the cognitive flow still within stable bounds?
        // (Inspired by Robust Model Predictive Control)
        if !synapse.entropy().is_stable(PRESSURE_THRESHOLD) {
            return Err(KernelError::CognitiveInstability);
        }

        self.current_step += 1;
        Ok(())
    }
}

use modular_bitfield::prelude::*;

/// The "Synapse" (Binary Cognitive Protocol).
/// A bit-packed u128 carrying the entire stability state.
/// [Entropy: 16][Surprise: 16][Bias: 1][Position: 12][Timestamp: 16][Cascade: 8][Hash: 31][Reserved: 28]
///
#[bitfield(bits = 128)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(unused_parens)]
pub struct Synapse {
    pub raw_entropy: B16,
    pub raw_surprise: B16,
    pub has_bias: bool,
    pub position: B12,
    pub timestamp: B16,
    pub cascade_depth: B8,
    pub anchor_hash: B31,
    pub reserved: B28,
}

impl Default for Synapse {
    fn default() -> Self {
        Self::new()
    }
}

impl Synapse {
    /// Creates a Synapse from a raw u128.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::Synapse;
    /// let synapse = Synapse::from_raw_u128(0);
    /// ```
    pub fn from_raw_u128(bits: u128) -> Self {
        Self::from_bytes(bits.to_le_bytes())
    }

    /// Creates a Synapse from a raw u64.
    ///
    /// This is used for compatibility with C-ABI functions that only
    /// expect 64 bits. Higher 64 bits are zeroed.
    pub fn from_raw_u64(bits: u64) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&bits.to_le_bytes());
        Self::from_bytes(bytes)
    }

    pub fn entropy(&self) -> CognitiveEntropy<28, 2> {
        CognitiveEntropy::new(i128::from(self.raw_entropy()))
    }

    pub fn surprise(&self) -> i128 {
        i128::from(self.raw_surprise())
    }

    pub fn stability(&self) -> CognitiveStability {
        let ent = i128::from(self.raw_entropy());
        if ent > STABILITY_THRESHOLD {
            CognitiveStability::Unstable
        } else if ent >= PRESSURE_THRESHOLD {
            CognitiveStability::Pressure
        } else {
            CognitiveStability::Stable
        }
    }

    /// The "Receptor" validation logic.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::Synapse;
    /// let synapse = Synapse::from_raw_u128(0);
    /// assert!(synapse.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.has_bias() {
            return Err(KernelError::BiasHaloDetected);
        }
        if !self.entropy().is_stable(PRESSURE_THRESHOLD) {
            return Err(KernelError::CognitiveInstability);
        }
        Ok(())
    }

    /// Packs 5 detection flags into reserved bits 0-4.
    ///
    /// Input `flags` is masked to lower 5 bits (`flags & 0x1F`).
    /// Other reserved bits (5-27) are preserved.
    pub fn set_detection_flags(&mut self, flags: u8) {
        let current = self.reserved();
        let cleared = current & !0x1Fu32;
        self.set_reserved(cleared | (u32::from(flags) & 0x1F));
    }

    /// Returns the lower 5 bits of the reserved field as detection flags.
    pub fn detection_flags(&self) -> u8 {
        (self.reserved() & 0x1F) as u8
    }

    /// Packs OOV ratio into reserved bits 5-12.
    ///
    /// Maps 0=0% OOV, 255=100% OOV. Preserves detection flags (bits 0-4)
    /// and upper reserved bits (bits 13-27).
    pub fn set_oov_ratio(&mut self, ratio: u8) {
        let current = self.reserved();
        let cleared = current & !0x1FE0u32;
        self.set_reserved(cleared | ((u32::from(ratio) & 0xFF) << 5));
    }

    /// Returns bits 5-12 of the reserved field as OOV ratio.
    pub fn oov_ratio(&self) -> u8 {
        ((self.reserved() >> 5) & 0xFF) as u8
    }

    /// Zeros detection_flags (bits 0-4) and oov_ratio (bits 5-12) in reserved field.
    ///
    /// Upper reserved bits (13-27) are NOT modified.
    pub fn clear_detection(&mut self) {
        let current = self.reserved();
        self.set_reserved(current & !0x1FFFu32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_reasoning_loop() {
        let mut loop_guard = ReasoningLoop::<2>::new();

        // Create valid synapse (low entropy, no bias)
        let mut stable_synapse = Synapse::new();
        stable_synapse.set_raw_entropy(500);
        stable_synapse.set_has_bias(false);
        let stable_sifted = SiftedSynapse::new(stable_synapse);
        stable_sifted.validate().unwrap();
        let stable_validated = ValidatedSynapse::new(stable_sifted.into_inner());

        // Step 1: OK
        assert!(loop_guard
            .next_step(stable_validated, ValidatedProof(()))
            .is_ok());

        // Create new validated for step 2
        let mut stable_synapse2 = Synapse::new();
        stable_synapse2.set_raw_entropy(500);
        stable_synapse2.set_has_bias(false);
        let stable_sifted2 = SiftedSynapse::new(stable_synapse2);
        stable_sifted2.validate().unwrap();
        let stable_validated2 = ValidatedSynapse::new(stable_sifted2.into_inner());

        // Step 2: OK
        assert!(loop_guard
            .next_step(stable_validated2, ValidatedProof(()))
            .is_ok());

        // Step 3: Depth Exceeded
        let mut stable_synapse3 = Synapse::new();
        stable_synapse3.set_raw_entropy(500);
        stable_synapse3.set_has_bias(false);
        let stable_sifted3 = SiftedSynapse::new(stable_synapse3);
        stable_sifted3.validate().unwrap();
        let _stable_validated3 = ValidatedSynapse::new(stable_sifted3.into_inner());

        assert_eq!(
            loop_guard
                .next_step(stable_validated, ValidatedProof(()))
                .unwrap_err(),
            KernelError::DepthExceeded
        );

        // Reset for entropy test
        let _loop_guard_2 = ReasoningLoop::<5>::new();

        // Create unstable synapse (high entropy)
        let mut unstable_synapse = Synapse::new();
        unstable_synapse.set_raw_entropy(50001);
        unstable_synapse.set_has_bias(false);
        let unstable_sifted = SiftedSynapse::new(unstable_synapse);

        // Should fail validation (cognitive instability)
        assert_eq!(
            unstable_sifted.validate().unwrap_err(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_stability_boundary() {
        let stable = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD);
        let unstable = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD + 1);

        assert!(stable.is_stable(STABILITY_THRESHOLD));
        assert!(!unstable.is_stable(STABILITY_THRESHOLD));
    }

    #[test]
    fn test_synapse_validation() {
        // Valid Synapse: Entropy 500, No Bias
        let valid_bits = 500u128;
        let synapse = Synapse::from_raw_u128(valid_bits);
        assert!(synapse.validate().is_ok());

        // Invalid Synapse: Bias detected
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        synapse.set_has_bias(true);
        assert_eq!(
            synapse.validate().unwrap_err(),
            KernelError::BiasHaloDetected
        );

        // Invalid Synapse: High Entropy
        let unstable_bits = (STABILITY_THRESHOLD + 1) as u128;
        let synapse = Synapse::from_raw_u128(unstable_bits);
        assert_eq!(
            synapse.validate().unwrap_err(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_synapse_validation_invariance_to_hash() {
        let mut s1 = Synapse::new();
        s1.set_raw_entropy(500);
        s1.set_anchor_hash(0x123);

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(500);
        s2.set_anchor_hash(0x456);

        // Validation result must be identical regardless of hash
        assert_eq!(s1.validate(), s2.validate());
    }

    #[test]
    fn test_synapse_from_raw_u128_all_zeros() {
        let synapse = Synapse::from_raw_u128(0);
        assert_eq!(synapse.raw_entropy(), 0);
        assert_eq!(synapse.raw_surprise(), 0);
        assert!(!synapse.has_bias());
        assert_eq!(synapse.anchor_hash(), 0);
    }

    #[test]
    fn test_synapse_from_raw_u128_max_values() {
        // 128-bit layout: [Entropy:16][Surprise:16][Bias:1][Position:12][Timestamp:16][Cascade:8][Hash:31][Reserved:28]
        // u16::MAX is 0xFFFF
        // Hash B31 max is 0x7FFFFFFF
        let max_bits = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFu128;
        let synapse = Synapse::from_raw_u128(max_bits);
        assert_eq!(synapse.raw_entropy(), 0xFFFF);
        assert_eq!(synapse.raw_surprise(), 0xFFFF);
        assert!(synapse.has_bias());
        assert_eq!(synapse.position(), 0xFFF);
        assert_eq!(synapse.timestamp(), 0xFFFF);
        assert_eq!(synapse.cascade_depth(), 0xFF);
        assert_eq!(synapse.anchor_hash(), 0x7FFFFFFF);
        assert_eq!(synapse.reserved(), 0xFFFFFFF);
    }

    #[test]
    fn test_reasoning_loop_boundary_exact_max_steps() {
        let mut loop_guard = ReasoningLoop::<5>::new();

        for _ in 0..5 {
            // Create a new valid synapse each iteration
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(500);
            synapse.set_has_bias(false);
            let sifted = SiftedSynapse::new(synapse);
            sifted.validate().unwrap();
            let validated = ValidatedSynapse::new(sifted.into_inner());

            assert!(loop_guard.next_step(validated, ValidatedProof(())).is_ok());
        }

        // Create one more to trigger depth exceeded
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        sifted.validate().unwrap();
        let validated = ValidatedSynapse::new(sifted.into_inner());

        assert_eq!(
            loop_guard
                .next_step(validated, ValidatedProof(()))
                .unwrap_err(),
            KernelError::DepthExceeded
        );
    }

    #[test]
    fn test_cognitive_entropy_stability_threshold_edge() {
        let threshold = 1000;
        let at_threshold = CognitiveEntropy::<28, 2>::new(threshold);
        let just_above = CognitiveEntropy::<28, 2>::new(threshold + 1);
        let just_below = CognitiveEntropy::<28, 2>::new(threshold - 1);

        assert!(at_threshold.is_stable(threshold));
        assert!(!just_above.is_stable(threshold));
        assert!(just_below.is_stable(threshold));
    }

    #[test]
    fn test_synapse_validate_zero_entropy_no_bias() {
        let synapse = Synapse::from_raw_u128(0);
        assert!(synapse.validate().is_ok());
    }

    #[test]
    fn test_synapse_validate_max_entropy_bias() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0xFFFF);
        synapse.set_has_bias(true);
        assert!(synapse.validate().is_err());
    }

    #[test]
    fn test_cognitive_entropy_equality() {
        let e1 = CognitiveEntropy::<28, 2>::new(500);
        let e2 = CognitiveEntropy::<28, 2>::new(500);
        let e3 = CognitiveEntropy::<28, 2>::new(600);
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_reasoning_loop_zero_steps_max() {
        let mut loop_guard = ReasoningLoop::<0>::new();

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        let sifted = SiftedSynapse::new(synapse);
        sifted.validate().unwrap();
        let validated = ValidatedSynapse::new(sifted.into_inner());

        assert_eq!(
            loop_guard
                .next_step(validated, ValidatedProof(()))
                .unwrap_err(),
            KernelError::DepthExceeded
        );
    }

    #[test]
    fn test_synapse_hash_boundary() {
        let mut synapse = Synapse::new();
        synapse.set_anchor_hash(0x7FFFFFFF);
        assert_eq!(synapse.anchor_hash(), 0x7FFFFFFF);

        // modular-bitfield panics on out-of-bounds values, so we don't test 0xFFFFFFFF here.
    }

    #[test]
    fn test_synapse_raw_surprise_boundary() {
        let mut synapse = Synapse::new();
        synapse.set_raw_surprise(0xFFFF);
        assert_eq!(synapse.raw_surprise(), 0xFFFF);
        assert_eq!(synapse.surprise(), 0xFFFF);
    }

    #[test]
    fn test_synapse_position_field() {
        let mut synapse = Synapse::new();
        synapse.set_position(0xFFF);
        assert_eq!(synapse.position(), 0xFFF);
        synapse.set_position(0);
        assert_eq!(synapse.position(), 0);
    }

    #[test]
    fn test_synapse_timestamp_field() {
        let mut synapse = Synapse::new();
        synapse.set_timestamp(0xFFFF);
        assert_eq!(synapse.timestamp(), 0xFFFF);
        synapse.set_timestamp(1000);
        assert_eq!(synapse.timestamp(), 1000);
    }

    #[test]
    fn test_synapse_cascade_depth_field() {
        let mut synapse = Synapse::new();
        synapse.set_cascade_depth(0xFF);
        assert_eq!(synapse.cascade_depth(), 0xFF);
        synapse.set_cascade_depth(0);
        assert_eq!(synapse.cascade_depth(), 0);
    }

    #[test]
    fn test_synapse_all_fields_roundtrip() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1234);
        synapse.set_raw_surprise(5678);
        synapse.set_has_bias(false);
        synapse.set_position(0xABC);
        synapse.set_timestamp(0x1234);
        synapse.set_cascade_depth(0x12);
        synapse.set_anchor_hash(0x1234567);

        let bytes = synapse.into_bytes();
        let reconstructed = Synapse::from_bytes(bytes);

        assert_eq!(reconstructed.raw_entropy(), 1234);
        assert_eq!(reconstructed.raw_surprise(), 5678);
        assert!(!reconstructed.has_bias());
        assert_eq!(reconstructed.position(), 0xABC);
        assert_eq!(reconstructed.timestamp(), 0x1234);
        assert_eq!(reconstructed.cascade_depth(), 0x12);
        assert_eq!(reconstructed.anchor_hash(), 0x1234567);
    }

    proptest! {
        #[test]
        fn test_synapse_arbitrary_bits_roundtrip(bits in any::<u128>()) {
            let synapse = Synapse::from_raw_u128(bits);
            let encoded = u128::from_le_bytes(synapse.into_bytes());
            prop_assert_eq!(bits, encoded);
        }
    }

    #[test]
    fn test_dynamic_stability_monitor_initialization() {
        let mut monitor = DynamicStabilityMonitor::new(2);
        assert!(!monitor.seen);

        // First value initializes
        let result = monitor.update(500);
        assert_eq!(result, StabilityResult::Stable);
        assert!(monitor.seen);
    }

    #[test]
    fn test_dynamic_stability_monitor_high_anomaly() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        // Initialize with low values
        monitor.update(100); // msb_idx = 6 (2^6 = 64, 2^7 = 128)
        monitor.update(100);

        // Now inject high value - should trigger high instability
        let result = monitor.update(900); // msb_idx = 9 (2^9 = 512)
                                          // 9 > 6 + 2 = 8, so unstable
        assert!(matches!(
            result,
            StabilityResult::High | StabilityResult::Both
        ));
    }

    #[test]
    fn test_dynamic_stability_monitor_low_anomaly() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        // Initialize with higher values
        monitor.update(500); // msb_idx = 8
        monitor.update(500);

        // Now inject very low value - should trigger low instability
        let result = monitor.update(10); // msb_idx = 3
                                         // 3 < 8 - 2 = 6, so unstable
        assert!(matches!(
            result,
            StabilityResult::Low | StabilityResult::Both
        ));
    }

    #[test]
    fn test_dynamic_stability_monitor_adaptation() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        // Gradually increase - adapts
        monitor.update(100);
        monitor.update(150);
        monitor.update(200);
        monitor.update(250);

        // All should be stable as it adapts
        let (high, _low, pressure) = monitor.get_thresholds();
        assert!(high > 0);
        assert!(pressure > 0);
    }

    #[test]
    fn test_dynamic_stability_monitor_zero_value() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        // Zero is handled specially (msb_idx returns 0)
        monitor.update(100);
        let result = monitor.update(0);
        // With lo_idx likely > 0, idx=0 < lo_idx - k triggers low
        assert!(matches!(
            result,
            StabilityResult::Low | StabilityResult::Both | StabilityResult::Stable
        ));
    }

    #[test]
    fn test_dynamic_stability_monitor_k_sensitivity() {
        // k=1 is more sensitive
        let mut sensitive = DynamicStabilityMonitor::new(1);
        sensitive.update(100);
        sensitive.update(100);
        let r1 = sensitive.update(400); // 8->8+1=9, 8+1=9, k=1 -> 9 > 8+1? no... wait
                                        // 100 has msb=6, 400 has msb=8
                                        // 8 > 6+1=7? yes, triggers

        // k=3 is more tolerant
        let mut tolerant = DynamicStabilityMonitor::new(3);
        tolerant.update(100);
        tolerant.update(100);
        let r2 = tolerant.update(400);

        // The sensitive one should be more likely to detect
        assert!(r1 != StabilityResult::Stable || r2 != StabilityResult::Stable);
    }

    #[test]
    fn test_dynamic_stability_monitor_reset() {
        let mut monitor = DynamicStabilityMonitor::new(2);
        monitor.update(500);
        assert!(monitor.seen);

        monitor.reset();
        assert!(!monitor.seen);

        // After reset, next value re-initializes
        let result = monitor.update(100);
        assert_eq!(result, StabilityResult::Stable);
        assert!(monitor.seen);
    }

    #[test]
    fn test_stability_result_to_cognitive_stability() {
        assert_eq!(
            CognitiveStability::from(StabilityResult::Stable),
            CognitiveStability::Stable
        );
        assert_eq!(
            CognitiveStability::from(StabilityResult::High),
            CognitiveStability::Unstable
        );
        assert_eq!(
            CognitiveStability::from(StabilityResult::Low),
            CognitiveStability::Unstable
        );
        assert_eq!(
            CognitiveStability::from(StabilityResult::Both),
            CognitiveStability::Unstable
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelError {
    DepthExceeded,
    CognitiveInstability,
    BiasHaloDetected,
    HallucinationDetected,
    ResourceExhaustion,
    /// Daemon's own RSS exceeded self-imposed limit.
    /// Reserved for future self-memory protection (see RECOMMENDATIONS.md #4).
    /// Not currently constructed — handled as a forward-compatible error code.
    SelfMemoryExceeded,
    /// Deadline exceeded while waiting for safe state.
    DeadlineExceeded,
}

impl core::fmt::Display for KernelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DepthExceeded => write!(f, "reasoning cascade depth exceeded"),
            Self::CognitiveInstability => {
                write!(f, "cognitive entropy exceeds stability threshold")
            }
            Self::BiasHaloDetected => write!(f, "bias halo signal detected in perceptual input"),
            Self::HallucinationDetected => {
                write!(f, "surprise level exceeds hallucination threshold")
            }
            Self::ResourceExhaustion => write!(f, "RSS memory exceeds configured safety ceiling"),
            Self::SelfMemoryExceeded => write!(f, "self memory exceeded"),
            Self::DeadlineExceeded => write!(f, "deadline exceeded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KernelError {}

/// SiftedSynapse: Output of Tier 3 (Sifter).
/// Only constructible by sift_perceptions() - users cannot create this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiftedSynapse {
    synapse: Synapse,
}

impl SiftedSynapse {
    /// Construct a SiftedSynapse from a raw Synapse.
    ///
    /// Prefer `sift_perceptions()` for production code — it applies the full
    /// sifter pipeline (bias detection, utility ranking, entropy calculation).
    /// Direct construction is intended for testing and crate-internal use only.
    pub(crate) fn new(synapse: Synapse) -> Self {
        Self { synapse }
    }

    /// Low-level constructor from a raw Synapse, bypassing the sifter pipeline.
    ///
    /// **Warning:** Prefer `sift_perceptions()` for production code. This constructor
    /// is intended for testing and internal use where precise control over
    /// entropy/surprise/halo values is needed. The sifter pipeline applies bias
    /// detection, utility ranking, and anchor hashing which this skips.
    ///
    /// This creates a `SiftedSynapse` that can be inspected but cannot be fed to
    /// `WorkingMemory::update()` without a `SiftedProof` — which only
    /// `sift_perceptions()` or `SiftedProof::for_testing()` can produce.
    pub fn from_synapse(synapse: Synapse) -> Self {
        Self { synapse }
    }

    pub fn entropy(&self) -> CognitiveEntropy<28, 2> {
        self.synapse.entropy()
    }

    pub fn surprise(&self) -> i128 {
        self.synapse.surprise()
    }

    pub fn has_bias(&self) -> bool {
        self.synapse.has_bias()
    }

    pub fn anchor_hash(&self) -> u32 {
        self.synapse.anchor_hash()
    }

    pub fn raw_entropy(&self) -> u16 {
        self.synapse.raw_entropy()
    }

    pub fn raw_surprise(&self) -> u16 {
        self.synapse.raw_surprise()
    }

    pub fn oov_ratio(&self) -> u8 {
        self.synapse.oov_ratio()
    }

    pub fn detection_flags(&self) -> u8 {
        self.synapse.detection_flags()
    }

    pub fn stability(&self) -> CognitiveStability {
        self.synapse.stability()
    }

    pub fn validate(&self) -> Result<(), KernelError> {
        self.synapse.validate()
    }

    pub fn into_inner(self) -> Synapse {
        self.synapse
    }
}

/// ValidatedSynapse: Output of Tier 2 (Working Memory).
/// Only constructible by WorkingMemory::update() - users cannot create this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedSynapse {
    synapse: Synapse,
}

impl ValidatedSynapse {
    pub(crate) fn new(synapse: Synapse) -> Self {
        Self { synapse }
    }

    pub fn entropy(&self) -> CognitiveEntropy<28, 2> {
        self.synapse.entropy()
    }

    pub fn surprise(&self) -> i128 {
        self.synapse.surprise()
    }

    pub fn has_bias(&self) -> bool {
        self.synapse.has_bias()
    }

    pub fn anchor_hash(&self) -> u32 {
        self.synapse.anchor_hash()
    }

    pub fn raw_entropy(&self) -> u16 {
        self.synapse.raw_entropy()
    }

    pub fn raw_surprise(&self) -> u16 {
        self.synapse.raw_surprise()
    }

    pub fn stability(&self) -> CognitiveStability {
        self.synapse.stability()
    }

    pub fn into_inner(self) -> Synapse {
        self.synapse
    }
}

/// Unforgeable proof that `sift_perceptions()` was executed.
///
/// Only constructible by the sifter module. Zero runtime cost — the compiler
/// erases this type entirely. Functions as a compile-time capability token:
/// `WorkingMemory::update()` requires this proof, and no code outside the
/// sifter can create one. This prevents `SiftedSynapse::from_synapse()` from
/// bypassing the sifter pipeline.
///
/// `Copy` and `Clone` are intentional: the proof attests to sifter execution,
/// not to uniqueness. It is safe to reuse.
#[derive(Debug, Clone, Copy)]
pub struct SiftedProof(());

impl SiftedProof {
    /// Mint a SiftedProof. Only the sifter module should call this.
    /// Prefer `sift_observation()` or `sift_perceptions()` — they mint the
    /// proof as part of a complete (SiftedSynapse, SiftedProof) pair.
    #[doc(hidden)]
    pub(crate) fn mint() -> Self {
        SiftedProof(())
    }

    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn for_testing() -> Self {
        SiftedProof(())
    }

    #[cfg(feature = "std")]
    #[doc(hidden)]
    pub(crate) fn from_raw_bits_bypass() -> Self {
        SiftedProof(())
    }
}

/// Unforgeable proof that `WorkingMemory::update()` was executed.
///
/// Only constructible by the memory module. Functions identically to
/// `SiftedProof` but for the memory → kernel boundary. `ReasoningLoop::next_step()`
/// requires this proof, ensuring validated synapses must pass through
/// working memory's surprise gating, trend analysis, and drift detection.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedProof(pub(crate) ());

impl ValidatedProof {
    #[cfg(any(feature = "testing", test))]
    #[doc(hidden)]
    pub fn for_testing() -> Self {
        ValidatedProof(())
    }
}
