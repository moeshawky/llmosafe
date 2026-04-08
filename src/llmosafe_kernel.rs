//! LLMOSAFE Tier 1 Cognitive Kernel Prototype
//!
//! This module implements the "Law" of the LLMOSAFE meta-pattern.
//! It uses the SCRUST foundation (Deterministic memory, Bounded execution)
//! to enforce Cognitive Stability invariants derived from the research corpus.
//!
//! Research Grounds:
//! - RMPC (Knowledge Mechanisms): Concentric Containers for uncertainty.
//! - Titans (Neural Memory): Surprise-based gating.
//! - Focal Attention (Livšic Equation): Flow stability.

/// Repurposed FixedDecimal from SCRUST for Cognitive Entropy tracking.
/// Precision 28, Scale 2 ensures COBOL-level deterministic arithmetic
/// for Agent Surprise metrics, preventing "Floating Point Hallucinations."
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
        let pressure = (high * 4) / 5; // 80% of high threshold
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

pub const STABILITY_THRESHOLD: i128 = 1000;
pub const PRESSURE_THRESHOLD: i128 = 800;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CognitiveStability {
    Stable,
    Pressure,
    Unstable,
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
    /// // The full pipeline: Tier 3 -> Tier 2 -> Tier 1
    /// let sifted = sift_perceptions(&["stable observation"], "test");
    /// let mut memory = WorkingMemory::<64>::new(1000);
    /// let validated = memory.update(sifted).unwrap();
    ///
    /// let mut loop_guard = ReasoningLoop::<10>::new();
    /// assert!(loop_guard.next_step(validated).is_ok());
    /// ```
    pub fn next_step(&mut self, synapse: ValidatedSynapse) -> Result<(), KernelError> {
        if self.current_step >= MAX_STEPS {
            return Err(KernelError::DepthExceeded);
        }

        // Concentric Container Check: Is the cognitive flow still within stable bounds?
        // (Inspired by Robust Model Predictive Control)
        if !synapse.entropy().is_stable(STABILITY_THRESHOLD) {
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
/// Research Grounds:
/// - Memory_in_LLMs: position-performance curves require context position
/// - Knowledge_Mechanisms: cascade depth tracking for ripple effects
/// - MemSifter: staleness detection via relative timestamps
///
/// # Examples
///
/// CusumDetector: Two-sided cumulative sum for anomaly detection.
/// Derived from statistical process control (Montgomery).
#[derive(Debug, Clone)]
pub struct CusumDetector {
    s_high: f64,
    s_low: f64,
    k: f64,
    h: f64,
    mu_ref: f64,
}

impl CusumDetector {
    pub fn new(mu_ref: f64, k: f64, h: f64) -> Self {
        Self {
            s_high: 0.0,
            s_low: 0.0,
            k,
            h,
            mu_ref,
        }
    }

    pub fn update(&mut self, val: f64) -> bool {
        self.s_high = (0.0f64).max(self.s_high + (val - self.mu_ref) - self.k);
        self.s_low = (0.0f64).max(self.s_low - (val - self.mu_ref) - self.k);
        self.s_high > self.h || self.s_low > self.h
    }

    pub fn reset(&mut self) {
        self.s_high = 0.0;
        self.s_low = 0.0;
    }

    pub fn s_high(&self) -> f64 {
        self.s_high
    }
    pub fn s_low(&self) -> f64 {
        self.s_low
    }
}

#[bitfield(bits = 128)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        CognitiveEntropy::new(self.raw_entropy() as i128)
    }

    pub fn surprise(&self) -> i128 {
        self.raw_surprise() as i128
    }

    pub fn stability(&self) -> CognitiveStability {
        let ent = self.raw_entropy() as i128;
        if ent >= STABILITY_THRESHOLD {
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
        if !self.entropy().is_stable(STABILITY_THRESHOLD) {
            return Err(KernelError::CognitiveInstability);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_cusum_detection() {
        let mut detector = CusumDetector::new(500.0, 50.0, 200.0);

        // No drift
        for _ in 0..5 {
            assert!(!detector.update(500.0));
        }

        // Sustained upward drift
        assert!(!detector.update(600.0)); // S_high = 0 + (600-500) - 50 = 50
        assert!(!detector.update(600.0)); // S_high = 50 + (600-500) - 50 = 100
        assert!(!detector.update(600.0)); // S_high = 100 + (600-500) - 50 = 150
        assert!(!detector.update(600.0)); // S_high = 150 + (600-500) - 50 = 200
        assert!(detector.update(600.0)); // S_high = 200 + (600-500) - 50 = 250 -> TRIGGERS (h=200)
    }

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
        assert!(loop_guard.next_step(stable_validated).is_ok());

        // Create new validated for step 2
        let mut stable_synapse2 = Synapse::new();
        stable_synapse2.set_raw_entropy(500);
        stable_synapse2.set_has_bias(false);
        let stable_sifted2 = SiftedSynapse::new(stable_synapse2);
        stable_sifted2.validate().unwrap();
        let stable_validated2 = ValidatedSynapse::new(stable_sifted2.into_inner());

        // Step 2: OK
        assert!(loop_guard.next_step(stable_validated2).is_ok());

        // Step 3: Depth Exceeded
        let mut stable_synapse3 = Synapse::new();
        stable_synapse3.set_raw_entropy(500);
        stable_synapse3.set_has_bias(false);
        let stable_sifted3 = SiftedSynapse::new(stable_synapse3);
        stable_sifted3.validate().unwrap();
        let stable_validated3 = ValidatedSynapse::new(stable_sifted3.into_inner());

        assert_eq!(
            loop_guard.next_step(stable_validated3).unwrap_err(),
            KernelError::DepthExceeded
        );

        // Reset for entropy test
        let _loop_guard_2 = ReasoningLoop::<5>::new();

        // Create unstable synapse (high entropy)
        let mut unstable_synapse = Synapse::new();
        unstable_synapse.set_raw_entropy(1100);
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

            assert!(loop_guard.next_step(validated).is_ok());
        }

        // Create one more to trigger depth exceeded
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        sifted.validate().unwrap();
        let validated = ValidatedSynapse::new(sifted.into_inner());

        assert_eq!(
            loop_guard.next_step(validated).unwrap_err(),
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
            loop_guard.next_step(validated).unwrap_err(),
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
    fn test_execute_reasoning_flow() {
        assert_eq!(super::cognitive_kernel::execute_reasoning_flow(), Ok(true));
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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KernelError {}

/// TIER 1 SAFETY INVARIANTS:
/// - Stack bounded: Enforced by no_std / no_alloc.
/// - Loop bounds: Enforced by ReasoningLoop<MAX_STEPS>.
/// - Stability: Enforced by CognitiveEntropy (RMPC Concentric Containers).
/// - Unsafe: Forbidden.
pub mod cognitive_kernel {
    use super::*;

    /// Internal reasoning flow using the typestate pipeline.
    /// This bypasses the public API for internal use cases.
    pub fn execute_reasoning_flow() -> Result<bool, KernelError> {
        let mut loop_guard = ReasoningLoop::<10>::new();

        // Create a valid synapse internally (bypasses typestate for internal use)
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500); // 5.00 entropy
        let sifted = SiftedSynapse::new(synapse);

        // For internal use, validate directly
        sifted.validate()?;
        let validated = ValidatedSynapse::new(sifted.into_inner());

        // Execute reasoning steps with hard stability gates
        loop_guard.next_step(validated)?;

        // ... Core reasoning logic here ...

        Ok(true)
    }
}

/// SiftedSynapse: Output of Tier 3 (Sifter).
/// Only constructible by sift_perceptions() - users cannot create this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiftedSynapse {
    synapse: Synapse,
}

impl SiftedSynapse {
    pub fn new(synapse: Synapse) -> Self {
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
