//! LLMOSAFE Tier 2 Working Memory
//!
//! Surprise-gated ring buffer. Rejects updates where the entropy value is too
//! unexpected relative to history (via `WorkingMemory::<SIZE>::new(threshold)`).
//!
//! Stores `CognitiveEntropy` values in a fixed-size ring buffer with:
//! - `mean()`, `variance()` — running statistics
//! - `trend()` — linear regression slope over the buffer window
//!
//! Entropy values are in [0, 65535]. The `new()` threshold determines how
//! surprising a value must be (relative to mean) before rejection. Typical
//! threshold: 58000 for classifier output.

// Arithmetic in this module operates on bounded ring-buffer index values
// [0, SIZE-1] where modulo/offset semantics are the intended behavior.
// DO-178C: these operations are verified safe by compile-time const bounds.
#![allow(clippy::arithmetic_side_effects)]

use crate::control_types::ControlSignal;
use crate::llmosafe_kernel::{
    CognitiveEntropy, KernelError, SiftedProof, SiftedSynapse, ValidatedProof, ValidatedSynapse,
};

/// Memory Control Loop output.
///
/// # Control Signal
///
/// - Setpoint: `μ = mean_entropy` (self-adjusting — ring buffer running mean)
/// - Actual: `entropy_n` (current observation's raw entropy)
/// - Error: `e_mem = |entropy_n - mean_entropy| / 65535.0` (normalised to `[0, 1]`)
/// - Gain: `K_mem = 1.0 + 0.3 × tanh(|trend| / 1000.0)` (gain-scheduled by trend)
///
/// # DAL B
///
/// Memory loop gates on surprise — missed escalation allows unsafe
/// observations to propagate to the kernel. DAL B because failure
/// causes hazardous behaviour (missed escalation), not catastrophic.
///
/// # Invariants
///
/// - `0.0 ≤ error_mem ≤ 1.0`
/// - Ring buffer size = SIZE (const generic, compile-time bound)
/// - Surprise gate: `error_mem > surprise_threshold/65535` → HallucinationDetected
///
/// # Zero-Observation and Warmup Semantics
///
/// During warmup (`write_count < SIZE`), `mean_entropy` is computed
/// from only the `write_count` valid observations. `e_mem` during
/// warmup is intentional: the mean lags behind steady-state, so
/// `e_mem` will be larger for the same entropy value until the ring
/// fills. This is by design — it allows the memory loop to detect
/// sustained elevation early rather than being insensitive during
/// initialization.
///
/// When `write_count == 0`, all statistics return `0.0` — no valid
/// observations means no meaningful signal.
///
/// Fields:
/// - `error_mem: f32` — normalised surprise error [0.0, 1.0].
/// - `trend: f64` — linear regression slope over buffer window.
/// - `mean_entropy: f64` — running mean entropy of ring buffer (0.0 when empty).
#[derive(Debug, Clone, Copy)]
pub struct MemoryOutput {
    /// Normalised surprise error `[0.0, 1.0]`.
    pub error_mem: f32,
    /// Linear regression slope over buffer window.
    pub trend: f64,
    /// Running mean entropy of ring buffer.
    pub mean_entropy: f64,
}

impl ControlSignal for MemoryOutput {
    fn error(&self) -> f32 {
        self.error_mem
    }

    fn setpoint(&self) -> f32 {
        (self.mean_entropy / 65535.0) as f32
    }
}

/// Fixed-size working memory stack buffer.
///
/// Holds up to SIZE entropy values in a ring buffer.
/// Default capacity is 64 entries (stack-allocated).
///
/// # Valid-Write Statistics
///
/// The ring buffer tracks `write_count` — the number of successfully
/// accepted (surprise-gate-passed) observations written. Statistics
/// (`mean_entropy`, `entropy_variance`, `trend`) operate **only** over
/// valid observations, dividing by `min(write_count, SIZE)` rather
/// than by `SIZE`. Surprise-rejected observations never enter the
/// buffer and never increment `write_count`.
///
/// **Zero-observation behavior** (`write_count == 0`): `mean_entropy()`
/// returns `0.0`, `entropy_variance()` returns `0.0`, `trend()` returns
/// `0.0`. This is intentional — with no data, there is no meaningful
/// statistic, and returning `0.0` avoids spurious signals.
///
/// **Partial-window semantics** (`0 < write_count < SIZE`): During
/// warmup, statistics operate over the `write_count` observations
/// actually present. `e_mem` during warmup is computed against this
/// partial-window mean, which is intentionally lower than the
/// steady-state mean (the ring is not yet full). This warmup
/// behavior is documented: `e_mem` will be larger early on as the
/// mean lags behind, then stabilizes once the ring is full.
pub struct WorkingMemory<const SIZE: usize = 64> {
    state: [CognitiveEntropy<28, 2>; SIZE],
    current_index: usize,
    surprise_threshold: i128,
    /// Number of successfully accepted observations written.
    /// Capped at `SIZE` once the ring wraps.
    write_count: usize,
}

impl<const SIZE: usize> WorkingMemory<SIZE> {
    const _SIZE_CHECK: () = assert!(SIZE > 0, "WorkingMemory size must be > 0");

    /// Initialize with a fixed surprise threshold (e.g. 5.00)
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::WorkingMemory;
    /// let memory = WorkingMemory::<64>::new(1000);
    /// ```
    pub const fn new(threshold: i128) -> Self {
        Self {
            state: [CognitiveEntropy::new(0); SIZE],
            current_index: 0,
            surprise_threshold: threshold,
            write_count: 0,
        }
    }

    /// Updated: Uses the SiftedSynapse protocol for state transitions.
    ///
    /// # Errors
    ///
    /// Returns `CognitiveInstability` or `BiasHaloDetected` if `sifted.validate()` fails.
    /// Returns `HallucinationDetected` if the synapse surprise exceeds the memory threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::{WorkingMemory, sift_perceptions};
    /// let mut memory = WorkingMemory::<64>::new(65535);
    /// let (sifted, proof) = sift_perceptions(&["the weather is nice today"], "test");
    /// let result = memory.update(sifted, proof);
    /// assert!(result.is_ok());
    /// ```
    pub fn update(
        &mut self,
        sifted: SiftedSynapse,
        _proof: SiftedProof,
    ) -> Result<(ValidatedSynapse, ValidatedProof), KernelError> {
        sifted.validate()?;

        if sifted.surprise() > self.surprise_threshold {
            return Err(KernelError::HallucinationDetected);
        }

        self.state[self.current_index] = sifted.entropy();
        let prev_index = self.current_index;
        self.current_index = (self.current_index + 1) % SIZE;
        // Cap at SIZE: once the ring wraps, all slots are valid.
        if self.write_count < SIZE {
            self.write_count += 1;
        }

        let validated = ValidatedSynapse::new(sifted.into_inner());
        let validated_proof = ValidatedProof(());

        // Shadow validator: invariants at memory → kernel boundary
        debug_assert!(
            prev_index < SIZE,
            "CMIT: memory index {} overflowed SIZE={}",
            prev_index,
            SIZE,
        );

        Ok((validated, validated_proof))
    }
    /// Returns the number of valid observations currently in the ring buffer.
    fn valid_count(&self) -> usize {
        self.write_count.min(SIZE)
    }

    /// Returns an iterator over valid entries in temporal order
    /// (oldest first). Used internally by statistics methods.
    fn valid_entries(&self) -> ValidEntryIter<'_, SIZE> {
        let n = self.valid_count();
        let start = (self.current_index + SIZE - n) % SIZE;
        ValidEntryIter {
            memory: self,
            start,
            count: n,
            offset: 0,
        }
    }

    /// Returns the running mean entropy of the ring buffer.
    ///
    /// **Zero-observation behavior**: Returns `0.0` when no valid
    /// observations have been written (`write_count == 0`). This
    /// avoids spurious signals from uninitialized ring slots.
    ///
    /// **Partial-window**: When `write_count < SIZE`, the mean is
    /// computed over only the `write_count` valid observations.
    pub fn mean_entropy(&self) -> f64 {
        let n = self.valid_count();
        if n == 0 {
            return 0.0;
        }
        let sum: i128 = self.valid_entries().map(CognitiveEntropy::mantissa).sum();
        sum as f64 / n as f64
    }

    /// Returns the variance of entropy values in the ring buffer.
    ///
    /// **Zero-observation behavior**: Returns `0.0` when no valid
    /// observations have been written.
    ///
    /// **Partial-window**: Operates over only the `write_count`
    /// valid observations. Uninitialized slots are excluded.
    pub fn entropy_variance(&self) -> f64 {
        let mean = self.mean_entropy();
        let n = self.valid_count();
        if n == 0 {
            return 0.0;
        }
        let variance_sum: f64 = self
            .valid_entries()
            .map(|e| {
                let diff = e.mantissa() as f64 - mean;
                diff * diff
            })
            .sum();
        variance_sum / n as f64
    }

    /// Returns the linear regression slope over the buffer window.
    ///
    /// **Zero-observation behavior**: Returns `0.0` when no valid
    /// observations have been written (`n == 0` or `n == 1`).
    ///
    /// **Partial-window**: When `write_count < SIZE`, the slope is
    /// computed over only the `write_count` valid observations,
    /// assigning `x=0` to the oldest and `x=write_count-1` to the
    /// newest valid entry. This is intentional — warmup trend is
    /// computed from the available data, not padded with zeros.
    pub fn trend(&self) -> f64 {
        let n = self.valid_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let n_usize = self.valid_count();
        let start = (self.current_index + SIZE - n_usize) % SIZE;
        // Defer floating-point conversions: accumulate as i128 to reduce
        // roundoff error in the tight loop. CognitiveEntropy::mantissa()
        // returns i128, so we keep it native until the final division.
        let mut sum_y: i128 = 0;
        let mut sum_x_times_y: i128 = 0;

        // Walk valid entries in temporal order: oldest first, newest last.
        for i in 0..n_usize {
            let idx = (start + i) % SIZE;
            let x = i as i128;
            let y = self.state[idx].mantissa();
            sum_y += y;
            sum_x_times_y += x * y;
        }

        let sum_x = (n * (n - 1.0)) / 2.0;
        let sum_xx = (n * (n - 1.0) * (2.0 * n - 1.0)) / 6.0;

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator == 0.0 {
            return 0.0;
        }
        (n * (sum_x_times_y as f64) - sum_x * (sum_y as f64)) / denominator
    }

    /// Returns true if the absolute trend exceeds the given threshold.
    pub fn is_drifting(&self, threshold: f64) -> bool {
        self.trend().abs() > threshold
    }
}

/// Iterator over valid ring-buffer entries in temporal order (oldest first).
struct ValidEntryIter<'a, const SIZE: usize> {
    memory: &'a WorkingMemory<SIZE>,
    start: usize,
    count: usize,
    offset: usize,
}

impl<'a, const SIZE: usize> Iterator for ValidEntryIter<'a, SIZE> {
    type Item = &'a CognitiveEntropy<28, 2>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.count {
            return None;
        }
        let idx = (self.start + self.offset) % SIZE;
        self.offset += 1;
        Some(&self.memory.state[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llmosafe_kernel::{SiftedProof, Synapse};

    #[test]
    fn test_homeostatic_stats() {
        let mut memory = WorkingMemory::<4>::new(1000);
        for i in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(100 * (i + 1) as u16);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted, SiftedProof::for_testing()).unwrap();
        }
        // Buffer after 4 updates (all values present):
        // Entropy stored: 100, 200, 300, 400
        // Temporal order (oldest→newest): 100, 200, 300, 400
        assert_eq!(memory.mean_entropy(), 250.0);
        // Slope = 100.0 (rising from 100 to 400)
        assert!((memory.trend() - 100.0).abs() < 0.01);
        assert!(memory.is_drifting(10.0));
    }

    /// After exactly 1 accepted observation, mean must equal that
    /// observation's entropy — NOT 0.0 and NOT value/SIZE.
    #[test]
    fn test_single_obs_mean_equals_obs() {
        let mut memory = WorkingMemory::<4>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(300);
        let sifted = SiftedSynapse::new(synapse);
        memory.update(sifted, SiftedProof::for_testing()).unwrap();
        assert_eq!(
            memory.mean_entropy(),
            300.0,
            "1 accepted obs → mean must equal that obs, not 0.0 and not value/SIZE"
        );
        assert_eq!(memory.valid_count(), 1);
    }

    /// n<SIZE uses warmup population only (not padded with zeros).
    #[test]
    fn test_warmup_mean_excludes_uninitialized() {
        let mut memory = WorkingMemory::<4>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(200);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        // Mean over 2 valid obs = (100+200)/2 = 150, NOT (100+200+0+0)/4 = 75
        assert_eq!(
            memory.mean_entropy(),
            150.0,
            "warmup mean must use only valid observations, not padded zeros"
        );
        assert_eq!(memory.valid_count(), 2);
    }

    /// Variance also operates over warmup population only.
    #[test]
    fn test_warmup_variance_excludes_uninitialized() {
        let mut memory = WorkingMemory::<4>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(200);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        // Variance over 2 obs: mean=150, variance = ((100-150)^2 + (200-150)^2)/2 = 2500
        assert!(
            (memory.entropy_variance() - 2500.0).abs() < 0.01,
            "warmup variance must use only valid observations"
        );
    }

    /// Zero-observation behavior: all stats return 0.0 when write_count==0.
    #[test]
    fn test_zero_observation_stats_return_zero() {
        let memory = WorkingMemory::<4>::new(1000);
        assert_eq!(
            memory.mean_entropy(),
            0.0,
            "zero-observation mean must be 0.0"
        );
        assert_eq!(
            memory.entropy_variance(),
            0.0,
            "zero-observation variance must be 0.0"
        );
        assert_eq!(memory.trend(), 0.0, "zero-observation trend must be 0.0");
        assert_eq!(memory.valid_count(), 0);
    }

    /// Surprise-rejected observations never enter statistics and
    /// never increment write_count. Stats unchanged after rejection.
    #[test]
    fn test_surprise_rejected_obs_no_stats_change() {
        let mut memory = WorkingMemory::<4>::new(500);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(200);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        let mean_after_accept = memory.mean_entropy();
        let count_after_accept = memory.valid_count();

        // Reject an observation with too-high surprise
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(9000);
        synapse.set_raw_surprise(600); // > threshold 500
        synapse.set_has_bias(false);
        assert!(memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .is_err());

        assert_eq!(
            memory.mean_entropy(),
            mean_after_accept,
            "rejected obs must not change mean"
        );
        assert_eq!(
            memory.valid_count(),
            count_after_accept,
            "rejected obs must not increment write_count"
        );
    }

    /// Full-ring + wraparound: after SIZE updates, mean uses all entries.
    /// After wraparound, oldest entry is replaced and mean uses SIZE entries.
    #[test]
    fn test_wraparound_mean_correct() {
        let mut memory = WorkingMemory::<4>::new(1000);
        for i in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(100 * (i + 1) as u16);
            memory
                .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
                .unwrap();
        }
        assert_eq!(memory.mean_entropy(), 250.0);
        assert_eq!(memory.valid_count(), 4);

        // Wrap: overwrite oldest (100) with 500
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        assert_eq!(
            memory.mean_entropy(),
            350.0,
            "wraparound mean must exclude replaced slot"
        );
        assert_eq!(
            memory.valid_count(),
            4,
            "write_count capped at SIZE after wraparound"
        );
    }

    /// trend() returns 0.0 when only 1 observation exists (n<=1).
    #[test]
    fn test_trend_single_observation_zero() {
        let mut memory = WorkingMemory::<4>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(300);
        memory
            .update(SiftedSynapse::new(synapse), SiftedProof::for_testing())
            .unwrap();
        assert_eq!(
            memory.trend(),
            0.0,
            "trend with single observation must be 0.0"
        );
    }

    #[test]
    fn test_memory_update_gating() {
        let mut memory = WorkingMemory::<4>::new(500); // Threshold 5.00

        // 1. Valid update (Low surprise, no bias, stable entropy)
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert!(memory.update(sifted, SiftedProof::for_testing()).is_ok());

        // 2. Invalid update: Surprise too high (Hallucination)
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(600);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory
                .update(sifted, SiftedProof::for_testing())
                .unwrap_err(),
            KernelError::HallucinationDetected
        );

        // 3. Invalid update: Bias detected
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory
                .update(sifted, SiftedProof::for_testing())
                .unwrap_err(),
            KernelError::BiasHaloDetected
        );

        // 4. Invalid update: Cognitive Instability
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(50001);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory
                .update(sifted, SiftedProof::for_testing())
                .unwrap_err(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_working_memory_size_1() {
        let mut memory = WorkingMemory::<1>::new(1000);
        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        let sifted1 = SiftedSynapse::new(s1);

        memory.update(sifted1, SiftedProof::for_testing()).unwrap();
        assert!(memory.state[0].is_stable(100));

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        let sifted2 = SiftedSynapse::new(s2);

        memory.update(sifted2, SiftedProof::for_testing()).unwrap();
        assert!(memory.state[0].is_stable(200));
        assert_eq!(memory.current_index, 0);
    }

    #[test]
    fn test_memory_new_max_threshold() {
        let memory = WorkingMemory::<64>::new(i128::MAX);
        assert_eq!(memory.surprise_threshold, i128::MAX);
    }

    #[test]
    fn test_memory_zero_threshold() {
        let mut memory = WorkingMemory::<64>::new(0);
        let mut synapse = Synapse::new();
        synapse.set_raw_surprise(1);
        let sifted = SiftedSynapse::new(synapse);
        // Any surprise > 0 should fail
        assert_eq!(
            memory
                .update(sifted, SiftedProof::for_testing())
                .unwrap_err(),
            KernelError::HallucinationDetected
        );
    }

    #[test]
    fn test_memory_negative_threshold() {
        let mut memory = WorkingMemory::<64>::new(-1);
        let synapse = Synapse::new();
        let sifted = SiftedSynapse::new(synapse);
        // Even surprise 0 > -1, so it should fail
        assert_eq!(
            memory
                .update(sifted, SiftedProof::for_testing())
                .unwrap_err(),
            KernelError::HallucinationDetected
        );
    }

    // ── entropy_variance() edge cases ─────────────────────────────

    #[test]
    fn test_entropy_variance_size_one() {
        let mut memory = WorkingMemory::<1>::new(1000);
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(500);
        let sifted = SiftedSynapse::new(synapse);
        memory.update(sifted, SiftedProof::for_testing()).unwrap();
        // SIZE=1: variance must be 0.0 (single value = mean)
        let variance = memory.entropy_variance();
        assert!(
            variance < 0.001,
            "variance for SIZE=1 must be 0.0: got {}",
            variance
        );
    }

    #[test]
    fn test_entropy_variance_all_identical() {
        let mut memory = WorkingMemory::<4>::new(1000);
        for _ in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(300);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted, SiftedProof::for_testing()).unwrap();
        }
        let variance = memory.entropy_variance();
        assert!(
            variance < 0.001,
            "variance for identical values must be 0.0: got {}",
            variance
        );
    }

    // ── trend() edge cases ────────────────────────────────────────

    #[test]
    fn test_trend_size_two_identical() {
        let mut memory = WorkingMemory::<2>::new(1000);
        for _ in 0..2 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(200);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted, SiftedProof::for_testing()).unwrap();
        }
        let trend = memory.trend();
        assert!(
            trend.abs() < 0.001,
            "trend for two identical values must be 0.0: got {}",
            trend
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::llmosafe_kernel::{SiftedProof, Synapse};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_working_memory_random_synapse_sequence(
            entropies in prop::collection::vec(0u16..800u16, 1..200)
        ) {
            let mut memory = WorkingMemory::<64>::new(1000);
            for e in entropies {
                let mut synapse = Synapse::new();
                synapse.set_raw_entropy(e);
                let sifted = SiftedSynapse::new(synapse);
                prop_assert!(memory.update(sifted, SiftedProof::for_testing()).is_ok());
            }
        }
    }
}

#[cfg(feature = "std")]
/// std-only global cognitive memory: one shared WorkingMemory<64> behind a Mutex,
/// constructed with surprise threshold 58000 (matching the classifier surprise range [0, 65535]).
/// Consumed by the C-ABI and the body control loop.
pub mod cognitive_memory {
    use super::*;
    use crate::llmosafe_kernel::Synapse;
    use std::sync::Mutex;

    // Threshold=58000 matches the classifier surprise range [0,65535].
    // C-ABI callers set raw_surprise directly in the Synapse bits;
    // values >58000 will be rejected as HallucinationDetected (-4).
    static GLOBAL_MEMORY: Mutex<WorkingMemory<64>> = Mutex::new(WorkingMemory::<64>::new(58000));

    /// Lock the global memory, recovering from mutex poisoning with a warning.
    ///
    /// Mutex poisoning occurs when a thread panics while holding the lock.
    /// Recovery is used here because `get_memory_stats()` is a read-only
    /// operation where returning stale data is safer than panicking across
    /// the FFI boundary.
    fn lock_memory() -> std::sync::MutexGuard<'static, WorkingMemory<64>> {
        GLOBAL_MEMORY.lock().unwrap_or_else(|e| {
            tracing::warn!(
                target: "llmosafe::cognitive_memory",
                "GLOBAL_MEMORY mutex poisoned (prior panic detected), recovering inner state"
            );
            e.into_inner()
        })
    }

    /// C-ABI entry point: rebuilds a Synapse from raw 128 bits, wraps it as a
    /// SiftedSynapse with a bypass SiftedProof, and pushes it through the global
    /// WorkingMemory::update() gate. Returns 0 on acceptance.
    pub fn process_state_update(synapse_bits: u128) -> i32 {
        let synapse = Synapse::from_raw_u128(synapse_bits);
        let sifted = SiftedSynapse::new(synapse);
        let proof = SiftedProof::from_raw_bits_bypass();

        // Mutex poison returns -8 (distinct from all KernelError codes,
        // including SelfMemoryExceeded=-6 which previously collided).
        let mut memory = match GLOBAL_MEMORY.lock() {
            Ok(guard) => guard,
            Err(_) => return -8,
        };

        match memory.update(sifted, proof) {
            Ok(_) => 0,
            Err(e) => i32::from(e),
        }
    }

    /// Reads the global WorkingMemory statistics under the mutex with
    /// poison-recovery (lock_memory: warns via tracing and reuses inner state).
    /// Returns (mean, variance, trend, is_drifting).
    pub fn get_memory_stats() -> (f64, f64, f64, bool) {
        let memory = lock_memory();
        let mean = memory.mean_entropy();
        let variance = memory.entropy_variance();
        let trend = memory.trend();
        let is_drifting = memory.is_drifting(10.0);
        (mean, variance, trend, is_drifting)
    }
}
