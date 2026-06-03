//! LLMOSAFE Tier 2 Cognitive Working Memory
//!
//! Implements surprise-gated state updates with fixed-size storage.
//! Uses a ring buffer of cognitive entropy values with momentum-based
//! drift detection.
//!
//! # Architecture
//!
//! Drawing from TransformerFAM and Infini-attention: surprise-based
//! gating prevents hallucination propagation, while fixed-size storage
//! ensures stack allocation and no heap fragmentation.

use crate::llmosafe_kernel::{
    CognitiveEntropy, KernelError, SiftedProof, SiftedSynapse, ValidatedProof, ValidatedSynapse,
};

/// Fixed-size working memory stack buffer.
///
/// Holds up to SIZE entropy values in a ring buffer.
/// Default capacity is 64 entries (stack-allocated).
pub struct WorkingMemory<const SIZE: usize = 64> {
    state: [CognitiveEntropy<28, 2>; SIZE],
    current_index: usize,
    surprise_threshold: i128,
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
        }
    }

    /// Updated: Uses the SiftedSynapse protocol for state transitions.
    ///
    /// # Examples
    ///
    /// ```
    /// use llmosafe::{WorkingMemory, sift_perceptions};
    /// let mut memory = WorkingMemory::<64>::new(1000);
    /// let (sifted, proof) = sift_perceptions(&["stable observation"], "test");
    /// let (validated, _proof) = memory.update(sifted, proof).unwrap();
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
    pub fn mean_entropy(&self) -> f64 {
        let sum: i128 = self.state.iter().map(|e| e.mantissa()).sum();
        sum as f64 / SIZE as f64
    }

    pub fn entropy_variance(&self) -> f64 {
        let mean = self.mean_entropy();
        let variance_sum: f64 = self
            .state
            .iter()
            .map(|e| {
                let diff = e.mantissa() as f64 - mean;
                diff * diff
            })
            .sum();
        variance_sum / SIZE as f64
    }

    pub fn trend(&self) -> f64 {
        let n = SIZE as f64;

        // ⚡ Bolt Optimization: Use i128 for accumulations inside the loop
        // deferring f64 conversions to the final reduction. This avoids
        // redundant float conversions and prevents floating-point roundoff error accumulation.
        let mut sum_y: i128 = 0;
        let mut sum_x_times_y: i128 = 0;

        // Walk the ring buffer in temporal order: oldest first, newest last.
        // After wraparound, buffer order is [current_index, ..., SIZE-1, 0, ..., current_index-1].
        // Assign x=0 to oldest, x=SIZE-1 to newest.
        for offset in 0..SIZE {
            let idx = (self.current_index + offset) % SIZE;
            let x = offset as i128;
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

    pub fn is_drifting(&self, threshold: f64) -> bool {
        self.trend().abs() > threshold
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
            memory.update(sifted, SiftedProof(())).unwrap();
        }
        // Buffer after 4 updates (all values present):
        // Entropy stored: 100, 200, 300, 400
        // Temporal order (oldest→newest): 100, 200, 300, 400
        assert_eq!(memory.mean_entropy(), 250.0);
        // Slope = 100.0 (rising from 100 to 400)
        assert!((memory.trend() - 100.0).abs() < 0.01);
        assert!(memory.is_drifting(10.0));
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
        assert!(memory.update(sifted, SiftedProof(())).is_ok());

        // 2. Invalid update: Surprise too high (Hallucination)
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(600);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted, SiftedProof(())).unwrap_err(),
            KernelError::HallucinationDetected
        );

        // 3. Invalid update: Bias detected
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted, SiftedProof(())).unwrap_err(),
            KernelError::BiasHaloDetected
        );

        // 4. Invalid update: Cognitive Instability
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1100);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted, SiftedProof(())).unwrap_err(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_working_memory_size_1() {
        let mut memory = WorkingMemory::<1>::new(1000);
        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        let sifted1 = SiftedSynapse::new(s1);

        memory.update(sifted1, SiftedProof(())).unwrap();
        assert!(memory.state[0].is_stable(100));

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        let sifted2 = SiftedSynapse::new(s2);

        memory.update(sifted2, SiftedProof(())).unwrap();
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
            memory.update(sifted, SiftedProof(())).unwrap_err(),
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
            memory.update(sifted, SiftedProof(())).unwrap_err(),
            KernelError::HallucinationDetected
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
                prop_assert!(memory.update(sifted, SiftedProof(())).is_ok());
            }
        }
    }
}

#[cfg(feature = "std")]
pub mod cognitive_memory {
    use super::*;
    use crate::llmosafe_kernel::Synapse;
    use std::sync::Mutex;

    static GLOBAL_MEMORY: Mutex<WorkingMemory<64>> = Mutex::new(WorkingMemory::<64>::new(500));

    pub fn process_state_update(synapse_bits: u128) -> i32 {
        let synapse = Synapse::from_raw_u128(synapse_bits);
        let sifted = SiftedSynapse::new(synapse);
        // C-ABI callers bypass the Rust-side sifter; the caller is responsible
        // for pre-sifting on their side. We mint the proof internally.
        let proof = SiftedProof(());

        let mut memory = GLOBAL_MEMORY.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        match memory.update(sifted, proof) {
            Ok(_) => 0,
            Err(KernelError::DepthExceeded) => -1,
            Err(KernelError::CognitiveInstability) => -2,
            Err(KernelError::BiasHaloDetected) => -3,
            Err(KernelError::HallucinationDetected) => -4,
            Err(KernelError::ResourceExhaustion) => -5,
            Err(KernelError::SelfMemoryExceeded) => -6,
            Err(KernelError::DeadlineExceeded) => -7,
        }
    }
}
