//! LLMOSAFE Tier 2 Cognitive Working Memory
//!
//! This module implements the "Memory Integrity" Axiom.
//! It uses the Seshat principle of "Tangible Ratios" (Sekel)
//! to maintain a fixed-size cognitive state without heap allocation.
//!
//! Research Grounds:
//! - Titans: Surprise-based gating with momentum.
//! - TransformerFAM: Feedback-loop working memory.
//! - Infini-attention: Compressive associative memory.

use crate::llmosafe_kernel::{
    CognitiveEntropy, KernelError, SiftedSynapse, Synapse, ValidatedSynapse,
};

/// The "Working Memory" container (The Sekel of State).
/// Ratio: 64 "Palms" (anchors) for persistent reasoning.
pub struct WorkingMemory<const SIZE: usize = 64> {
    state: [CognitiveEntropy<28, 2>; SIZE],
    current_index: usize,
    surprise_threshold: i128,
}

impl<const SIZE: usize> WorkingMemory<SIZE> {
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
    /// let sifted = sift_perceptions(&["stable observation"], "test");
    /// let validated = memory.update(sifted).unwrap();
    /// ```
    pub fn update(&mut self, sifted: SiftedSynapse) -> Result<ValidatedSynapse, KernelError> {
        sifted.validate()?;

        if sifted.surprise() > self.surprise_threshold {
            return Err(KernelError::HallucinationDetected);
        }

        self.state[self.current_index] = sifted.entropy();
        self.current_index = (self.current_index + 1) % SIZE;

        Ok(ValidatedSynapse::new(sifted.into_inner()))
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
        let (sum_x, sum_y, sum_xy, sum_xx) = self.state.iter().enumerate().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(sum_x, sum_y, sum_xy, sum_xx), (i, e)| {
                let x = i as f64;
                let y = e.mantissa() as f64;
                (sum_x + x, sum_y + y, sum_xy + x * y, sum_xx + x * x)
            },
        );

        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    }

    pub fn is_drifting(&self, threshold: f64) -> bool {
        self.trend().abs() > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llmosafe_kernel::Synapse;

    #[test]
    fn test_homeostatic_stats() {
        let mut memory = WorkingMemory::<4>::new(1000);
        for i in 0..4 {
            let mut synapse = Synapse::new();
            synapse.set_raw_entropy(100 * (i + 1) as u16);
            let sifted = SiftedSynapse::new(synapse);
            memory.update(sifted).unwrap();
        }
        // Entropy: 100, 200, 300, 400
        assert_eq!(memory.mean_entropy(), 250.0);
        assert!(memory.trend() > 0.0);
        assert!(memory.is_drifting(50.0));
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
        assert!(memory.update(sifted).is_ok());

        // 2. Invalid update: Surprise too high (Hallucination)
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(600);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted).unwrap_err(),
            KernelError::HallucinationDetected
        );

        // 3. Invalid update: Bias detected
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(400);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(true);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted).unwrap_err(),
            KernelError::BiasHaloDetected
        );

        // 4. Invalid update: Cognitive Instability
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(1100);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::new(synapse);
        assert_eq!(
            memory.update(sifted).unwrap_err(),
            KernelError::CognitiveInstability
        );
    }

    #[test]
    fn test_working_memory_size_1() {
        let mut memory = WorkingMemory::<1>::new(1000);
        let mut s1 = Synapse::new();
        s1.set_raw_entropy(100);
        let sifted1 = SiftedSynapse::new(s1);

        memory.update(sifted1).unwrap();
        assert!(memory.state[0].is_stable(100));

        let mut s2 = Synapse::new();
        s2.set_raw_entropy(200);
        let sifted2 = SiftedSynapse::new(s2);

        memory.update(sifted2).unwrap();
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
            memory.update(sifted).unwrap_err(),
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
            memory.update(sifted).unwrap_err(),
            KernelError::HallucinationDetected
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::llmosafe_kernel::Synapse;
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
                prop_assert!(memory.update(sifted).is_ok());
            }
        }
    }
}

#[cfg(feature = "std")]
pub mod cognitive_memory {
    use super::*;
    use std::sync::Mutex;

    static GLOBAL_MEMORY: Mutex<WorkingMemory<64>> = Mutex::new(WorkingMemory::<64>::new(500));

    pub fn process_state_update(synapse_bits: u128) -> i32 {
        let synapse = Synapse::from_raw_u128(synapse_bits);
        let sifted = SiftedSynapse::new(synapse);

        let mut memory = GLOBAL_MEMORY.lock().unwrap();

        match memory.update(sifted) {
            Ok(_) => 0,
            Err(KernelError::DepthExceeded) => -1,
            Err(KernelError::CognitiveInstability) => -2,
            Err(KernelError::BiasHaloDetected) => -3,
            Err(KernelError::HallucinationDetected) => -4,
            Err(KernelError::ResourceExhaustion) => -5,
        }
    }
}
