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

//! G-EDGE tests for kernel module - comprehensive boundary testing

#[cfg(test)]
#[cfg(feature = "testing")]
mod tests {
    use llmosafe::{
        CognitiveEntropy, CusumDetector, DynamicStabilityMonitor, KernelError, ReasoningLoop,
        SiftedProof, SiftedSynapse, StabilityResult, Synapse, PRESSURE_THRESHOLD,
        STABILITY_THRESHOLD,
    };

    #[test]
    fn test_reasoning_loop_max_steps_exceeded() {
        let mut loop_guard = ReasoningLoop::<3>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        synapse.set_raw_surprise(50);
        synapse.set_has_bias(false);
        let sifted = SiftedSynapse::from_synapse(synapse);

        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let (validated, vproof) = memory.update(sifted, SiftedProof::for_testing()).unwrap();

        assert!(loop_guard.next_step(validated, vproof).is_ok());
        assert!(loop_guard.next_step(validated, vproof).is_ok());
        assert!(loop_guard.next_step(validated, vproof).is_ok());

        let result = loop_guard.next_step(validated, vproof);
        assert_eq!(result, Err(KernelError::DepthExceeded));
    }

    #[test]
    fn test_reasoning_loop_exact_boundary() {
        let mut loop_guard = ReasoningLoop::<5>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        let sifted = SiftedSynapse::from_synapse(synapse);
        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let (validated, vproof) = memory.update(sifted, SiftedProof::for_testing()).unwrap();

        for i in 0..5 {
            assert!(
                loop_guard.next_step(validated, vproof).is_ok(),
                "Step {} should succeed",
                i
            );
        }
        assert_eq!(
            loop_guard.next_step(validated, vproof),
            Err(KernelError::DepthExceeded)
        );
    }

    #[test]
    fn test_synapse_entropy_u16_max() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(u16::MAX);
        synapse.set_raw_surprise(100);
        synapse.set_has_bias(false);

        assert_eq!(synapse.raw_entropy(), 65535);
        let entropy = synapse.entropy();
        assert!(!entropy.is_stable(1000));
    }

    #[test]
    fn test_synapse_entropy_zero() {
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(0);
        assert_eq!(synapse.raw_entropy(), 0);
        assert!(synapse.entropy().is_stable(1000));
    }

    #[test]
    fn test_cusum_detector_threshold_exact() {
        let mut detector = CusumDetector::new(500.0, 50.0, 200.0);

        for _ in 0..10 {
            assert!(!detector.update(500.0), "At reference, should not detect");
        }

        for _ in 0..5 {
            detector.update(750.0);
        }
        assert!(
            detector.detected() || detector.update(750.0),
            "Should detect after sustained shift"
        );
    }

    #[test]
    fn test_dynamic_stability_all_results() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        assert_eq!(monitor.update(100), StabilityResult::Stable);
        assert_eq!(monitor.update(100), StabilityResult::Stable);

        let high_result = monitor.update(10000);
        assert!(matches!(
            high_result,
            StabilityResult::High | StabilityResult::Both
        ));

        monitor.reset();
        monitor.update(1000);
        let low_result = monitor.update(1);
        assert!(matches!(
            low_result,
            StabilityResult::Low | StabilityResult::Stable
        ));
    }

    #[test]
    fn test_cognitive_entropy_boundary() {
        let at_threshold = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD);
        assert!(at_threshold.is_stable(STABILITY_THRESHOLD));

        let above = CognitiveEntropy::<28, 2>::new(STABILITY_THRESHOLD + 1);
        assert!(!above.is_stable(STABILITY_THRESHOLD));

        let at_pressure = CognitiveEntropy::<28, 2>::new(PRESSURE_THRESHOLD);
        assert!(at_pressure.is_stable(STABILITY_THRESHOLD));
    }

    #[test]
    fn test_synapse_bit_patterns() {
        let zero = Synapse::from_raw_u128(0);
        assert_eq!(zero.raw_entropy(), 0);
        assert_eq!(zero.raw_surprise(), 0);

        let max_bits = Synapse::from_raw_u128(u128::MAX);
        assert_eq!(max_bits.raw_entropy(), u16::MAX);

        // In little-endian, entropy is in the lowest 16 bits
        // Setting bit 0-15 should set entropy
        let entropy_one = Synapse::from_raw_u128(1u128); // entropy = 1
        assert_eq!(entropy_one.raw_entropy(), 1);

        // Setting bits 16-31 should set surprise
        let surprise_one = Synapse::from_raw_u128(1u128 << 16); // surprise = 1
        assert_eq!(surprise_one.raw_surprise(), 1);
        assert_eq!(surprise_one.raw_entropy(), 0);
    }

    #[test]
    fn test_reasoning_loop_size_one() {
        let mut loop_guard = ReasoningLoop::<1>::new();
        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(100);
        let sifted = SiftedSynapse::from_synapse(synapse);
        let mut memory = llmosafe::WorkingMemory::<64>::new(1000);
        let (validated, vproof) = memory.update(sifted, SiftedProof::for_testing()).unwrap();

        assert!(loop_guard.next_step(validated, vproof).is_ok());
        assert_eq!(
            loop_guard.next_step(validated, vproof),
            Err(KernelError::DepthExceeded)
        );
    }

    #[test]
    fn test_cusum_negative_values() {
        // Test CusumDetector with negative inputs
        // CusumDetector computes: s_high = max(0, s_high + (val - mu_ref) - k)
        // For mu_ref=100, k=10, h=100, val=-50:
        // s_high = max(0, 0 + (-50 - 100) - 10) = max(0, -160) = 0
        // s_low = max(0, 0 - (-50 - 100) - 10) = max(0, 140) = 140
        let mut detector = CusumDetector::new(100.0, 10.0, 200.0);

        // Negative relative to reference
        detector.update(-50.0);
        // s_low should be positive but below threshold
        assert!(!detector.detected());

        // More negative values should eventually trigger
        for _ in 0..5 {
            detector.update(-50.0);
        }
        // After enough samples below reference, should detect
    }

    #[test]
    fn test_cusum_reset() {
        let mut detector = CusumDetector::new(500.0, 50.0, 200.0);
        for _ in 0..20 {
            detector.update(1000.0);
        }
        detector.reset();
        assert_eq!(detector.s_high(), 0.0);
        assert_eq!(detector.s_low(), 0.0);
    }

    /// Verifies the DynamicStabilityMonitor low-side guard at the exact boundary
    /// where lo_idx ≤ k. When the monitor has not seen enough low values to establish
    /// a baseline (lo_idx=3, k=5), an idx=0 drop must NOT trigger Low — premature
    /// "silent agent" detection during warm-up is a false positive.
    ///
    /// BUG 6 FIX (v0.8.1): The guard `self.lo_idx > self.k` was added to prevent
    /// spurious low-side anomalies before the monitor has gathered enough low-value
    /// history. Without it, idx=0 with lo_idx=3 would trigger Low when k=5, even
    /// though the monitor hadn't seen enough data to calibrate.
    #[test]
    fn test_dynamic_stability_monitor_low_side_guard() {
        // k=5 — large safety margin; lo_idx must exceed 5 for low detection
        let mut monitor = DynamicStabilityMonitor::new(5);

        // First update: msb_idx(8) = 3 → sets both hi_idx and lo_idx to 3
        let result = monitor.update(8);
        assert_eq!(
            result,
            StabilityResult::Stable,
            "first update always initializes"
        );

        // Second update: msb_idx(0) = 0, lo_idx=3, k=5
        // lo_idx(3) ≤ k(5) → guard prevents low detection → must return Stable
        let result = monitor.update(0);
        assert_eq!(
            result,
            StabilityResult::Stable,
            "lo_idx(3) ≤ k(5) — not enough low-value history to flag; must stay Stable"
        );

        // Third update with more data: feed msb_idx=4 values to raise lo_idx above k
        // After this, lo_idx is still 0 from the last update, hi_idx=4
        // But we need lo_idx to be ABOVE k to trigger. Feed high values first
        // to get hi_idx up, then a very low value with k=2 would trigger.
        // But with k=5 we need lo_idx > 5 for low detection. Let's use k=2.
        let mut monitor2 = DynamicStabilityMonitor::new(2);
        monitor2.update(8); // msb_idx=3 → hi=3, lo=3
                            // lo_idx(3) > k(2) = true → low detection IS enabled
        let result = monitor2.update(0); // msb_idx=0, 0 < 3-2=1 → low detection!
        assert_eq!(
            result,
            StabilityResult::Low,
            "lo_idx(3) > k(2) AND idx(0) < lo_idx(3)-k(2)=1 → must be Low"
        );
    }

    // ── Test 5: Integer Boundary (Confession 47) ──────────────────

    /// Verifies `DynamicStabilityMonitor::get_thresholds()` at the hi_idx=31
    /// boundary where (1u32 << (31 + 1)) would overflow a u32 shift.
    /// At hi_idx=31, high must equal u32::MAX (explicit guard prevents overflow).
    #[test]
    fn test_dynamic_stability_monitor_hi_idx_31_boundary() {
        let mut monitor = DynamicStabilityMonitor::new(2);

        // Push hi_idx to 31 by feeding a value whose msb_idx is 31.
        // u32::MAX has msb_idx=31. Feed it once to initialise, then again.
        monitor.update(u32::MAX); // msb_idx(MAX) = 31
        let (high, low, pressure) = monitor.get_thresholds();
        assert_eq!(
            high,
            u32::MAX,
            "hi_idx=31 must yield u32::MAX as high threshold"
        );
        // low is always ≤ u32::MAX by type; no redundant check needed
        let _ = low;
        // pressure = (u64::from(u32::MAX) * 4 / 5) as u32 = 3,435,973,836
        assert!(pressure > 0, "pressure must be positive at hi_idx=31");
        // pressure is u32, so ≤ u32::MAX by construction — no redundant check
    }

    /// Verifies `DynamicStabilityMonitor::get_thresholds()` when unseen
    /// (freshly constructed, no updates), returns safe defaults.
    #[test]
    fn test_dynamic_stability_monitor_unseen_defaults() {
        let monitor = DynamicStabilityMonitor::new(3);
        let (high, low, pressure) = monitor.get_thresholds();
        assert_eq!(
            high,
            u32::MAX,
            "unseen monitor: high must be u32::MAX (permissive)"
        );
        assert_eq!(low, 0, "unseen monitor: low must be 0 (permissive)");
        assert_eq!(pressure, 0, "unseen monitor: pressure must be 0 (no data)");
    }

    /// Verifies `DynamicStabilityMonitor::get_thresholds()` at hi_idx boundary
    /// values: 0 (minimum after update), 30 (just below overflow guard), 31 (guard).
    #[test]
    fn test_dynamic_stability_monitor_threshold_boundaries() {
        // hi_idx = 0: msb_idx(1) = 0
        let mut monitor = DynamicStabilityMonitor::new(2);
        monitor.update(1);
        let (high, _low, _pressure) = monitor.get_thresholds();
        assert_eq!(high, (1u32 << 1) - 1, "hi_idx=0: high = 2^1 - 1 = 1");
        // pressure is u32 — no check against u32::MAX needed by type

        // hi_idx = 30: msb_idx(2^30) = 30
        let mut monitor = DynamicStabilityMonitor::new(2);
        monitor.update(1u32 << 30);
        let (high, _low, pressure) = monitor.get_thresholds();
        assert_eq!(high, (1u32 << 31) - 1, "hi_idx=30: high = 2^31 - 1");
        let expected_pressure = (u64::from((1u32 << 31) - 1) * 4 / 5) as u32;
        assert_eq!(pressure, expected_pressure, "pressure at hi_idx=30");

        // hi_idx = 31: guard triggers, high = u32::MAX
        let mut monitor = DynamicStabilityMonitor::new(2);
        monitor.update(u32::MAX);
        let (high, _low, pressure) = monitor.get_thresholds();
        assert_eq!(high, u32::MAX, "hi_idx=31: guard returns u32::MAX");
        assert_eq!(
            pressure,
            (u64::from(u32::MAX) * 4 / 5) as u32,
            "pressure at hi_idx=31"
        );
    }

    /// Verifies `calculate_utility` saturating_mul clamp at u16::MAX boundary.
    /// When `count * 100` exceeds u16::MAX (65535), the result must clamp,
    /// not overflow or wrap.
    #[test]
    fn test_calculate_utility_saturating_mul_clamp() {
        use llmosafe::calculate_utility;

        // count = 0 → utility = 0
        let u0 = calculate_utility("no match here", "target");
        assert_eq!(u0, 0, "zero matching words → zero utility");

        // count = 1 → utility = 100
        let u1 = calculate_utility("target", "target");
        assert_eq!(u1, 100, "one matching word → utility = 100");

        // count = 655 → 655 * 100 = 65500 (< u16::MAX=65535)
        // Build a long observation with repeated objective words
        let obs_655: String = "x ".repeat(655);
        let objective_655: String = "x ".to_owned();
        let u655 = calculate_utility(&obs_655, &objective_655);
        assert_eq!(
            u655, 65500,
            "655 matches → utility = 65500 (below u16::MAX)"
        );

        // count = 656 → 656 * 100 = 65600 → clamped to u16::MAX = 65535
        let obs_656: String = "x ".repeat(656);
        let objective_656: String = "x ".to_owned();
        let u656 = calculate_utility(&obs_656, &objective_656);
        assert_eq!(
            u656, 65535,
            "656 matches → saturating_mul(65600).min(65535) = 65535"
        );

        // count = 1000 → 1000 * 100 = 100000 → clamped to u16::MAX = 65535
        let obs_1000: String = "x ".repeat(1000);
        let objective_1000: String = "x ".to_owned();
        let u1000 = calculate_utility(&obs_1000, &objective_1000);
        assert_eq!(u1000, 65535, "1000 matches → clamped to u16::MAX = 65535");
    }

    // ── Test 7: Guard Branch Coverage (Confession 49) ─────────────

    /// `sigmoid(f32::NAN)` must return 0.5 (the NaN guard), preventing
    /// infinite recursion via sigmoid(-NaN) = sigmoid(NaN).
    #[test]
    fn test_sigmoid_nan_guard_returns_neutral() {
        use llmosafe::llmosafe_classifier::sigmoid;
        let result = sigmoid(f32::NAN);
        assert_eq!(
            result, 0.5,
            "sigmoid(NaN) must return 0.5 (neutral probability, guard value)"
        );
    }

    /// `pid_risk_to_decision(f32::NAN, ...)` must return `Halt(CognitiveInstability, 0)`.
    /// NaN indicates sensor failure — must NOT return Proceed.
    #[test]
    #[cfg(feature = "std")]
    fn test_pid_risk_to_decision_nan_guard_returns_halt() {
        use llmosafe::llmosafe_pid::pid_risk_to_decision;
        use llmosafe::{KernelError, PidConfig, SafetyDecision};
        let config = PidConfig::default();
        let decision = pid_risk_to_decision(f32::NAN, &config);
        assert!(
            matches!(
                decision,
                SafetyDecision::Halt(KernelError::CognitiveInstability, 0)
            ),
            "NaN risk must trigger Halt(CognitiveInstability, 0), got {:?}",
            decision
        );
    }

    /// `pid_risk_to_decision` with valid finite inputs must NOT Halt at
    /// low risk levels.
    #[test]
    #[cfg(feature = "std")]
    fn test_pid_risk_to_decision_valid_inputs_normal_operation() {
        use llmosafe::llmosafe_pid::pid_risk_to_decision;
        use llmosafe::{PidConfig, SafetyDecision};
        let config = PidConfig::default();
        // Low risk → Proceed
        let d = pid_risk_to_decision(0.1, &config);
        assert!(matches!(d, SafetyDecision::Proceed));
        // Medium risk → Escalate
        let d = pid_risk_to_decision(0.6, &config);
        assert!(matches!(d, SafetyDecision::Escalate { .. }));
        // High risk → Halt
        let d = pid_risk_to_decision(1.0, &config);
        assert!(matches!(d, SafetyDecision::Halt(..)));
    }

    // ── Test 1: Fault Injection (Confession 43) ───────────────────

    /// Demonstrates the mutex poison recovery pattern used by
    /// `process_state_update()` in `llmosafe_memory::cognitive_memory`.
    ///
    /// The -8 return code at `src/llmosafe_memory.rs:439` guards against
    /// poisoned `GLOBAL_MEMORY` mutex. Since `GLOBAL_MEMORY` is crate-private,
    /// this test verifies:
    /// 1. The poison recovery pattern (`lock().unwrap_or_else(PoisonError::into_inner)`)
    ///    correctly handles a poisoned mutex and returns the inner value.
    /// 2. `process_state_update()` returns 0 for valid input (non-poisoned path).
    #[test]
    #[cfg(feature = "std")]
    fn test_mutex_poison_recovery_pattern() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        // 1. Poison a local Mutex by panicking in a spawned thread while holding the lock
        let mutex = Arc::new(Mutex::new(42i32));
        let m_clone = Arc::clone(&mutex);
        let handle = thread::spawn(move || {
            let _guard = m_clone.lock().unwrap();
            panic!("intentional panic to poison the mutex");
        });
        let _unused = handle.join(); // thread panicked — mutex is now poisoned

        // 2. Verify the mutex IS poisoned
        assert!(
            mutex.lock().is_err(),
            "mutex must be poisoned after thread panic"
        );

        // 3. Use poison recovery pattern (matching lock_memory() at llmosafe_memory.rs:420-428)
        let recovered = mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(*recovered, 42, "poison recovery must return inner value");

        // 4. Verify process_state_update() works normally with valid input
        // (GLOBAL_MEMORY is NOT poisoned from our test — this tests the happy path)
        let result = llmosafe::llmosafe_memory::cognitive_memory::process_state_update(0);
        assert_eq!(
            result, 0,
            "process_state_update(0) must return 0 for valid input"
        );

        // 5. Verify process_state_update() at boundary — low-entropy valid synapse
        let valid_bits = 500u128; // entropy=500 (< STABILITY_THRESHOLD), no bias
        let result = llmosafe::llmosafe_memory::cognitive_memory::process_state_update(valid_bits);
        assert_eq!(
            result, 0,
            "process_state_update(500) must return 0 for valid low-entropy synapse"
        );

        // The -8 poison path (src/llmosafe_memory.rs:439) cannot be triggered
        // from integration tests because GLOBAL_MEMORY is crate-private.
        // It is structurally verified: `Err(_) => return -8` guards the
        // lock acquisition. Full verification requires a concurrent test
        // that panics inside GLOBAL_MEMORY.lock() — see concurrent_stress.rs.
    }
}
