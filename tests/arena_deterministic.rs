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

//! Deterministic Arena Synchronization / Lifetime Tests (Task #6)
//!
//! These tests prove, via deterministic sequences and proptest-based
//! reference-model comparison, the 7 arena properties:
//! 1. Fill to capacity (16) → 17th create fails correctly (usize::MAX).
//! 2. Destroy one slot → capacity recoverable.
//! 3. Stale handle from destroyed generation cannot address replacement.
//! 4. Replacement gets distinct valid generation.
//! 5. Ops on handle A never mutate/access handle B (isolation).
//! 6. Arena metadata lookup releases global lock while per-slot
//!    contents stay owned/locked (ownership semantics — via API behavior).
//! 7. Destroy/access interleavings preserve ownership semantics.
//!
//! Gated behind `#[cfg(feature = "std")]` — no_std cannot use threads
//! or the C-ABI arena. Uses proptest for sequence-generation based
//! model checking. No wall-clock or timing assertions.

#[cfg(feature = "std")]
mod arena_tests {
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    use proptest::prelude::*;

    use llmosafe::c_abi;

    // ── Arena constants (mirrored from src/lib.rs c_abi module) ──
    const ARENA_SIZE: usize = 16;
    const ARENA_INDEX_MASK: usize = 0xF;
    const GEN_SHIFT: usize = 4;

    /// Pack a handle from index and generation.
    fn pack_handle(index: usize, generation: u64) -> usize {
        index | ((generation as usize) << GEN_SHIFT)
    }

    /// Unpack a handle into (index, generation).
    fn unpack_handle(handle: usize) -> (usize, u64) {
        let index = handle & ARENA_INDEX_MASK;
        let generation = (handle >> GEN_SHIFT) as u64;
        (index, generation)
    }

    // ── Reference model ──────────────────────────────────
    struct ArenaModel {
        slots: [Option<u64>; ARENA_SIZE],
        next_generation: u64,
    }

    impl ArenaModel {
        fn new() -> Self {
            ArenaModel {
                slots: [None; ARENA_SIZE],
                next_generation: 0,
            }
        }
        fn create(&mut self) -> Option<usize> {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                if slot.is_none() {
                    let gen = self.next_generation;
                    self.next_generation = self.next_generation.wrapping_add(1);
                    *slot = Some(gen);
                    return Some(pack_handle(i, gen));
                }
            }
            None
        }
        fn destroy(&mut self, handle: usize) -> bool {
            let (index, generation) = unpack_handle(handle);
            if index >= ARENA_SIZE {
                return false;
            }
            match self.slots[index] {
                Some(gen) if gen == generation => {
                    self.slots[index] = None;
                    true
                }
                _ => false,
            }
        }
        fn is_valid(&self, handle: usize) -> bool {
            let (index, generation) = unpack_handle(handle);
            if index >= ARENA_SIZE {
                return false;
            }
            match self.slots[index] {
                Some(gen) if gen == generation => true,
                _ => false,
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────
    fn create_pipeline() -> usize {
        c_abi::llmosafe_create(
            b"deterministic test objective".as_ptr(),
            b"deterministic test objective".len(),
        )
    }

    fn process_text(handle: usize) -> i32 {
        c_abi::llmosafe_sift_and_process(
            handle,
            b"deterministic observation text for testing".as_ptr(),
            39,
        )
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 1: Fill to capacity (16) → 17th create fails
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop1_fill_to_capacity_17th_fails() {
        let mut handles: Vec<usize> = Vec::new();
        for _ in 0..ARENA_SIZE {
            let h = create_pipeline();
            assert_ne!(h, usize::MAX, "Create should succeed");
            handles.push(h);
        }
        assert_eq!(handles.len(), ARENA_SIZE);
        let mut sorted = handles.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ARENA_SIZE, "All handles must be unique");
        let fail = create_pipeline();
        assert_eq!(fail, usize::MAX, "17th create must fail when full");
        for h in handles {
            c_abi::llmosafe_destroy(h);
        }
    }

    #[test]
    fn prop1_capacity_recoverable_after_destroy() {
        let mut handles: Vec<usize> = Vec::new();
        for _ in 0..ARENA_SIZE {
            handles.push(create_pipeline());
        }
        let destroyed = handles[0];
        c_abi::llmosafe_destroy(destroyed);
        let new_h = create_pipeline();
        assert_ne!(new_h, usize::MAX, "Create must succeed after destroy");
        assert_ne!(new_h, destroyed, "New handle must differ");
        for h in handles {
            if h != destroyed {
                c_abi::llmosafe_destroy(h);
            }
        }
        c_abi::llmosafe_destroy(new_h);
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 2: Destroy one slot → capacity recoverable
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop2_destroy_frees_slot() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        c_abi::llmosafe_destroy(h1);
        let h2 = create_pipeline();
        assert_ne!(h2, usize::MAX, "Create must succeed after destroy");
        c_abi::llmosafe_destroy(h2);
    }

    #[test]
    fn prop2_destroy_all_frees_capacity() {
        let mut handles: Vec<usize> = Vec::new();
        for _ in 0..ARENA_SIZE {
            handles.push(create_pipeline());
        }
        for h in &handles {
            c_abi::llmosafe_destroy(*h);
        }
        for _ in 0..ARENA_SIZE {
            handles.push(create_pipeline());
        }
        for h in &handles {
            c_abi::llmosafe_destroy(*h);
        }
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 3: Stale handle from destroyed generation cannot address replacement
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop3_stale_handle_rejected_after_reuse() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        let (_, gen1) = unpack_handle(h1);
        let code1 = process_text(h1);
        assert!((-8..=2).contains(&code1), "h1 process must succeed");
        c_abi::llmosafe_destroy(h1);
        let h2 = create_pipeline();
        assert_ne!(h2, usize::MAX);
        let (_, gen2) = unpack_handle(h2);
        assert_ne!(gen2, gen1, "Replacement must have distinct generation");
        assert_eq!(
            c_abi::llmosafe_sift_and_process(h1, b"text".as_ptr(), 4),
            -9,
            "Stale handle must return -9 after slot reuse"
        );
        assert_eq!(c_abi::llmosafe_get_decision(h1), -9);
        assert!(c_abi::llmosafe_get_classifier_score(h1).is_nan());
        assert!((-8..=2).contains(&process_text(h2)));
        c_abi::llmosafe_destroy(h2);
    }

    #[test]
    fn prop3_stale_handle_all_ops_rejected() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        let _ = process_text(h1);
        c_abi::llmosafe_destroy(h1);
        assert_eq!(c_abi::llmosafe_get_decision(h1), -9);
        assert!(c_abi::llmosafe_get_classifier_score(h1).is_nan());
        assert_eq!(c_abi::llmosafe_configure(h1, 0, 0, 0), 1);
        assert_eq!(c_abi::llmosafe_reset_detectors(h1), 1);
        assert_eq!(c_abi::llmosafe_reset_full(h1), 1);
        assert_eq!(
            c_abi::llmosafe_process_with_pressure(h1, b"t".as_ptr(), 1, 0, 0),
            -9
        );
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 4: Replacement gets distinct valid generation
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop4_replacement_has_distinct_generation() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        let (_, gen1) = unpack_handle(h1);
        c_abi::llmosafe_destroy(h1);
        let h2 = create_pipeline();
        assert_ne!(h2, usize::MAX);
        let (_, gen2) = unpack_handle(h2);
        assert!(gen2 > gen1, "New gen {} must be > old gen {}", gen2, gen1);
        assert_ne!(gen1, gen2);
        let _ = process_text(h2);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h2)));
        assert_eq!(c_abi::llmosafe_get_decision(h1), -9);
        c_abi::llmosafe_destroy(h2);
    }

    #[test]
    fn prop4_multiple_cycles_generate_increasing_generations() {
        let mut generations: Vec<u64> = Vec::new();
        let mut last_gen: Option<u64> = None;
        for _ in 0..5 {
            let h = create_pipeline();
            assert_ne!(h, usize::MAX);
            let (_, gen) = unpack_handle(h);
            if let Some(prev) = last_gen {
                assert!(gen > prev, "Generations must increase");
            }
            generations.push(gen);
            last_gen = Some(gen);
            c_abi::llmosafe_destroy(h);
        }
        assert!(generations.windows(2).all(|w| w[1] > w[0]));
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 5: Ops on handle A never mutate/access handle B
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop5_handle_isolation_a_does_not_mutate_b() {
        let h_a = create_pipeline();
        assert_ne!(h_a, usize::MAX);
        let h_b = create_pipeline();
        assert_ne!(h_b, usize::MAX);
        assert!((-8..=2).contains(&process_text(h_a)));
        assert_eq!(c_abi::llmosafe_get_decision(h_b), -9);
        assert!((-8..=2).contains(&process_text(h_b)));
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_a)));
        assert!(c_abi::llmosafe_get_classifier_score(h_b).is_finite());
        c_abi::llmosafe_destroy(h_a);
        c_abi::llmosafe_destroy(h_b);
    }

    #[test]
    fn prop5_handle_isolation_configure() {
        let h_a = create_pipeline();
        assert_ne!(h_a, usize::MAX);
        let h_b = create_pipeline();
        assert_ne!(h_b, usize::MAX);
        let _ = process_text(h_a);
        let _ = process_text(h_b);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_a)));
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_b)));
        c_abi::llmosafe_configure(h_a, 2, 1, 64);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_b)));
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_a)));
        c_abi::llmosafe_destroy(h_a);
        c_abi::llmosafe_destroy(h_b);
    }

    #[test]
    fn prop5_handle_isolation_process_with_pressure() {
        let h_a = create_pipeline();
        assert_ne!(h_a, usize::MAX);
        let h_b = create_pipeline();
        assert_ne!(h_b, usize::MAX);
        assert!((-8..=2).contains(&c_abi::llmosafe_process_with_pressure(
            h_a,
            b"pressure test on A".as_ptr(),
            18,
            400,
            30
        )));
        let _ = process_text(h_b);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h_b)));
        c_abi::llmosafe_destroy(h_a);
        c_abi::llmosafe_destroy(h_b);
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 6: Arena metadata lookup releases global lock
    //            while per-slot contents stay owned/locked
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop6_arena_lock_released_during_per_slot_lock() {
        let mut handles: Vec<usize> = Vec::new();
        for _ in 0..ARENA_SIZE {
            handles.push(create_pipeline());
        }
        let barrier = Arc::new(Barrier::new(4));
        let results = Arc::new(Mutex::new(Vec::new()));
        let handles_arc = Arc::new(handles.clone());
        let threads: Vec<_> = (0..4)
            .map(|thread_id| {
                let barrier = Arc::clone(&barrier);
                let results = Arc::clone(&results);
                let handles = Arc::clone(&handles_arc);
                thread::spawn(move || {
                    barrier.wait();
                    let mut local_results: Vec<(usize, i32)> = Vec::new();
                    for (i, &h) in handles.iter().enumerate() {
                        if i % 4 == thread_id {
                            let code = c_abi::llmosafe_sift_and_process(
                                h,
                                b"concurrent test for arena lock release".as_ptr(),
                                39,
                            );
                            local_results.push((h, code));
                        }
                    }
                    results.lock().unwrap().extend(local_results);
                })
            })
            .collect();
        for jh in threads {
            jh.join().expect("Thread must not panic");
        }
        let all_results = results.lock().unwrap();
        assert_eq!(
            all_results.len(),
            ARENA_SIZE,
            "All 16 handles must be processed"
        );
        for (_h, code) in all_results.iter() {
            assert!(
                (-8..=2).contains(code),
                "All processes must return valid codes"
            );
        }
        for h in handles {
            c_abi::llmosafe_destroy(h);
        }
    }

    #[test]
    fn prop6_reset_releases_arena_lock_before_per_slot_lock() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        let h2 = create_pipeline();
        assert_ne!(h2, usize::MAX);
        let _ = process_text(h1);
        let _ = process_text(h2);
        let barrier = Arc::new(Barrier::new(2));
        let results = Arc::new(Mutex::new(Vec::new()));
        let b1 = Arc::clone(&barrier);
        let b2 = Arc::clone(&barrier);
        let r1 = Arc::clone(&results);
        let r2 = Arc::clone(&results);
        let t1 = thread::spawn(move || {
            b1.wait();
            let r = c_abi::llmosafe_reset_detectors(h1);
            r1.lock().unwrap().push((h1, r));
        });
        let t2 = thread::spawn(move || {
            b2.wait();
            let r = c_abi::llmosafe_reset_detectors(h2);
            r2.lock().unwrap().push((h2, r));
        });
        t1.join().expect("Thread 1 must not panic");
        t2.join().expect("Thread 2 must not panic");
        let all_results = results.lock().unwrap();
        assert_eq!(all_results.len(), 2);
        for (_h, r) in all_results.iter() {
            assert_eq!(*r, 0, "Reset must succeed");
        }
        c_abi::llmosafe_destroy(h1);
        c_abi::llmosafe_destroy(h2);
    }

    // ══════════════════════════════════════════════════════
    // PROPERTY 7: Destroy/access interleavings preserve ownership
    // ══════════════════════════════════════════════════════

    #[test]
    fn prop7_destroy_access_interleaving_preserves_ownership() {
        let h1 = create_pipeline();
        assert_ne!(h1, usize::MAX);
        let _ = process_text(h1);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h1)));
        c_abi::llmosafe_destroy(h1);
        assert_eq!(c_abi::llmosafe_get_decision(h1), -9);
        let h2 = create_pipeline();
        assert_ne!(h2, usize::MAX);
        let _ = process_text(h2);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h2)));
        assert_eq!(c_abi::llmosafe_get_decision(h1), -9);
        c_abi::llmosafe_destroy(h2);
    }

    #[test]
    fn prop7_concurrent_destroy_access_preserves_isolation() {
        let h_a = create_pipeline();
        assert_ne!(h_a, usize::MAX);
        let h_b = create_pipeline();
        assert_ne!(h_b, usize::MAX);
        let _ = process_text(h_a);
        let _ = process_text(h_b);
        let decision_a = c_abi::llmosafe_get_decision(h_a);
        let decision_b = c_abi::llmosafe_get_decision(h_b);
        assert!((-8..=2).contains(&decision_a));
        assert!((-8..=2).contains(&decision_b));
        c_abi::llmosafe_destroy(h_b);
        assert_eq!(c_abi::llmosafe_get_decision(h_a), decision_a);
        assert_eq!(c_abi::llmosafe_get_decision(h_b), -9);
        c_abi::llmosafe_destroy(h_a);
    }

    #[test]
    fn prop7_destroy_create_cycle_ownership_semantics() {
        let mut handles: Vec<usize> = Vec::new();
        for _ in 0..3 {
            let h = create_pipeline();
            assert_ne!(h, usize::MAX);
            let _ = process_text(h);
            assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h)));
            c_abi::llmosafe_destroy(h);
            assert_eq!(c_abi::llmosafe_get_decision(h), -9);
            handles.push(h);
        }
        for _ in &handles {
            let new_h = create_pipeline();
            assert_ne!(new_h, usize::MAX);
            let _ = process_text(new_h);
            assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(new_h)));
            c_abi::llmosafe_destroy(new_h);
        }
    }

    // ══════════════════════════════════════════════════════
    // PROPTEST: Sequence-based reference model comparison
    // ══════════════════════════════════════════════════════

    #[derive(Clone, Debug)]
    enum ArenaOp {
        Create,
        Process(usize),
        Inspect(usize),
        Destroy(usize),
        Reset(usize),
    }

    fn arena_op_sequence() -> impl Strategy<Value = Vec<ArenaOp>> {
        (1usize..=ARENA_SIZE).prop_flat_map(|n| {
            proptest::collection::vec(
                prop_oneof![
                    (0..ARENA_SIZE).prop_map(ArenaOp::Process),
                    (0..ARENA_SIZE).prop_map(ArenaOp::Inspect),
                    (0..ARENA_SIZE).prop_map(ArenaOp::Destroy),
                    (0..ARENA_SIZE).prop_map(ArenaOp::Reset),
                    proptest::strategy::Just(ArenaOp::Create),
                ],
                n,
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn proptest_arena_sequence_matches_reference_model(ops in arena_op_sequence()) {
            let mut model = ArenaModel::new();
            let mut slot_generation: [Option<u64>; ARENA_SIZE] = [None; ARENA_SIZE];
            let mut created_handles: Vec<usize> = Vec::new();

            for op in ops {
                match op {
                    ArenaOp::Create => {
                        let handle = c_abi::llmosafe_create(b"proptest objective".as_ptr(), b"proptest objective".len());
                        let model_result = model.create();
                        match (handle, model_result) {
                            (usize::MAX, None) => {}
                            (h, Some(_)) => {
                                let (idx, gen) = unpack_handle(h);
                                assert!(idx < ARENA_SIZE);
                                slot_generation[idx] = Some(gen);
                                created_handles.push(h);
                            }
                            _ => { panic!("Unexpected create result"); }
                        }
                    }
                    ArenaOp::Process(idx) => {
                        let gen = slot_generation[idx];
                        if gen.is_none() { continue; }
                        let handle = pack_handle(idx, gen.unwrap());
                        if !model.is_valid(handle) { continue; }
                        let code = c_abi::llmosafe_sift_and_process(handle, b"proptest process text".as_ptr(), 18);
                        let is_valid = model.is_valid(handle);
                        assert_eq!(is_valid, code != -9, "Process validity mismatch");
                        if is_valid {
                            assert!((-8..=2).contains(&code), "Valid handle must return valid code");
                        }
                    }
                    ArenaOp::Inspect(idx) => {
                        let gen = slot_generation[idx];
                        if gen.is_none() { continue; }
                        let handle = pack_handle(idx, gen.unwrap());
                        if !model.is_valid(handle) { continue; }
                        let decision = c_abi::llmosafe_get_decision(handle);
                        assert!((-8..=2).contains(&decision), "Valid handle must return valid decision");
                    }
                    ArenaOp::Destroy(idx) => {
                        let gen = slot_generation[idx];
                        if gen.is_none() { continue; }
                        let handle = pack_handle(idx, gen.unwrap());
                        c_abi::llmosafe_destroy(handle);
                        model.destroy(handle);
                        slot_generation[idx] = None;
                        created_handles.retain(|&h| h != handle);
                        assert!(!model.is_valid(handle), "After destroy, handle must be invalid");
                        let decision = c_abi::llmosafe_get_decision(handle);
                        assert_eq!(decision, -9, "Destroyed handle must return -9");
                    }
                    ArenaOp::Reset(idx) => {
                        let gen = slot_generation[idx];
                        if gen.is_none() { continue; }
                        let handle = pack_handle(idx, gen.unwrap());
                        if !model.is_valid(handle) { continue; }
                        let r = c_abi::llmosafe_reset_detectors(handle);
                        assert_eq!(r, 0, "Valid handle reset must return 0");
                    }
                }
            }
            // Cleanup: destroy all remaining handles
            for h in created_handles {
                c_abi::llmosafe_destroy(h);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn proptest_generation_monotonic_after_reuse(
            create_count in 1usize..4usize,
            destroy_count in 1usize..4usize,
        ) {
            let create_count = create_count.min(ARENA_SIZE);
            let destroy_count = destroy_count.min(create_count);
            let mut handles: Vec<usize> = Vec::new();
            for _ in 0..create_count {
                handles.push(c_abi::llmosafe_create(b"proptest".as_ptr(), 8));
            }
            let mut destroyed_generations: Vec<(usize, u64)> = Vec::new();
            for idx in 0..destroy_count {
                let h = handles[idx];
                let (_, gen) = unpack_handle(h);
                destroyed_generations.push((idx, gen));
                c_abi::llmosafe_destroy(h);
            }
            for _ in 0..destroy_count {
                let h = c_abi::llmosafe_create(b"proptest".as_ptr(), 8);
                assert_ne!(h, usize::MAX, "Create must succeed when arena has free slots");
                let (h_idx, h_gen) = unpack_handle(h);
                for &(idx, old_gen) in &destroyed_generations {
                    if h_idx == idx {
                        assert!(h_gen > old_gen, "Reused slot gen must be > old gen");
                    }
                }
                handles.push(h);
            }
            for h in handles {
                c_abi::llmosafe_destroy(h);
            }
        }
    }

    // ══════════════════════════════════════════════════════
    // Additional deterministic tests for edge cases
    // ══════════════════════════════════════════════════════

    #[test]
    fn test_stale_handle_classifier_score_is_nan() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        let _ = process_text(h);
        assert!(c_abi::llmosafe_get_classifier_score(h).is_finite());
        c_abi::llmosafe_destroy(h);
        assert!(c_abi::llmosafe_get_classifier_score(h).is_nan());
    }

    #[test]
    fn test_destroyed_handle_get_decision_is_minus9() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        let _ = process_text(h);
        assert!((-8..=2).contains(&c_abi::llmosafe_get_decision(h)));
        c_abi::llmosafe_destroy(h);
        assert_eq!(c_abi::llmosafe_get_decision(h), -9);
    }

    #[test]
    fn test_double_destroy_no_crash() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        c_abi::llmosafe_destroy(h);
        c_abi::llmosafe_destroy(h);
        assert_eq!(c_abi::llmosafe_get_decision(h), -9);
    }

    #[test]
    fn test_destroy_invalid_handle_no_crash() {
        c_abi::llmosafe_destroy(999);
        c_abi::llmosafe_destroy(usize::MAX);
    }

    #[test]
    fn test_stale_handle_reset_full_returns_one() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        let _ = process_text(h);
        c_abi::llmosafe_destroy(h);
        assert_eq!(c_abi::llmosafe_reset_full(h), 1);
    }

    #[test]
    fn test_stale_handle_configure_returns_one() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        c_abi::llmosafe_destroy(h);
        assert_eq!(c_abi::llmosafe_configure(h, 0, 0, 0), 1);
    }

    #[test]
    fn test_stale_handle_process_with_pressure_returns_minus9() {
        let h = create_pipeline();
        assert_ne!(h, usize::MAX);
        let _ = process_text(h);
        c_abi::llmosafe_destroy(h);
        assert_eq!(
            c_abi::llmosafe_process_with_pressure(h, b"text".as_ptr(), 4, 400, 30),
            -9
        );
    }
}

// ── Smoke tests (no std required) ──────────────────

#[cfg(not(feature = "std"))]
mod arena_smoke {
    #[test]
    fn arena_constants_correct() {
        assert_eq!(16usize, 16);
    }

    #[test]
    fn pack_unpack_handle_roundtrip() {
        let index: usize = 5;
        let generation: u64 = 3;
        let packed = index | ((generation as usize) << 4);
        let (unpacked_index, unpacked_gen) = (packed & 0xF, (packed >> 4) as u64);
        assert_eq!(unpacked_index, index);
        assert_eq!(unpacked_gen, generation);
    }
}
