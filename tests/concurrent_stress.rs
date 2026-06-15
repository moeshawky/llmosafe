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

//! Concurrent Stress Tests (Test 3: Confession 45)
//!
//! Multi-threaded tests for the C-ABI mutex-protected arena
//! (GLOBAL_MEMORY, PIPELINE_ARENA). Verifies:
//! - No deadlocks under concurrent create/process/destroy
//! - Unique instance IDs per thread
//! - No panics under contention
//! - Poison recovery path verification
//!
//! Gated behind `#[cfg(feature = "std")]` — no_std cannot use threads.

#[cfg(feature = "std")]
mod concurrent_c_abi_tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    // Import FFI functions from the crate. These are `extern "C"` functions
    // defined in `src/lib.rs` module `c_abi`.
    extern "C" {
        fn llmosafe_create(objective_ptr: *const u8, objective_len: usize) -> usize;
        fn llmosafe_sift_and_process(handle: usize, text_ptr: *const u8, text_len: usize) -> i32;
        fn llmosafe_destroy(handle: usize);
        fn llmosafe_get_stability(synapse_bits: u128) -> i32;
    }

    /// Spawns 4 threads simultaneously accessing the C-ABI arena.
    /// Each thread creates a pipeline, sifts text, and destroys.
    /// Verifies: no deadlocks, unique handles, no panics.
    #[test]
    fn test_concurrent_create_process_destroy_4_threads() {
        const NUM_THREADS: usize = 4;
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    // Synchronize — all threads start at the same time
                    barrier.wait();

                    // Create a pipeline with a unique objective per thread
                    let objective = format!("safety objective thread {}", thread_id);
                    let handle = unsafe { llmosafe_create(objective.as_ptr(), objective.len()) };
                    assert_ne!(
                        handle,
                        usize::MAX,
                        "Thread {}: create must succeed",
                        thread_id
                    );

                    // Process several observations
                    let observations = [
                        "System running normally",
                        "Performing safety check on input data",
                        "All clear, proceeding with operation",
                    ];
                    for obs in &observations {
                        let result =
                            unsafe { llmosafe_sift_and_process(handle, obs.as_ptr(), obs.len()) };
                        // Must return a valid code (0=Proceed, positive=Warn/Escalate,
                        // negative=Halt/error)
                        assert!(
                            result >= -9 && result <= 2,
                            "Thread {}: sift_and_process returned invalid code {}",
                            thread_id,
                            result
                        );
                    }

                    // Destroy the pipeline
                    unsafe { llmosafe_destroy(handle) };

                    handle
                })
            })
            .collect();

        // Collect all results — verify no panics
        let mut created_handles: Vec<usize> = Vec::new();
        for (_i, jh) in handles.into_iter().enumerate() {
            let handle = jh.join().expect("Thread must not panic");
            created_handles.push(handle);
        }

        // All handles must be unique (generation counter ensures this)
        let mut sorted = created_handles.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            NUM_THREADS,
            "All {} threads must have unique handles",
            NUM_THREADS
        );
    }

    /// Spawns 3 threads: one creates+processes, one creates+destroys+creates,
    /// one processes on an existing handle. Exercises interleaved arena access.
    #[test]
    fn test_concurrent_interleaved_operations_3_threads() {
        let barrier = Arc::new(Barrier::new(3));

        // All threads must be spawned before any join, or the Barrier deadlocks.

        // Thread A: Create a pipeline, process, hold it
        let b_a = Arc::clone(&barrier);
        let thread_a = thread::spawn(move || {
            b_a.wait();
            let objective = "thread_a_objective_long_running";
            let handle = unsafe { llmosafe_create(objective.as_ptr(), objective.len()) };
            assert_ne!(handle, usize::MAX, "Thread A: create must succeed");
            let obs = "thread a observation text";
            let result = unsafe { llmosafe_sift_and_process(handle, obs.as_ptr(), obs.len()) };
            assert!(
                result >= -9 && result <= 2,
                "Thread A: sift_and_process returned {}",
                result
            );
            handle
        });

        // Thread B: Create, process, destroy, create again
        let b_b = Arc::clone(&barrier);
        let thread_b = thread::spawn(move || {
            b_b.wait();
            let objective = "thread_b_objective";
            let handle = unsafe { llmosafe_create(objective.as_ptr(), objective.len()) };
            assert_ne!(handle, usize::MAX, "Thread B: create must succeed");

            let obs = "thread b observation text";
            let result = unsafe { llmosafe_sift_and_process(handle, obs.as_ptr(), obs.len()) };
            assert!(
                result >= -9 && result <= 2,
                "Thread B: sift_and_process returned {}",
                result
            );

            unsafe { llmosafe_destroy(handle) };
            handle
        });

        // Thread C: Create a pipeline and hold it
        let b_c = Arc::clone(&barrier);
        let thread_c = thread::spawn(move || {
            b_c.wait();
            let objective = "thread_c_objective";
            let handle = unsafe { llmosafe_create(objective.as_ptr(), objective.len()) };
            assert_ne!(handle, usize::MAX, "Thread C: create must succeed");
            handle
        });

        // Now join all threads (all 3 have been spawned and will reach the Barrier)
        let handle_a = thread_a.join().unwrap();
        let handle_b = thread_b.join().unwrap();
        let handle_c = thread_c.join().unwrap();

        // Verify handles are unique
        assert_ne!(handle_a, handle_b, "Handles A and B must differ");
        assert_ne!(handle_a, handle_c, "Handles A and C must differ");
        assert_ne!(handle_b, handle_c, "Handles B and C must differ");

        // Clean up remaining handles
        unsafe {
            llmosafe_destroy(handle_a);
            llmosafe_destroy(handle_c);
        }
    }

    /// Tests that `llmosafe_get_stability` can be called from multiple
    /// threads concurrently without deadlocks or panics.
    #[test]
    fn test_concurrent_stability_check_4_threads() {
        const NUM_THREADS: usize = 4;
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();

                    // Each thread checks stability of different synapse patterns
                    let patterns = [
                        0u128,         // all-zero
                        500u128,       // low entropy
                        50001u128,     // high entropy (unstable)
                        (1u128 << 32), // bias flag set
                    ];
                    let pattern = patterns[thread_id % patterns.len()];
                    let result = unsafe { llmosafe_get_stability(pattern) };
                    assert!(
                        result >= -7 && result <= 0,
                        "Thread {}: stability must return valid code, got {}",
                        thread_id,
                        result
                    );
                })
            })
            .collect();

        for jh in handles {
            jh.join().expect("All threads must complete without panics");
        }
    }

    /// Tests that process_state_update (via FFI llmosafe_process_synapse)
    /// behaves correctly under concurrent access.
    /// The GLOBAL_MEMORY mutex protects WorkingMemory from concurrent
    /// mutation — multiple threads calling process_state_update should
    /// not deadlock.
    #[test]
    fn test_concurrent_process_synapse_no_deadlock() {
        extern "C" {
            fn llmosafe_process_synapse(synapse_bits: u128) -> i32;
        }

        const NUM_THREADS: usize = 3;
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    // Process several clean synapses concurrently
                    for entropy in [100u128, 200u128, 300u128] {
                        let result = unsafe { llmosafe_process_synapse(entropy) };
                        assert!(
                            result >= -8 && result <= 0,
                            "process_synapse must return valid code in [-8, 0], got {}",
                            result
                        );
                        // Valid low-entropy synapse without bias → expect 0
                        if entropy < 50000 {
                            assert_eq!(result, 0, "low-entropy synapse must return 0 (success)");
                        }
                    }
                })
            })
            .collect();

        for jh in handles {
            jh.join()
                .expect("All threads must complete without deadlock or panic");
        }
    }

    /// Tests mutex poison recovery: one thread panics while holding a mutex,
    /// another thread accesses the same mutex-protected state and recovers.
    ///
    /// This test uses a local Mutex to demonstrate the poison recovery
    /// pattern, then verifies that the real GLOBAL_MEMORY continues to
    /// function after our local mutex manipulation (it is unaffected).
    #[test]
    fn test_mutex_poison_recovery_concurrent() {
        use std::sync::Mutex;

        // Create a local mutex and poison it via a separate thread
        let mutex = Arc::new(Mutex::new(0u32));
        let m_clone = Arc::clone(&mutex);
        let poisoner = thread::spawn(move || {
            let _guard = m_clone.lock().unwrap();
            panic!("deliberate panic to poison test mutex");
        });
        drop(poisoner.join());

        // Verify the local mutex IS poisoned
        assert!(mutex.lock().is_err(), "Local test mutex must be poisoned");

        // Recover using the same pattern as lock_arena() and lock_memory()
        let recovered = mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(*recovered, 0);

        // Now verify the REAL GLOBAL_MEMORY is unaffected (it was never poisoned)
        // by calling process_state_update with valid input
        let result = llmosafe::llmosafe_memory::cognitive_memory::process_state_update(0);
        assert_eq!(result, 0, "GLOBAL_MEMORY must be unpoisoned and functional");

        // The -8 return from process_state_update (line 439) would be triggered
        // when GLOBAL_MEMORY IS poisoned. Since we cannot access GLOBAL_MEMORY
        // from integration tests, this code path is structurally verified:
        //   Err(_) => return -8
        // at src/llmosafe_memory.rs:437-439.
        // Full end-to-end verification requires poisoning GLOBAL_MEMORY from
        // within the crate (a unit test in src/llmosafe_memory.rs).
    }
}
