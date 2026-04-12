## 2025-03-08 - [Sifter Benchmarks]
**Learning:** `sift_perceptions` in `src/llmosafe_sifter.rs` takes ~9us to run. Inside it calls `calculate_utility`, which uses nested loops to match words: O(N*M) time complexity.
**Action:** Optimize `calculate_utility`.
