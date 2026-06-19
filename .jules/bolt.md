## 2025-02-28 - Avoid Intermediate Array Collection When Hashing
**Learning:** In `DriftDetector::observe`, collecting hashed values into an `ArrayVec` before performing existence checks incurred a ~67% overhead (231ms vs 138ms in synthetic benchmarking of 1M ops). The `ArrayVec` allocation and the second pass over the elements were unnecessary.
**Action:** When streaming string tokens (e.g., via `split_whitespace().take(N)`), perform operations (like hashing and comparison) directly within the same iterator pass to avoid intermediate data structures and second loops.
