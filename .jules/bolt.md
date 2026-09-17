## 2024-03-24 - [Initialization]
**Learning:** Created initial journal file for Bolt persona.
**Action:** Track performance-critical learnings going forward.

## 2026-09-17 - [Optimize DriftDetector Observation]
**Learning:** When searching small fixed-size arrays within hot loops, using the slice `.contains()` method is faster than manual iteration with `.iter().any()`, and avoiding intermediate heapless vector allocations (like `ArrayVec`) for temporary items reduces overhead.
**Action:** Optimize `DriftDetector::observe` to use a fixed-size array and `[..obs_len].contains(&goal_hash)`.
