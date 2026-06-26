## 2024-05-28 - Optimize string parsing loops
**Learning:** When processing streaming string tokens (e.g., via `split_whitespace().take(N)`), perform operations such as hashing and matching inline within the same iterator pass rather than allocating intermediate collections (like `ArrayVec`) to avoid the overhead of secondary loops and allocations.
**Action:** Evaluate if parsed tokens can be processed in-place without materializing a temporary collection.
