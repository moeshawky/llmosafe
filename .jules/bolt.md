## 2025-06-21 - Inline Streaming String Tokens Optimization
**Learning:** When processing streaming string tokens (like `split_whitespace().take(N)`), allocating intermediate collections (such as `ArrayVec`) introduces unnecessary overhead from secondary loops and allocations.
**Action:** Perform operations like hashing and matching inline within the same iterator pass rather than collecting into intermediate buffers to improve performance.
