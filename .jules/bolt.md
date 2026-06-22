## 2023-10-25 - Avoid intermediate collections for streaming string tokens
**Learning:** When processing streaming string tokens (e.g., via `split_whitespace().take(N)`), allocating intermediate collections like `ArrayVec` before matching incurs overhead from secondary loops and allocations.
**Action:** Perform operations such as hashing and matching inline within the same iterator pass.
