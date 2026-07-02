## 2024-07-02 - Optimize inline processing of string tokens
**Learning:** Processing streaming string tokens (like `split_whitespace().take(N)`) into intermediate collections (like `ArrayVec`) before matching causes unnecessary allocations and secondary loop overhead.
**Action:** Perform operations such as hashing and matching inline within the same iterator pass rather than allocating intermediate collections to avoid the overhead of secondary loops and allocations.
