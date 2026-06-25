## 2024-06-25 - Avoid Intermediate Allocations in Streaming Tokens
**Learning:** When processing streaming string tokens via `split_whitespace().take(N)`, allocating intermediate collections like `ArrayVec` introduces overhead from secondary loops and allocations.
**Action:** Perform operations such as hashing and matching inline within the same iterator pass.
