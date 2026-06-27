## 2024-06-27 - Inline processing of streaming string tokens
**Learning:** When processing streaming string tokens (e.g., via `split_whitespace().take(N)`), performing operations such as hashing and matching inline within the same iterator pass is more performant than allocating intermediate collections (like `ArrayVec`) and doing a secondary loop.
**Action:** Avoid allocating intermediate collections for streaming string processing; process tokens inline during the iterator pass to save allocations and secondary iteration overhead.
