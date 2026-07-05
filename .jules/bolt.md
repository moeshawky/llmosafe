## 2023-10-27 - Inline Hashing and Matching over Intermediate Collection
**Learning:** When processing streaming string tokens via iterators (e.g., `split_whitespace().take(N)`), buffering them into an intermediate collection like `ArrayVec` before performing a secondary matching loop adds significant performance overhead and allocation costs in tight loops.
**Action:** Perform required operations such as hashing and matching inline within the same iterator pass to avoid secondary loops and collection overheads.
