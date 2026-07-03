## 2026-07-03 - Inline string matching
**Learning:** When processing streaming string tokens via `split_whitespace().take(N)`, performing operations like hashing inline within the same iterator pass avoids the overhead of secondary loops and allocations like `ArrayVec`.
**Action:** Avoid intermediate collections for small streamed string matches.
