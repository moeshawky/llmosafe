## 2025-02-24 - Inline Iterator Matching
**Learning:** When processing streaming string tokens (e.g., via split_whitespace()), allocating intermediate collections like ArrayVec before matching adds measurable overhead from secondary loops and allocations.
**Action:** Perform operations such as hashing and matching inline within the same iterator pass.
