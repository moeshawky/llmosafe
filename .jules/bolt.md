## 2026-09-03 - [Slice contains over iter().any()]
**Learning:** Using slice `.contains()` is much faster (using direct CPU instructions and avoiding iterator overhead) than using `.iter().any()` for searching within arrays/vectors of primitives like `u32`.
**Action:** Always prefer using `.contains()` over `.iter().any(|&val| val == target)` for slices.
