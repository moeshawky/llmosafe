## 2024-09-20 - Slice contains optimization
**Learning:** In Rust, when searching small fixed-size arrays within hot loops, manual iteration with `.iter().take(count)` incurs iterator overhead compared to slice methods.
**Action:** Always prefer using the slice `.contains()` method (e.g., `[..count].contains(&val)`) to leverage direct CPU instructions and avoid iterator overhead.
