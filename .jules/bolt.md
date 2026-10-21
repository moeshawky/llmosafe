## 2025-01-01 - Optimize array searches in hot loops
**Learning:** In Rust hot loops, manual iteration with `.iter().take()` over fixed-size arrays incurs measurable iterator overhead compared to slice operations.
**Action:** Always prefer slice methods like `.contains()` (e.g., `[..count].contains(&val)`) when searching small fixed-size arrays within hot loops.
