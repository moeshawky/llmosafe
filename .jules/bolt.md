## 2025-05-18 - Optimize array search in hot loops
**Learning:** When searching small fixed-size arrays within hot loops, using `.iter().take(count)` incurs measurable iterator overhead compared to slice `.contains()`.
**Action:** Prefer using the slice `.contains()` method (e.g., `[..count].contains(&val)`) instead of manual iteration to leverage direct CPU instructions.
