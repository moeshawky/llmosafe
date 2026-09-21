## 2025-02-18 - Avoid iterator overhead in small fixed-size array searches
**Learning:** Using manual iteration with `.iter().take(count)` for searching small fixed-size arrays within hot loops incurs measurable performance overhead compared to direct slice operations.
**Action:** Always prefer using the built-in slice `.contains(&target)` method (e.g., `slice[..count].contains(&val)`) for exact matches, or direct slice iteration (e.g., `for item in &slice[..len]`) for custom comparisons like `eq_ignore_ascii_case()` to leverage direct CPU instructions and avoid iterator abstraction overhead.
