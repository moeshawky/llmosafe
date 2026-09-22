## 2024-05-23 - Optimizing Array Search in Hot Loops
**Learning:** In Rust, searching small fixed-size arrays with manual iteration (`.iter().take(count)`) incurs measurable iterator overhead compared to slice methods.
**Action:** Always prefer using the slice `.contains()` method (e.g., `[..count].contains(&val)`) instead of manual iteration to leverage direct CPU instructions and avoid iterator overhead.
