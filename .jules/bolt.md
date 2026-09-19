## 2024-05-18 - Avoid iterator overhead in hot loops
**Learning:** Using `.iter().take(n)` on fixed-size arrays or slices inside hot loops incurs measurable performance overhead compared to direct slice operations. Additionally, slice `.contains()` should be used for simple value searches, but direct iteration over a slice `&slice[..n]` with `.eq_ignore_ascii_case()` is required for case-insensitive string matching.
**Action:** Replace `.iter().take(n)` with slice slicing `&slice[..n]`. Use `.contains(&val)` when searching for primitive values, and direct iteration for string comparisons.
