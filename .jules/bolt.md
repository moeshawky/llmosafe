## 2025-01-01 - Avoid iterator overhead in small fixed array searches
**Learning:** Using `.iter().take(n)` on fixed-size arrays introduces measurable overhead compared to `.contains()` or slice iterators (`&array[..n]`).
**Action:** Always prefer slice methods like `.contains()` or direct slice iteration when searching populated prefixes of fixed-size arrays in hot loops.
