## 2024-05-24 - [Avoid iter().take() in hot loops]
**Learning:** Using slice `[..count].contains()` is significantly faster than `iter().take(count)` combined with manual loops for searching small arrays.
**Action:** Prefer slice `contains` over manual iterator limiting when searching arrays.
