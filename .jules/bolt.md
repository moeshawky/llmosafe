## 2024-05-24 - Array Search Optimization
**Learning:** In Rust no_std environments, searching small fixed-size arrays within hot loops is significantly faster using the slice `.contains()` method over manual iterator combinations like `.iter().take(count)`, avoiding iterator overhead.
**Action:** Always prefer slice `.contains()` for searching arrays when bounds are strictly mathematically guaranteed.
