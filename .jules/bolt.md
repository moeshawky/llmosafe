## 2025-03-08 - [Optimize manual array iteration]
**Learning:** [Using `.iter().take(count)` for small fixed arrays incurs overhead compared to slice `.contains()` method which leverages direct CPU instructions.]
**Action:** [Prefer `[..count].contains(&val)` over manual array iteration loop for small structures in Rust.]
