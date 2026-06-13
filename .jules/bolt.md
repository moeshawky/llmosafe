## 2024-06-03 - Avoid .skip() on string split iterators
**Learning:** Using `.skip()` on string split iterators like `split_whitespace()` causes measurable performance overhead due to re-iteration over characters when compared to pre-parsing everything into a contiguous fixed-size array in a single pass.
**Action:** When implementing functions requiring multiple passes over substrings (especially in hot loop pathways), parse the string once into a single array structure instead of sequentially calling splitting functions or manually skipping via `.skip()`.
## 2024-06-13 - Reusing Vec Allocation in Hot Loops
**Learning:** Re-allocating a new `Vec` inside a tight loop for parsing string constants incurs measurable overhead.
**Action:** Lift the `Vec` allocation outside the hot loop and reuse it via `.clear()` and `.extend()` to minimize dynamic allocations while maintaining safety bounds.
