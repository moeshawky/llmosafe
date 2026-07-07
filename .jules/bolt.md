## 2024-05-24 - Optimization: Inline Hash Iteration

**Learning:** When searching small fixed-size arrays within hot loops, utilizing iterator methods like `.iter().any()` is faster than allocating intermediate collections like `ArrayVec` or manual array creation, especially when the intermediate collection is just a mapping of the stream.
**Action:** Always prefer inline iteration or `.contains()`/`.iter().any()` on pre-existing structures directly inside hot string-processing loops instead of allocating intermediate buffers.
