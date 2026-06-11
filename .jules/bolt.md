## 2024-06-11 - Optimization in String Matching Loops
**Learning:** We observed significant overhead when trying to search two subsets of parsed string iterators by skipping through string split iterators. Using `iterator.skip(N)` in tight O(N*M) text processing loops generates measurable performance loss.
**Action:** When comparing arrays of split strings, pre-parse them into a single contiguous structure to avoid iterating/skipping overhead.
