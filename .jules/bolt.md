## 2025-02-12 - [Initial performance investigation]
**Learning:** Checking for bottlenecks in streaming string tokenization.
**Action:** Optimize tokenization loop if needed.
## 2025-02-12 - [Pre-parsing split iterators]
**Learning:** Found optimization in llmosafe_sifter.rs: calculate_utility function parses 'objective' words upfront to avoid O(N*M) string parsing overhead in hot loops.
**Action:** The same optimization technique can be applied to other string split operations in similar hot loops across the codebase if any exist.
## 2025-02-12 - [Tiny file parsing]
**Learning:** Found optimization in llmosafe_body.rs: For reading tiny pseudo-files (like /proc/loadavg or /proc/stat) where only the first line is needed, prefer `fs::read_to_string` over `BufReader::lines().next()` to avoid the overhead of dynamically allocating and setting up `BufReader`'s internal buffer.
**Action:** Verify if `llmosafe_body.rs` uses `fs::read_to_string` and extract string slices directly instead of full `lines()` splits.
## 2025-02-12 - [Streaming Tokens]
**Learning:** Found optimization in llmosafe_detection.rs: In `observe()`, streaming string tokens via `split_whitespace().take(N)` can be hashed and checked against the `goal_hashes` array inline, avoiding the intermediate `ArrayVec` allocation.
**Action:** Optimize `observe()` in `DriftDetector`.
