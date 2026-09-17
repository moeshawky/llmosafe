## 2024-05-24 - Avoid dynamic BufReader for tiny files
**Learning:** For reading tiny pseudo-files (like `/proc/loadavg`, `/proc/stat`, or small proc files like `/proc/self/status` where only one line is needed), prefer `fs::read_to_string` over `BufReader::lines().next()` to avoid the overhead of dynamically allocating and setting up `BufReader`'s internal buffer, despite possible variations in microbenchmark results.
**Action:** Replace `BufReader::new(file).lines()` with `fs::read_to_string` for single-read or tiny pseudo-files.
## 2024-05-24 - Avoid iterator overhead for small slice searches
**Learning:** In Rust, when searching small fixed-size arrays within hot loops, prefer using the slice `.contains()` method (e.g., `[..count].contains(&val)`) instead of manual iteration with `.iter().take(count)` to leverage direct CPU instructions and avoid iterator overhead.
**Action:** Replace `seen.iter().take(count)` with `seen[..count].contains(&val)` when searching arrays.
