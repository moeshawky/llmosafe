## 2024-03-24 - Initial
## 2024-05-24 - split_ascii_whitespace for system files
**Learning:** In Rust, parsing system files (like `/proc/stat`) guaranteed to contain ASCII data using `split_ascii_whitespace()` is significantly faster than `split_whitespace()`, as it skips full Unicode property checks.
**Action:** Always prefer `split_ascii_whitespace()` when processing guaranteed ASCII text streams, especially in hot paths like resource metrics polling.
