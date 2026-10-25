## 2023-10-25 - [Replacing `split_whitespace` with `split_ascii_whitespace` for guaranteed ASCII system files]
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (like Linux system metrics in `/proc/stat` or `/proc/loadavg`) should use `str::split_ascii_whitespace()` instead of `str::split_whitespace()`. It bypasses full Unicode property checks, yielding significant performance improvement in hot paths.
**Action:** When parsing `/proc` or `/sys` files, or any known ASCII-only data, use `split_ascii_whitespace()`.
