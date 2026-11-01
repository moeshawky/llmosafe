## 2024-11-01 - Replace split_whitespace with split_ascii_whitespace in ASCII-only parsing
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), using `str::split_ascii_whitespace()` instead of `str::split_whitespace()` bypasses full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` when processing guaranteed ASCII text streams.
