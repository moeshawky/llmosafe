## 2026-10-26 - [Use split_ascii_whitespace for system files]
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`) should use `str::split_ascii_whitespace()` instead of `str::split_whitespace()`. It bypasses full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Replace `split_whitespace()` with `split_ascii_whitespace()` when processing ASCII-only inputs like Linux `/proc/` files.
