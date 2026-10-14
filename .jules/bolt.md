## 2024-10-14 - Optimize ASCII parsing in system metrics
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` to bypass full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Always prefer `split_ascii_whitespace` over `split_whitespace` when parsing predictable system files to avoid Unicode overhead.
