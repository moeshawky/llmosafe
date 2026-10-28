## 2026-10-28 - Optimize ASCII-only file parsing
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), using `str::split_ascii_whitespace()` instead of `str::split_whitespace()` bypasses full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Use `split_ascii_whitespace()` instead of `split_whitespace()` when parsing known ASCII-only system files.
