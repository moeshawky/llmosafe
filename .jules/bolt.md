## 2024-11-09 - Faster parsing of Linux /proc files
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc`), using `split_ascii_whitespace()` instead of `split_whitespace()` yields a significant performance improvement in hot paths because it bypasses full Unicode property checks.
**Action:** Always use `split_ascii_whitespace()` for processing guaranteed ASCII system metric files like `/proc/stat`, `/proc/loadavg`, and `/proc/meminfo`.
