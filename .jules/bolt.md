## 2024-10-12 - Optimize System File Parsing with ASCII Whitespace Split
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` to bypass full Unicode property checks, yielding a significant performance improvement in hot paths.
**Action:** Use `split_ascii_whitespace` instead of `split_whitespace` when processing system files in Linux.
