## 2024-05-24 - Faster parsing of system files
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in `/proc/stat` or `/proc/loadavg`), `str::split_whitespace()` performs unnecessary full Unicode property checks.
**Action:** Use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` for these files to bypass the Unicode overhead and achieve a significant performance improvement in hot paths.
