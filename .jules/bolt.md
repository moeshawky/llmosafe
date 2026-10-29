## 2024-05-24 - Faster parsing of system metrics
**Learning:** When parsing system files guaranteed to contain only ASCII data (like Linux metrics in `/proc/stat` or `/proc/loadavg`), using `str::split_whitespace()` incurs overhead from full Unicode property checks.
**Action:** Use `str::split_ascii_whitespace()` instead for parsing such files to improve performance in hot paths.
