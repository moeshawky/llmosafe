## 2024-05-01 - Avoid Unicode overhead when parsing ASCII system files
**Learning:** System metric files like `/proc/stat` and `/proc/loadavg` are guaranteed to contain only ASCII characters. Using `split_whitespace()` incurs unnecessary Unicode property check overhead in these hot paths.
**Action:** Always prefer `split_ascii_whitespace()` when parsing known ASCII-only data to improve performance.
