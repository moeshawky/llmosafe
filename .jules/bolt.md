## 2024-10-06 - Replace split_whitespace with split_ascii_whitespace for ASCII system files
**Learning:** In Rust, when parsing system files guaranteed to contain only ASCII data (such as Linux system metrics in /proc/stat or /proc/loadavg), using str::split_whitespace() incurs unnecessary performance overhead from full Unicode property checks.
**Action:** Use str::split_ascii_whitespace() instead of str::split_whitespace() when parsing system files known to be ASCII-only to significantly improve performance in hot paths.
