## 2024-10-24 - Faster parsing of ASCII system files
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (like Linux system metrics) using `str::split_whitespace()` incurs unnecessary Unicode property checks.
**Action:** Always use `str::split_ascii_whitespace()` for these files to bypass full Unicode property checks and improve performance in hot paths.
