## 2026-10-24 - Use split_ascii_whitespace for system files
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data (like Linux metrics in `/proc/stat`) using `split_whitespace()` incurs unnecessary overhead from full Unicode property checks.
**Action:** Use `str::split_ascii_whitespace()` instead to bypass these checks and significantly improve performance in hot paths.
