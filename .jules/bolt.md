## 2024-05-18 - Replacing split_whitespace with split_ascii_whitespace
**Learning:** In Rust, parsing system files guaranteed to contain only ASCII data using str::split_whitespace() incurs measurable performance overhead due to Unicode property checks.
**Action:** Use str::split_ascii_whitespace() instead for purely ASCII data like /proc/stat.
