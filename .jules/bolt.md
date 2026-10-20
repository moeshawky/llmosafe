## 2024-05-24 - Use split_ascii_whitespace for /proc/stat
**Learning:** /proc/stat is guaranteed to be ASCII. Using `split_whitespace()` incurs unnecessary Unicode property check overhead.
**Action:** Use `split_ascii_whitespace()` when parsing known ASCII system files.
