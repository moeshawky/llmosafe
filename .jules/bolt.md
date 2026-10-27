## 2024-10-27 - Fast ASCII parsing in /proc files
**Learning:** Using `split_whitespace()` for parsing `/proc` files incurs unnecessary Unicode property checks. Linux system metric files are guaranteed to be ASCII.
**Action:** Always use `str::split_ascii_whitespace()` instead of `str::split_whitespace()` when parsing known ASCII text like system metrics, which provides up to a 2x speedup in hot paths.
