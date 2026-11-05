## 2024-05-25 - Use split_ascii_whitespace for guaranteed ASCII sources
**Learning:** In Rust, parsing system files like Linux `/proc` metrics (which are guaranteed ASCII) using `split_whitespace()` unnecessarily performs full Unicode property checks.
**Action:** Always use `split_ascii_whitespace()` when processing guaranteed ASCII strings (like `/proc` files) to bypass Unicode checks for significant performance gains in hot paths.
