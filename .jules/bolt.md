## 2026-10-16 - Use split_ascii_whitespace for system files
**Learning:** Parsing ASCII-only system files (like /proc metrics) with standard split_whitespace incurs unnecessary Unicode overhead.
**Action:** Use split_ascii_whitespace() to parse data from system files to bypass full Unicode property checks and improve performance in hot paths.
