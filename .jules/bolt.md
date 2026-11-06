
## 2024-11-06 - Fast Path for Case-Insensitive Slices
**Learning:** In Rust `no_std` environments, case-insensitive string searches using slices (e.g., `window.eq_ignore_ascii_case`) evaluate every byte. Adding a fast path to match the first character (`window[0].to_ascii_lowercase() == pattern[0].to_ascii_lowercase()`) before doing full comparison reduces search time significantly (from ~401ms to ~224ms for 1M checks).
**Action:** When implementing substring matching via `eq_ignore_ascii_case`, check the first byte before delegating to the full case-insensitive string check to skip invalid window early.
