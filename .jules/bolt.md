## 2024-10-24 - slice iter() any() vs take() overhead
**Learning:** In Rust `no_std` environments, slice `.contains()` cannot be used for case-insensitive string slice comparisons because it performs a case-sensitive exact match. When falling back to manual loops with `.eq_ignore_ascii_case()`, `.iter().any()` on a pre-sliced active array `&obj_words[..obj_len]` is faster than `.iter().take(len)` combined with a manual loop and `break`.
**Action:** Use `.iter().any()` on pre-sliced active arrays when possible to avoid iterator `.take()` overhead in hot loops.
