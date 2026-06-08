## 2024-05-30 - Invalid UTF-8 due to arbitrary truncation
**Vulnerability:** The C-ABI `store_objective` function blindly truncated the string at a fixed byte offset (`MAX_OBJECTIVE_LEN - 1`) which can split multi-byte UTF-8 characters. This caused the subsequent `from_utf8_unchecked` to yield undefined behavior or panic down the line if the invalid bytes were used.
**Learning:** `&str` byte truncation must respect character boundaries (`is_char_boundary`).
**Prevention:** Use `while !input.is_char_boundary(len) { len -= 1; }` when truncating string slices, especially before `unsafe` blocks expecting valid UTF-8.
