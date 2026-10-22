## 2024-05-23 - ArrayVec Overhead in Hot Loops
**Learning:** In no_std environments, generic wrappers like ArrayVec incur measurable initialization and iterator overhead when used for small temporary arrays in hot loops. Additionally, iterator chains are slower than slice .contains() for primitive searches.
**Action:** Replace temporary ArrayVec instances with simple stack-allocated arrays (e.g., [0u32; N]) and manual length counters, and use slice .contains() for primitive lookups in hot loops.
