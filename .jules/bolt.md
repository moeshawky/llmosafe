
## 2024-05-19 - [O(N*M) calculation with fixed-size cache optimization in `no_std`]
**Learning:** Found a quadratic loop `calculate_utility` where it re-iterates and re-trims `objective` for every word in `observation`. We can optimize this by caching up to 64 words on the stack to maintain `no_std` zero-allocation guarantees while drastically reducing work for shorter sentences.
**Action:** Use an array `[""; 64]` as a cache for trimmed words, with an iterator clone fallback for objectives larger than the cache.
