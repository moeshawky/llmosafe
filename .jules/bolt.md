## 2025-05-18 - Optimize pairwise differences sum to O(1)
**Learning:** Calculating trend by summing consecutive pairwise differences (s[i] - s[i-1]) in a sequence telescopes to simply s[n-1] - s[0]. Iterating over the collection is an unnecessary O(N) operation.
**Action:** Recognize telescoping sums in statistical and tracking windows to replace O(N) loops with O(1) bounds lookups.
