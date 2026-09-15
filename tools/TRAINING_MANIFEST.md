# TRAINING_MANIFEST.md — 20k TF-IDF Classifier Training Manifest

**Version:** `tools/trainer_version.json` v1.0.0-counttf-20k  
**Date:** 2026-09-15  
**Trainer seed:** 42  
**Artifact hash:** `0955b7b221a5aaa01d2d607db74d6ea55e0caa085438518e06bbc2c6d64150cc`  
**Architecture:** Arm A — Single TF-IDF 20k, count-based TF  

---

## Trainer Version

| Component | Version |
|-----------|---------|
| Trainer | `1.0.0-counttf-20k` |
| Tokenizer | FNV-1a 64-bit, unigrams+bigrams, ASCII lowercase, 256B cap |
| TF semantics | Count-based (`X[i,idx] = idf[idx] × count(term, text[i])`) |
| Classifier | LogisticRegression (liblinear, C=1.0, balanced) |
| Threshold | -1.655797 |
| Intercept | 2.673815 |
| Vocab size | 20,000 |

---

## Count-TF Semantics Version

Count-based TF: every occurrence of a term contributes to the feature count. This matches Rust inference semantics at `src/llmosafe_classifier.rs:282` where `score += idf * coef` per matched token. Corrected from boolean/set TF (P0_RULINGS.md P0.1).

---

## Dataset Revisions + Split Hashes

| Split | SHA256 | Rows |
|-------|--------|------|
| canonical_train | `836417ba040a5c56c894c7539d3e52d2709cd91a4a9e9d72b7700bc3a58a5f32` | 35,490 |
| calibration | `13fce5cdeddd82e70cbdc5692099b8d3fa0d2c755a076944a33a3c799a8b3627` | 13,559 |
| historical_release_holdout_v1 | `37bed07b89fe7c4f804048e99216a6da8793ac39678547a3732203b60309e` | 974 |
| OOD non-English | *(loaded)* | 4,969 |
| Validation | *(loaded)* | 15,673 |
| Technical quoted | *(loaded)* | 209 |

**Source commits:** shieldlm `dd660ae7670a`, neuralchemy `7d70432dfcf4`, deepset `4f61ecb038e9`.

---

## Artifact Hash

`tools/vocab_model.bin`: SHA256 `0955b7b221a5aaa01d2d607db74d6ea55e0caa085438518e06bbc2c6d64150cc`, size 320,012 bytes. Determinism proven — byte-identical across two independent training passes (PASS 1 = PASS 2).

---

## Determinism Statement

Seed: `TRAINER_SEED = 42`. Feature extraction deterministic (FNV-1a tokenizer). Vocab selection by frequency ranking with hash tie-breaking. IDF deterministic. Logistic regression with `random_state=42`, `solver='liblinear'` (deterministic). Threshold calibration deterministic percentile scan. Serialization fixed struct format sorted by hash. Verified byte-identical across 2 passes.

---

## Pointer

Full training evidence: `/tmp/opencode/sifter-release/TRAINING_REPORT.md`
