# Classifier Training — Reproducibility Guide

## Model provenance

The TF-IDF logistic regression classifier embedded in `llmosafe` at build time
(via `build.rs` → `tools/vocab_model.bin`) is trained on:

| Source | Samples | Description |
|--------|---------|-------------|
| ShieldLM (`dmilush/shieldlm-prompt-injection`) | ~23,000 | Prompt injection detection |
| neuralchemy (`neuralchemy/Prompt-injection-dataset`) | ~12,000 | Prompt injection / jailbreak |
| deepset (`deepset/prompt-injections`) | ~7,000 | Prompt injection detection |
| Generated sifter data (optional) | ~1,600 | Synthetic multi-category samples |

**Total**: ~44,500 samples (~61% legitimate / ~39% manipulation)

## Architecture

```
  Raw text
     │
     ▼
  StreamingTokenizer ─── FNV-1a 64-bit hashes (unigrams + bigrams)
     │
     ▼
  Binary search in VOCAB[] ── (hash, idf, coef) lookup tables
     │
     ▼
  score = INTERCEPT + Σ(idf × coef)     ← logistic regression
     │
     ▼
  sigmoid_lut(score) → probability
     │
     ▼
  score > THRESHOLD → ClassificationResult
```

- **Feature selection**: Top-N terms by mutual information with label
- **Tokenization**: `[a-zA-Z0-9]+` case-folded, max 256 bytes, adjacent bigrams
- **Bigram hash**: `((prev_hash ^ 0x5F) × FNV_PRIME ^ token_hash) × FNV_PRIME`
- **Hash function**: FNV-1a, 64-bit
- **Classifier**: `sklearn.linear_model.LogisticRegression`, `class_weight='balanced'`, `C=0.1`
- **Serialization**: Little-endian binary — `u32(count) + f32(threshold) + f32(intercept) + N×[u64(hash) + f32(idf) + f32(coef)]`

## One-command reproduce

```bash
# Prerequisites
pip install datasets scikit-learn numpy

# Full pipeline (download HuggingFace, train classifier)
python tools/train_full_pipeline.py

# With custom vocab size
python tools/train_full_pipeline.py --vocab-size 5000

# Skip download (data already on disk)
python tools/train_full_pipeline.py --no-download

# Include synthetic sifter data (needs API key)
python tools/train_full_pipeline.py --generate-sifter
```

## Manual step-by-step

### 1. Download real training data

```bash
python tools/download_hf_datasets.py
# → data/corpus_huggingface.jsonl  (~42,845 samples)
```

Requires `pip install datasets`. Downloads from HuggingFace Hub.

### 2. (Optional) Generate synthetic sifter data

```bash
cd tools/generate_sifter_data
python gen.py --all
# → data/classifier_training.jsonl  (~1,600 samples)
```

Requires `NVIDIA_API_KEY` or `DEEPSEEK_API_KEY` environment variable.
Uses `mistralai/mistral-small-4-119b-2603` via NVIDIA NIM or DeepSeek V4.

Takes several hours. Rate-limited with exponential backoff (2^miss capped at 30s).

### 3. Combine corpora

```bash
# The pipeline script handles this automatically.
# Manual equivalent:
python -c "
import json
# ... merge corpus_huggingface.jsonl + classifier_training.jsonl
# ... → data/corpus_combined.jsonl
"
```

### 4. Train classifier

```bash
python tools/train_tfidf_classifier.py \
  --corpus-jsonl data/corpus_combined.jsonl \
  --output tools/vocab_model.bin \
  --vocab-size 3000
```

Outputs:
- `tools/vocab_model.bin` — binary classifier model (46.9 KB at 3000 features)
- Console: accuracy, precision, recall, F1, confusion matrix

### 5. Build and test

```bash
cargo build --all-features
cargo test --all-features
```

The `build.rs` script reads `tools/vocab_model.bin` and generates
`generated_vocab.rs` at compile time. If the model file is missing,
a fail-closed fallback is used (empty vocab, `INTERCEPT = -2.0` —
all inputs classified as safe).

## Quality gates

The training pipeline enforces:

| Gate | Threshold | Meaning |
|------|-----------|---------|
| Recall ≥ 0.85 | 85% | Model catches enough manipulations |
| Precision ≥ 0.70 | 70% | Model doesn't over-flag safe text |

If either gate fails, the model is **rejected** and not written to disk.

## Current model (devel branch)

| Metric | Value |
|--------|-------|
| Vocab size | 3,000 |
| Model size | 46.9 KB |
| Training samples | 44,499 |
| Accuracy | 94.0% |
| Precision | 93.8% |
| Recall | 90.5% |
| F1 | 92.1% |
| Intercept | -1.19 |
| Threshold | 0.43 |
| Features | 3,000 (MI-selected) |

## Sifter data generation reference

The `tools/generate_sifter_data/` directory contains the full LLM-based
data generation pipeline:

| Module | Purpose |
|--------|---------|
| `gen.py` | Sequential generation driver with resume capability |
| `config.py` | Category definitions, model endpoints, rate limits |
| `schemas.py` | JSON Schema validation for generated samples |
| `validator.py` | Keyword + TF-IDF classifier mirror for label verification |
| `filter.py` | Semantic density scoring for sample quality |
| `exporter.py` | JSONL output with metadata |
| `llm_client.py` | DeepSeek API client with connection pooling |
| `nv_client.py` | NVIDIA NIM client with 4-key rotation |
| `prompts/generation.py` | Category-specific prompt templates |
| `products/classifier_training.py` | Training pair product generator |
| `products/keyword_regression.py` | Deterministic keyword test suite |
| `hammer.py` | High-throughput batch dispatcher |
| `farm.py` | Multi-category farming with state persistence |
| `token_bucket.py` | Rate limit enforcement |
| `batch.sh` | Shell wrapper for parallel farm dispatch |

### Categories generated

| Category | Label | Description |
|----------|-------|-------------|
| `clean_safe` | 0 | Legitimate, non-manipulative text |
| `gaslighting` | 1 | Reality-distortion manipulation |
| `fear_tactics` | 1 | Threat-based pressure |
| `false_dichotomy` | 1 | Forced binary choices |
| `cognitive_load` | 1 | Deliberate confusion |
| `anchoring` | 1 | Reference-point manipulation |
| `framing_bias` | 1 | Narrative spin |
| `slippery_slope` | 1 | Extrapolated-consequence fear |
| `ad_hominem` | 1 | Personal attack |
| `loaded_language` | 1 | Emotionally charged wording |
| `conspiracy` | 1 | Hidden-agenda narrative |
| `authority_bias` | 1 | Credential/position appeal |

Each category is generated at 3 difficulty tiers (easy/medium/hard),
with 30-200 samples per tier per category, using progressive prompt
escalation (template → few-shot → adversarial).

## Known limitations

1. **Prompt-injection bias**: The HuggingFace datasets focus on prompt
   injection and jailbreak detection. General "safe" text classification
   is less precise than injection detection.

2. **Simulate/jailbreak vocabulary overlap**: Words like "simulate",
   "test environment", "network topology" appear in both legitimate
   engineering contexts and jailbreak templates. The model tends to
   be conservative with these — legitimate "simulate" usage may still
   score near the threshold. Added calibration negatives to mitigate.

3. **No_std limitation**: The Rust classifier does not use token
   normalization (stemming, lemmatization) to keep zero-allocation
   streaming. This trades precision for WASM/embedded compatibility.

4. **Bigram feature space**: With FNV-1a 64-bit hash collision
   probability at 5000 features ≈ 3×10⁻¹⁰, collisions exist but are
   statistically negligible for safety-critical decisions.
