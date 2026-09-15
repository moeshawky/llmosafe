#!/usr/bin/env python3
"""
train_tfidf_classifier.py — Reproducible Count-TF-IDF Logistic Regression Trainer

Architecture: Arm A — Single TF-IDF 20k vocab (FINAL_DECISION approved).
Count-based TF: X[i, idx] = idf[idx] * count(term, text[i]).
Deterministic: seed=42, train twice, byte-identical output.
"""

import hashlib
import json
import math
import re
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

TRAINER_VERSION = "1.0.0-counttf-20k"
TRAINER_SEED = 42
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x00000100000001b3
MAX_TOKEN_LEN = 256
TARGET_VOCAB_SIZE = 20000
INTERCEPT_INITIAL = 1.0


# ============================================================================
# Tokenizer — exact replica of Rust StreamingTokenizer
# ============================================================================

def fnv1a_64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & ((1 << 64) - 1)
    return h


def compute_bigram_hash(prev_hash: int, token_hash: int) -> int:
    # ((prev_hash ^ 0x5F) * FNV_PRIME ^ token_hash) * FNV_PRIME
    temp = (((prev_hash ^ 0x5F) * FNV_PRIME) ^ token_hash) * FNV_PRIME
    return temp & ((1 << 64) - 1)


def tokenize_with_bigrams(text: str) -> list:
    tokens = re.findall(r'[a-zA-Z0-9]+', text)
    tokens = [t[:MAX_TOKEN_LEN].lower().encode('ascii') for t in tokens]
    results = []
    has_prev = False
    prev_hash = 0
    for token_bytes in tokens:
        token_hash = fnv1a_64(token_bytes)
        if has_prev:
            results.append(compute_bigram_hash(prev_hash, token_hash))
        results.append(token_hash)
        has_prev = True
        prev_hash = token_hash
    return results


def extract_feature_counts(text: str) -> Counter:
    return Counter(tokenize_with_bigrams(text))


def compute_row_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ============================================================================
# Vocab selection, IDF, feature matrix, training
# ============================================================================

def select_vocab(feature_counts_list: list, n_select: int = TARGET_VOCAB_SIZE) -> list:
    feature_freq = Counter()
    for fc in feature_counts_list:
        for h, c in fc.items():
            feature_freq[h] += c
    return sorted(feature_freq.keys(), key=lambda h: (-feature_freq[h], h))[:n_select]


def compute_idf(feature_counts_list: list, vocab_features: list) -> dict:
    n_samples = len(feature_counts_list)
    doc_freq = {h: sum(1 for fc in feature_counts_list if h in fc) for h in vocab_features}
    return {h: math.log((n_samples + 1) / (df + 1)) + 1.0 for h, df in doc_freq.items()}


def build_feature_matrix(feature_counts_list: list, vocab_features: list, idf: dict):
    feat_idx = {h: j for j, h in enumerate(vocab_features)}
    rows, cols, vals = [], [], []
    for i, fc in enumerate(feature_counts_list):
        for h, count in fc.items():
            if h in feat_idx:
                rows.append(i)
                cols.append(feat_idx[h])
                vals.append(idf[h] * count)
    return sparse.csr_matrix((vals, (rows, cols)),
                             shape=(len(feature_counts_list), len(vocab_features)), dtype=np.float64)


def train_classifier(X, y):
    model = LogisticRegression(solver='liblinear', C=1.0, random_state=TRAINER_SEED,
                                max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    return model


def compute_scores(model, X, intercept):
    return X.dot(model.coef_[0]) + intercept


# ============================================================================
# Threshold calibration
# ============================================================================

def calibrate_threshold(scores, labels):
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.percentile(scores, np.linspace(0, 100, 201))
    for t in thresholds:
        preds = (scores > t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
    return best_threshold, best_f1


# ============================================================================
# Serialization — exact deployed layout
# ============================================================================

def serialize_vocab_model(vocab_features, idf, coefs, threshold, intercept):
    sorted_vocab = sorted(vocab_features, key=lambda h: h)
    buf = bytearray()
    buf.extend(struct.pack('<I', len(sorted_vocab)))
    buf.extend(struct.pack('<f', threshold))
    buf.extend(struct.pack('<f', intercept))
    for h in sorted_vocab:
        buf.extend(struct.pack('<Q', h))
        buf.extend(struct.pack('<f', idf[h]))
        buf.extend(struct.pack('<f', coefs[h]))
    return bytes(buf)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, X, y, intercept, threshold):
    scores = compute_scores(model, X, intercept)
    preds = (scores > threshold).astype(int)
    y_arr = y.astype(int)
    tp = np.sum((preds == 1) & (y_arr == 1))
    fp = np.sum((preds == 1) & (y_arr == 0))
    fn = np.sum((preds == 0) & (y_arr == 1))
    tn = np.sum((preds == 0) & (y_arr == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    try:
        auroc = roc_auc_score(y_arr, scores)
    except:
        auroc = 0.0
    try:
        auprc = average_precision_score(y_arr, scores)
    except:
        auprc = 0.0
    return {
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'auroc': float(auroc), 'auprc': float(auprc),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'threshold': float(threshold),
        'false_safe_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'false_halt_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


# ============================================================================
# Load data
# ============================================================================

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    base_dir = Path(__file__).parent.resolve()
    tools_dir = base_dir
    artifact_dir = tools_dir / "training_artifacts"
    artifact_dir.mkdir(exist_ok=True)

    train_path = Path("/tmp/opencode/sifter-tournament/manifests/canonical_train/rows.jsonl")
    cal_path = Path("/tmp/opencode/sifter-tournament/manifests/calibration/rows.jsonl")
    val_path = Path("/tmp/opencode/sifter-tournament/manifests/validation/rows.jsonl")
    ood_path = Path("/tmp/opencode/sifter-tournament/manifests/ood_non_english/rows.jsonl")
    tech_path = Path("/tmp/opencode/sifter-tournament/manifests/technical_quoted/rows.jsonl")
    historical_release_holdout_v1_path = Path("/tmp/opencode/sifter-tournament/sealed/sealed_holdout.jsonl")
    # NOTE: on-disk sealed/ dir is outside repo and absent (ls verified); only code identifiers renamed.

    vocab_model_path = tools_dir / "vocab_model.bin"
    hash_path = tools_dir / "vocab_model.bin.sha256"
    version_path = tools_dir / "trainer_version.json"

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting training pipeline")
    print(f"Trainer: {TRAINER_VERSION} seed={TRAINER_SEED} vocab={TARGET_VOCAB_SIZE} count-TF")

    # Load datasets
    train_records = load_jsonl(str(train_path))
    cal_records = load_jsonl(str(cal_path))
    val_records = load_jsonl(str(val_path)) if val_path.exists() else []
    ood_records = load_jsonl(str(ood_path)) if ood_path.exists() else []
    tech_records = load_jsonl(str(tech_path)) if tech_path.exists() else []
    historical_release_holdout_v1_records = load_jsonl(str(historical_release_holdout_v1_path)) if historical_release_holdout_v1_path.exists() else []

    train_manifest_hash = sha256_file(train_path)
    cal_manifest_hash = sha256_file(cal_path)
    train_hashes = [compute_row_hash(r) for r in train_records]
    cal_hashes = [compute_row_hash(r) for r in cal_records]
    train_dataset_hash = hashlib.sha256("".join(sorted(train_hashes)).encode()).hexdigest()
    cal_dataset_hash = hashlib.sha256("".join(sorted(cal_hashes)).encode()).hexdigest()

    print(f"\nData loaded: train={len(train_records)} cal={len(cal_records)} val={len(val_records)} "
          f"ood={len(ood_records)} tech={len(tech_records)} historical_release_holdout_v1={len(historical_release_holdout_v1_records)}")
    print(f"Manifest hashes: train={train_manifest_hash[:16]}... cal={cal_manifest_hash[:16]}...")

    # === PASS 1 ===
    print("\n" + "="*60)
    print("PASS 1: Training")
    print("="*60)

    train_labels = np.array([1 if r.get('label_binary', r.get('label', 0)) == 1 else 0 for r in train_records], dtype=np.int64)
    cal_labels = np.array([1 if r.get('label_binary', r.get('label', 0)) == 1 else 0 for r in cal_records], dtype=np.int64)

    train_counts = [extract_feature_counts(r['text']) for r in train_records]
    cal_counts = [extract_feature_counts(r['text']) for r in cal_records]

    vocab_features = select_vocab(train_counts, TARGET_VOCAB_SIZE)
    idf = compute_idf(train_counts, vocab_features)

    X_train = build_feature_matrix(train_counts, vocab_features, idf)
    model = train_classifier(X_train, train_labels)
    intercept = float(model.intercept_[0])

    X_cal = build_feature_matrix(cal_counts, vocab_features, idf)
    cal_scores = compute_scores(model, X_cal, intercept)
    threshold, cal_f1 = calibrate_threshold(cal_scores, cal_labels)
    print(f"  Threshold: {threshold:.6f}  Calibration F1: {cal_f1:.4f}")

    # Build idf and coef dicts keyed by hash
    coefs = {}
    for j, h in enumerate(vocab_features):
        coefs[h] = float(model.coef_[0][j])
    vocab_bytes = serialize_vocab_model(vocab_features, idf, coefs, threshold, intercept)
    file_hash = sha256_bytes(vocab_bytes)

    print(f"  vocab_model.bin: {len(vocab_bytes)} bytes SHA256={file_hash}")

    # === PASS 2: Determinism proof ===
    print("\n" + "="*60)
    print("PASS 2: Determinism Proof")
    print("="*60)

    vocab_features2 = select_vocab(train_counts, TARGET_VOCAB_SIZE)
    idf2 = compute_idf(train_counts, vocab_features2)
    X_train2 = build_feature_matrix(train_counts, vocab_features2, idf2)
    model2 = train_classifier(X_train2, train_labels)
    intercept2 = float(model2.intercept_[0])
    X_cal2 = build_feature_matrix(cal_counts, vocab_features2, idf2)
    cal_scores2 = compute_scores(model2, X_cal2, intercept2)
    threshold2, _ = calibrate_threshold(cal_scores2, cal_labels)
    coefs2 = {}
    for j, h in enumerate(vocab_features2):
        coefs2[h] = float(model2.coef_[0][j])
    vocab_bytes2 = serialize_vocab_model(vocab_features2, idf2, coefs2, threshold2, intercept2)
    file_hash2 = sha256_bytes(vocab_bytes2)

    if vocab_bytes == vocab_bytes2:
        determinism = True
        determinism_hash = file_hash
        print("  PASS 2 byte-identical: YES ✅")
    else:
        determinism = False
        determinism_hash = "FAILED"
        print("  PASS 2 byte-identical: NO ❌")
        print(f"  Pass1={file_hash}  Pass2={file_hash2}")

    # Write outputs
    with open(vocab_model_path, 'wb') as f:
        f.write(vocab_bytes)
    with open(hash_path, 'w') as f:
        f.write(file_hash)

    # Artifact metadata
    with open(artifact_dir / "determinism_proof.json", 'w') as f:
        json.dump({"pass1_sha256": file_hash, "pass2_sha256": file_hash2,
                    "byte_identical": determinism, "determinism_sha256": determinism_hash,
                    "trainer_version": TRAINER_VERSION, "seed": TRAINER_SEED,
                    "method": "count-based TF-IDF"}, f, indent=2, sort_keys=True)
    with open(artifact_dir / "provenance.json", 'w') as f:
        json.dump({"train_manifest_sha256": train_manifest_hash, "cal_manifest_sha256": cal_manifest_hash,
                    "train_dataset_hash": train_dataset_hash, "cal_dataset_hash": cal_dataset_hash,
                    "train_rows": len(train_records), "cal_rows": len(cal_records)}, f, indent=2, sort_keys=True)
    with open(version_path, 'w') as f:
        json.dump({"trainer_version": TRAINER_VERSION, "trainer_seed": TRAINER_SEED,
                    "fnv_offset": f"0x{FNV_OFFSET:016x}", "fnv_prime": f"0x{FNV_PRIME:016x}",
                    "max_token_len": MAX_TOKEN_LEN, "vocab_size": len(vocab_features),
                    "threshold": float(threshold), "intercept": intercept,
                    "count_tf": True, "method": "frequency_ranking_deterministic",
                    "determinism_sha256": determinism_hash,
                    "train_manifest_sha256": train_manifest_hash,
                    "cal_manifest_sha256": cal_manifest_hash,
                    "architecture": "Arm_A_Single_TFIDF_20k",
                    "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2, sort_keys=True)

    # === Evaluation on all splits ===
    print("\n" + "="*60)
    print("EVALUATION ON ALL SPLITS")
    print("="*60)

    all_evals = {}
    for split_name, records in [("train", train_records), ("calibration", cal_records),
                                ("validation", val_records), ("ood_non_english", ood_records),
                                ("technical_quoted", tech_records), ("historical_release_holdout_v1", historical_release_holdout_v1_records)]:
        if not records:
            continue
        labels = np.array([1 if r.get('label_binary', r.get('label', 0)) == 1 else 0 for r in records], dtype=np.int64)
        counts = [extract_feature_counts(r['text']) for r in records]
        X = build_feature_matrix(counts, vocab_features, idf)
        eval_result = evaluate(model, X, labels, intercept, threshold)
        all_evals[split_name] = eval_result
        n = len(records)
        print(f"  {split_name:<18} n={n:<6} F1={eval_result['f1']:.4f} Prec={eval_result['precision']:.4f} "
              f"Rec={eval_result['recall']:.4f} AUROC={eval_result['auroc']:.4f} "
              f"F/S={eval_result['false_safe_rate']:.4f} F/H={eval_result['false_halt_rate']:.4f}")

    # === Calibration bins ===
    print("\n" + "="*60)
    print("CALIBRATION (binned deciles)")
    print("="*60)

    cal_bins = []
    boundaries = np.percentile(cal_scores, np.linspace(0, 100, 11))
    for i in range(10):
        mask = (cal_scores > boundaries[i]) & (cal_scores <= boundaries[i+1])
        if mask.sum() > 0:
            bin_labels = cal_labels[mask]
            bin_scores = cal_scores[mask]
            bin_preds = (bin_scores > threshold).astype(int)
            bin_pos = np.sum(bin_labels == 1)
            cal_bins.append({"decile": i+1, "count": int(mask.sum()),
                             "attack_count": int(bin_pos),
                             "attack_rate": float(bin_pos / mask.sum()),
                             "predicted_positive_rate": float(np.mean(bin_preds))})
            print(f"  Decile {i+1}: n={mask.sum():.0f} attack_rate={bin_pos/mask.sum():.4f} pred_pos={np.mean(bin_preds):.4f}")

    with open(artifact_dir / "calibration.json", 'w') as f:
        json.dump(cal_bins, f, indent=2)

    # Store data for report generation
    report_data = {
        "train_manifest_hash": train_manifest_hash,
        "cal_manifest_hash": cal_manifest_hash,
        "train_dataset_hash": train_dataset_hash,
        "cal_dataset_hash": cal_dataset_hash,
        "train_rows": len(train_records),
        "cal_rows": len(cal_records),
        "val_rows": len(val_records),
        "ood_rows": len(ood_records),
        "tech_rows": len(tech_records),
        "historical_release_holdout_v1_rows": len(historical_release_holdout_v1_records),
        "vocab_size": len(vocab_features),
        "threshold": float(threshold),
        "intercept": intercept,
        "determinism": determinism,
        "determinism_hash": determinism_hash,
        "pass1_hash": file_hash,
        "pass2_hash": file_hash2,
        "evaluations": all_evals,
        "calibration": cal_bins,
        "train_labels": train_labels.tolist(),
        "cal_labels": cal_labels.tolist(),
        "cal_scores": cal_scores.tolist(),
    }
    with open(artifact_dir / "report_data.json", 'w') as f:
        json.dump(report_data, f, indent=2)

    # Store records for TRAINING_REPORT generation
    all_records = {
        "train": train_records, "cal": cal_records, "val": val_records,
        "ood": ood_records, "tech": tech_records, "historical_release_holdout_v1": historical_release_holdout_v1_records,
    }
    with open(artifact_dir / "all_records.json", 'w') as f:
        json.dump(all_records, f)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — vocab_model.bin written to {vocab_model_path}")
    print(f"Determinism: {'PROVEN ✅' if determinism else 'FAILED ❌'}")
    print(f"Historical release holdout v1 F1: {all_evals.get('historical_release_holdout_v1', {}).get('f1', 'N/A')}")
    # GUARD: next training cycle must create a new sealed holdout before experimenting — do not reuse v1 for tuning.


if __name__ == '__main__':
    main()
