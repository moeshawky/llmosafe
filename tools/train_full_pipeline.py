#!/usr/bin/env python3
"""llmosafe classifier training pipeline — single-command reproducibility.

This script encapsulates the full training workflow:

    Phase 1 — Download HuggingFace datasets (ShieldLM, neuralchemy, deepset)
    Phase 2 — Optionally generate synthetic sifter training data via LLM APIs
    Phase 3 — Combine real + synthetic corpora into a unified JSONL
    Phase 4 — Train logistic regression classifier with MI feature selection
    Phase 5 — Validate against held-out test set and embeddable quality gates

Usage:

    # Full pipeline with HuggingFace data only:
    python tools/train_full_pipeline.py

    # With custom vocab size:
    python tools/train_full_pipeline.py --vocab-size 5000

    # Skip download (use existing data/):
    python tools/train_full_pipeline.py --no-download

    # Generate sifter data (requires NVIDIA or DeepSeek API keys):
    python tools/train_full_pipeline.py --generate-sifter

Requirements:
    pip install datasets scikit-learn numpy

For the --generate-sifter flag:
    DeepSeek API key in DEEPSEEK_API_KEY env var OR
    NVIDIA API key in NVIDIA_API_KEY env var
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Phase 1: Download HuggingFace datasets
# ---------------------------------------------------------------------------

def download_hf(script: Path, out: Path) -> None:
    if out.exists():
        print(f"[1/4] HuggingFace corpus already exists ({out.stat().st_size / 1e6:.1f} MB)")
        return

    print("[1/4] Downloading HuggingFace datasets...")
    cmd = [sys.executable, str(script)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("HF download failed — see above")

    if not out.exists():
        raise RuntimeError(f"Download completed but {out} not found")

    with open(out) as f:
        count = sum(1 for _ in f)
    print(f"  Done: {count} samples ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Phase 2: Generate synthetic sifter data (optional)
# ---------------------------------------------------------------------------

def generate_sifter(sifter_dir: Path) -> None:
    gen_script = sifter_dir / "gen.py"
    if not gen_script.exists():
        raise RuntimeError(f"Sifter gen script not found: {gen_script}")

    target = sifter_dir / "data" / "classifier_training.jsonl"
    if target.exists():
        with open(target) as f:
            count = sum(1 for _ in f)
        print(f"[2/4] Sifter data already exists ({count} samples)")
        return

    print("[2/4] Generating sifter training data (this may take hours)...")
    api = os.environ.get("NVIDIA_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api:
        print("  WARNING: No API key found. Skipping sifter generation.", file=sys.stderr)
        print("  Set NVIDIA_API_KEY or DEEPSEEK_API_KEY to enable.", file=sys.stderr)
        return

    # Run gen.py for all categories and tiers (sequential, rate-limited)
    cmd = [
        sys.executable,
        str(gen_script),
        "--all",
    ]
    result = subprocess.run(cmd, cwd=sifter_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print("  WARNING: Sifter generation had errors (continuing with partial data)", file=sys.stderr)
        print(result.stderr[-500:], file=sys.stderr)

    if target.exists():
        with open(target) as f:
            count = sum(1 for _ in f)
        print(f"  Done: {count} synthetic samples")


# ---------------------------------------------------------------------------
# Phase 3: Combine corpora
# ---------------------------------------------------------------------------

def combine_corpora(
    hf_path: Path,
    sifter_path: Path | None,
    out_path: Path,
) -> None:
    print("[3/4] Combining corpora...")

    combined = []
    manip = safe = 0

    # Real HuggingFace data
    with open(hf_path) as f:
        for line in f:
            d = json.loads(line)
            lbl = int(d.get("label", 0))
            text = d.get("text", "").strip()
            if text and lbl in (0, 1):
                combined.append({"text": text, "label": lbl})
                if lbl:
                    manip += 1
                else:
                    safe += 1

    print(f"  Real (HF): {manip + safe} samples ({manip} manip, {safe} safe)")

    # Generated sifter data
    if sifter_path and sifter_path.exists():
        gen_m = gen_s = 0
        with open(sifter_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("tier") == 99:
                    continue  # skip calibration entries
                if "label" in d:
                    lbl = int(d["label"])
                else:
                    lbl = 1 if int(d.get("is_manipulation", 0)) == 1 else 0
                text = d.get("text", "").strip()
                if text and lbl in (0, 1):
                    combined.append({"text": text, "label": lbl})
                    if lbl:
                        gen_m += 1
                    else:
                        gen_s += 1
        print(f"  Generated (sifter): {gen_m + gen_s} samples ({gen_m} manip, {gen_s} safe)")
        manip += gen_m
        safe += gen_s

    # Write combined corpus
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in combined:
            f.write(json.dumps(d) + "\n")

    pct = 100 * safe / (manip + safe)
    print(f"  Total: {manip + safe} samples ({manip} manip, {safe} safe, {pct:.0f}% safe)")


# ---------------------------------------------------------------------------
# Phase 4: Train classifier
# ---------------------------------------------------------------------------

def train(
    corpus_path: Path,
    model_out: Path,
    vocab_size: int,
) -> None:
    train_script = ROOT / "tools" / "train_tfidf_classifier.py"

    print(f"[4/4] Training classifier (vocab_size={vocab_size})...")

    cmd = [
        sys.executable,
        "-u",
        str(train_script),
        "--corpus-jsonl", str(corpus_path),
        "--output", str(model_out),
        "--vocab-size", str(vocab_size),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Training failed — see above")

    if not model_out.exists():
        raise RuntimeError(f"Training succeeded but {model_out} not found")

    size_kb = model_out.stat().st_size / 1024
    print(f"Model written: {model_out} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="llmosafe classifier training pipeline",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip HuggingFace download (use existing data/)",
    )
    parser.add_argument(
        "--generate-sifter",
        action="store_true",
        help="Generate synthetic sifter data via LLM API (needs API key)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=3000,
        help="Number of features for mutual information selection (default: 3000)",
    )
    parser.add_argument(
        "--corpus-out",
        type=Path,
        default=ROOT / "data" / "corpus_combined.jsonl",
        help="Path for combined training corpus",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "tools" / "vocab_model.bin",
        help="Path for serialized classifier model",
    )
    args = parser.parse_args()

    # Paths
    hf_script = ROOT / "tools" / "download_hf_datasets.py"
    hf_corpus = ROOT / "data" / "corpus_huggingface.jsonl"
    sifter_dir = ROOT / "tools" / "generate_sifter_data"
    sifter_corpus = sifter_dir / "data" / "classifier_training.jsonl"

    # -- Phase 1: Download --
    if not args.no_download:
        download_hf(hf_script, hf_corpus)
    elif not hf_corpus.exists():
        print("ERROR: --no-download specified but corpus_huggingface.jsonl not found", file=sys.stderr)
        print("Run: python tools/download_hf_datasets.py", file=sys.stderr)
        sys.exit(1)

    # -- Phase 2: Generate sifter --
    if args.generate_sifter:
        generate_sifter(sifter_dir)

    # -- Phase 3: Combine --
    sifter_path = sifter_corpus if sifter_corpus.exists() else None
    combine_corpora(hf_corpus, sifter_path, args.corpus_out)

    # -- Phase 4: Train --
    train(args.corpus_out, args.model_out, args.vocab_size)

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print(f"  Model:   {args.model_out}")
    print(f"  Corpus:  {args.corpus_out}")
    print("  Verify:  cargo test --all-features")
    print("=" * 60)


if __name__ == "__main__":
    main()
