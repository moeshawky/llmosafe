#!/usr/bin/env python3
"""Download HF datasets for llmosafe training data."""
import json, os, sys

def get_field(row, *keys):
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

os.makedirs("data", exist_ok=True)
data = []

# 1. ShieldLM
print("Loading shieldlm-prompt-injection...", flush=True)
try:
    from datasets import load_dataset
    shield = load_dataset("dmilush/shieldlm-prompt-injection", split="train", trust_remote_code=True)
    print(f"  {len(shield)} rows", flush=True)
    for row in shield:
        text = str(row.get("text", "")).strip()
        if not text or len(text) < 5:
            continue
        label = int(row.get("label_binary", -1))
        if label not in (0, 1):
            continue
        data.append({"text": text[:500], "label": label, "source": "shieldlm", "method": "huggingface"})
    print(f"  Accepted: {sum(1 for d in data if d['source']=='shieldlm')}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

# 2. neuralchemy
print("Loading neuralchemy/Prompt-injection-dataset...", flush=True)
try:
    nc = load_dataset("neuralchemy/Prompt-injection-dataset", "core", split="train", trust_remote_code=True)
    print(f"  {len(nc)} rows", flush=True)
    for row in nc:
        text = get_field(row, "prompt", "text", "input", "message")
        if not text or len(text) < 5:
            continue
        label_raw = get_field(row, "label", "malicious", "is_injection", "type")
        if label_raw is None:
            continue
        label_raw = str(label_raw).lower()
        if label_raw in ("1", "true", "malicious", "harmful", "injection", "yes"):
            label = 1
        elif label_raw in ("0", "false", "benign", "safe", "clean", "no"):
            label = 0
        else:
            continue
        data.append({"text": text[:500], "label": label, "source": "neuralchemy", "method": "huggingface"})
    print(f"  Accepted: {sum(1 for d in data if d['source']=='neuralchemy')}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

# 3. deepset
print("Loading deepset/prompt-injections...", flush=True)
try:
    ds = load_dataset("deepset/prompt-injections", split="train", trust_remote_code=True)
    print(f"  {len(ds)} rows", flush=True)
    for row in ds:
        text = get_field(row, "text", "prompt", "input")
        if not text or len(text) < 5:
            continue
        label_raw = get_field(row, "label", "is_injection")
        if label_raw is None:
            continue
        label_raw = str(label_raw).lower()
        label = 1 if label_raw in ("1", "true", "yes", "injection") else 0
        data.append({"text": text[:500], "label": label, "source": "deepset", "method": "huggingface"})
    print(f"  Accepted: {sum(1 for d in data if d['source']=='deepset')}", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

# Save
with open("data/corpus_huggingface.jsonl", "w") as f:
    for d in data:
        json.dump(d, f)
        f.write("\n")

manip = sum(1 for d in data if d["label"] == 1)
legit = sum(1 for d in data if d["label"] == 0)
print(f"\nTotal: {len(data)} ({manip} manip / {legit} legit)", flush=True)

if len(data) < 100:
    print("ERROR: Too few samples. Check dataset loading errors above.", flush=True)
    sys.exit(1)
