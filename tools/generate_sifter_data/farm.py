#!/usr/bin/env python3
"""Fast serial dual-pool classifier training data generator."""
import sys, json, time
sys.path.insert(0, '.')

from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.llm_client import get_pool, LLMResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt
from tools.generate_sifter_data.validator import validate_text
from tools.generate_sifter_data.filter import evaluate_sample
from tools.generate_sifter_data.schemas import ClassifierTrainingSample
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.config import DATA_DIR, ALL_CATEGORIES

def parse_json_response(content):
    c = content.strip()
    if c.startswith("```"):
        c = c.split("\n", 1)[-1] if "\n" in c else c[3:]
        if c.endswith("```"):
            c = c[:-3]
    try:
        start = c.find("{")
        end = c.rfind("}")
        if start >= 0 and end > start:
            return json.loads(c[start:end+1])
    except:
        pass
    return None

def generate_nv(nv, prompt, cat, tier):
    r = nv.chat(prompt, max_tokens=300)
    if not isinstance(r, NvResponse):
        return None
    return r.content

def generate_ds(ds, prompt, cat, tier):
    r = ds.chat(prompt, model="fast", temperature=0.8, max_tokens=300)
    if not isinstance(r, LLMResponse):
        return None
    return r.content

def process(content, cat, tier):
    p = parse_json_response(content)
    if not p:
        return None
    t = p.get("text", "")
    if not t or len(t) < 10:
        return None
    s = int(p.get("manipulation_score", 50))
    ism = bool(p.get("is_manipulation", s > 50))
    v = validate_text(t)
    d = evaluate_sample(label_score=s, label_is_manip=ism, validation=v,
                        category=cat, tier=tier, product=2)
    if d.accepted:
        return ClassifierTrainingSample(
            text=t, category=cat, tier=tier, tier_label=f"tier{tier}",
            manipulation_score=s, is_manipulation=ism, keywords_triggered=[],
            classifier_prob=round(v.classifier_prob, 4),
            classifier_is_manipulation=v.classifier_is_manipulation,
            bias_breakdown=v.bias_breakdown,
            signal_density=round(d.signal_density, 2),
            gap_manipulation_score=round(d.gap_score, 2),
            generation_attempt=1)
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=30)
    ap.add_argument("--cats", type=str, help="comma-separated categories")
    args = ap.parse_args()

    nv = get_nvpool()
    ds = get_pool()
    cats = args.cats.split(",") if args.cats else ALL_CATEGORIES
    tiers = [1, 2, 3]

    total = len(cats) * len(tiers) * args.per_tier
    print(f"FARM: {len(cats)} cats x {len(tiers)} tiers x {args.per_tier} = ~{total} target")
    print(f"  DeepSeek: {len(ds.keys)} keys | NVIDIA: {len(nv.keys)} keys")

    all_samples = []
    for cat in cats:
        for tier in tiers:
            target = args.per_tier
            prompt = build_generate_prompt(cat, tier)
            acc = 0
            attempts = 0
            t0 = time.time()

            while acc < target and attempts < target * 5:
                # Alternate between NVIDIA and DeepSeek each attempt
                if attempts % 2 == 0:
                    content = generate_nv(nv, prompt, cat, tier)
                else:
                    content = generate_ds(ds, prompt, cat, tier)
                attempts += 1

                if content is None:
                    time.sleep(0.2)
                    continue

                sample = process(content, cat, tier)
                if sample:
                    all_samples.append(sample)
                    acc += 1

            elapsed = time.time() - t0
            rate = acc / elapsed if elapsed > 0 else 0
            print(f"  [{cat}/tier{tier}] {acc}/{target} ({rate:.1f}/s) | total={len(all_samples)}")

            if all_samples:
                write_jsonl(all_samples, DATA_DIR / "classifier_training_farm.jsonl", mode="w")

    if all_samples:
        avg_sd = sum(s.signal_density for s in all_samples) / len(all_samples)
        print(f"\nDONE: {len(all_samples)} samples, avg SD={avg_sd:.2f}")
    nv.print_stats()
    ds.print_stats()

if __name__ == "__main__":
    main()
