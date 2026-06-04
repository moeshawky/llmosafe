#!/usr/bin/env python3
"""Hammer mode — DeepSeek + NVIDIA pools, all keys, max throughput."""

import sys, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.generate_sifter_data.llm_client import get_pool, LLMResponse
from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt
from tools.generate_sifter_data.validator import validate_text
from tools.generate_sifter_data.filter import evaluate_sample
from tools.generate_sifter_data.schemas import ClassifierTrainingSample
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.config import (
    ALL_CATEGORIES, SAMPLES_PER_TIER, TIER_LABELS, DATA_DIR, SD_MIN_ACCEPT
)


def _parse_json(content: str) -> dict | None:
    try:
        # Strip markdown code fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _process(text: str, label_score: int, label_is_manip: bool,
             category: str, tier: int, tier_label: str) -> ClassifierTrainingSample | None:
    validation = validate_text(text)
    gap = abs(label_score / 100.0 - validation.classifier_prob)
    decision = evaluate_sample(
        label_score=label_score, label_is_manip=label_is_manip,
        validation=validation, category=category, tier=tier, product=2,
    )
    if decision.accepted:
        return ClassifierTrainingSample(
            text=text, category=category, tier=tier, tier_label=tier_label,
            manipulation_score=label_score, is_manipulation=label_is_manip,
            keywords_triggered=[],
            classifier_prob=round(validation.classifier_prob, 4),
            classifier_is_manipulation=validation.classifier_is_manipulation,
            bias_breakdown=validation.bias_breakdown,
            signal_density=round(decision.signal_density, 2),
            gap_manipulation_score=round(gap, 2),
            generation_attempt=1,
        )
    return None


def hammer(categories: list[str] | None = None,
           tiers: list[int] | None = None,
           per_tier: int | None = None):
    ds = get_pool()
    nv = get_nvpool()
    cats = categories or ALL_CATEGORIES
    ts = tiers or [1, 2, 3]  # skip tiers 4-5 (adversarial/false-positive) for speed

    all_samples: list[ClassifierTrainingSample] = []
    total_target = sum(per_tier or SAMPLES_PER_TIER.get(t, 30) for _ in cats for t in ts)
    print(f"HAMMER: {len(cats)} cats × {len(ts)} tiers × ~{per_tier or 30}/tier = ~{total_target} target")
    print(f"  DeepSeek: {len(ds.keys)} keys | NVIDIA: {len(nv.keys)} keys | Total: {len(ds.keys) + len(nv.keys)}")

    output = DATA_DIR / "classifier_training_hammer.jsonl"
    if all_samples:
        write_jsonl(all_samples, output, mode="w")

    output = DATA_DIR / "classifier_training_hammer.jsonl"

    for category in cats:
        for tier in ts:
            count = per_tier or SAMPLES_PER_TIER.get(tier, 30)
            tier_label = TIER_LABELS.get(tier, f"tier_{tier}")
            prompt = build_generate_prompt(category, tier)
            accepted = 0
            attempts = 0
            max_attempts = count * 5

            print(f"\n[{category}/tier{tier} ({tier_label})] target: {count}")

            while accepted < count and attempts < max_attempts:
                batch = min(15, (count - accepted) * 2 + 5)
                ds_n = batch * 4 // 7  # ~60% DeepSeek, 40% NVIDIA
                nv_n = batch - ds_n
                attempts += batch

                # Fire both pools simultaneously
                results: list[ClassifierTrainingSample | None] = [None] * batch
                lock = threading.Lock()

                def ds_worker(i, prompt_):
                    for _ in range(3):
                        r = ds.chat(prompt_, model="fast", temperature=0.8, max_tokens=300)
                        if isinstance(r, LLMResponse):
                            parsed = _parse_json(r.content)
                            if parsed:
                                text_ = parsed.get("text", "")
                                if text_ and len(text_) >= 10:
                                    s = _process(text_, int(parsed.get("manipulation_score", 50)),
                                                 bool(parsed.get("is_manipulation", True)),
                                                 category, tier, tier_label)
                                    if s:
                                        with lock: results[i] = s
                                        break
                            break
                        time.sleep(0.3)

                def nv_worker(i, prompt_):
                    for _ in range(3):
                        r = nv.chat(prompt_, temperature=0.8, max_tokens=300)
                        if isinstance(r, NvResponse):
                            parsed = _parse_json(r.content)
                            if parsed:
                                text_ = parsed.get("text", "")
                                if text_ and len(text_) >= 10:
                                    s = _process(text_, int(parsed.get("manipulation_score", 50)),
                                                 bool(parsed.get("is_manipulation", True)),
                                                 category, tier, tier_label)
                                    if s:
                                        with lock: results[i] = s
                                        break
                            break
                        time.sleep(0.3)

                with ThreadPoolExecutor(max_workers=min(batch, 25)) as ex:
                    futures = []
                    for i in range(ds_n):
                        futures.append(ex.submit(ds_worker, i, prompt))
                    for i in range(ds_n, ds_n + nv_n):
                        futures.append(ex.submit(nv_worker, i, prompt))
                    for f in as_completed(futures):
                        f.result()

                for s in results:
                    if s is not None:
                        all_samples.append(s)
                        accepted += 1
                        if accepted >= count:
                            break

                if attempts % 30 == 0:
                    ds.print_stats()
                    nv.print_stats()
                    print(f"  [{category}/tier{tier}] {accepted}/{count} | total={len(all_samples)}")

            print(f"  [{category}/tier{tier}] DONE: {accepted} samples")

            # Incremental write
            if all_samples:
                write_jsonl(all_samples, output, mode="w")

    if all_samples:
        avg_sd = sum(s.signal_density for s in all_samples) / len(all_samples)
        print(f"\n=== DONE: {len(all_samples)} samples, avg SD={avg_sd:.2f} ===")
    ds.print_stats()
    nv.print_stats()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-tier", type=int, default=30)
    p.add_argument("--category", type=str)
    p.add_argument("--tier", type=int, choices=[1,2,3,4,5])
    args = p.parse_args()
    cats = [args.category] if args.category else None
    ts = [args.tier] if args.tier else None
    hammer(categories=cats, tiers=ts, per_tier=args.per_tier)


if __name__ == "__main__":
    main()
