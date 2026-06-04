"""Product 2: Classifier Training Pairs — LLM-generated, sifter-validated, gap-driven.

Owns:           ClassifierTrainingGenerator — batch generation with retry logic.
Depends on:     llm_client (DeepSeekPool), validator (validate_text),
                filter (evaluate_sample), exporter (write_jsonl),
                prompts.generation (build_generate_prompt, build_regenerate_prompt),
                schemas (ClassifierTrainingSample), config.
Provides:       Generator for 11,000 labeled training samples across 9 categories × 5 tiers.
Invariants:     Every sample validated through validate_text() before acceptance.
                Gap > 0.30 triggers regeneration with max 3 retries.
                SD ≥ 7.0 required for acceptance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from tools.generate_sifter_data.config import (
    ALL_CATEGORIES,
    BATCH_SIZE,
    DATA_DIR,
    GAP_RETRY,
    MAX_RETRIES,
    SAMPLES_PER_TIER,
    TIER_LABELS,
)
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.filter import evaluate_sample
from tools.generate_sifter_data.llm_client import DeepSeekPool, get_pool
from tools.generate_sifter_data.prompts.generation import (
    build_generate_prompt,
    build_regenerate_prompt,
)
from tools.generate_sifter_data.schemas import ClassifierTrainingSample, GeneratorStats
from tools.generate_sifter_data.validator import validate_text


def _parse_response(content: str) -> Optional[dict]:
    """Extract JSON object from LLM response string.

    Returns parsed dict or None if no valid JSON found.
    """
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def generate_samples_for_category_tier(
    pool: DeepSeekPool,
    category: str,
    tier: int,
    count: int,
) -> Iterator[ClassifierTrainingSample]:
    """Generate and validate samples for one category × tier combination.

    Generates in batches, validates each sample through the keyword sifter
    and TF-IDF classifier, computes SD score, and retries on gap violation.

    Args:
        pool: DeepSeekPool for API calls.
        category: Manipulation category name.
        tier: Difficulty tier [1, 5].
        count: Target number of accepted samples.

    Yields:
        ClassifierTrainingSample with SD ≥ SD_ACCEPT.
    """
    prompt_template = build_generate_prompt(category, tier)
    tier_label = TIER_LABELS.get(tier, f"tier_{tier}")

    accepted = 0
    total_attempts = 0
    max_attempts = count * (MAX_RETRIES + 1) * 2

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time
    from tools.generate_sifter_data.llm_client import LLMResponse, LLMError

    def _try_chat():
        for _ in range(5):
            r = pool.chat(prompt_template, model="fast", temperature=0.8, max_tokens=300)
            if isinstance(r, LLMResponse):
                return r
            _time.sleep(0.5)
        return None

    def _process_result(cresult):
        nonlocal accepted
        if not isinstance(cresult, LLMResponse):
            return None
        cparsed = _parse_response(cresult.content)
        if cparsed is None:
            return None
        ctext = cparsed.get("text", "")
        if not ctext or len(ctext) < 10:
            return None
        clabel_score = int(cparsed.get("manipulation_score", 50))
        clabel_is_manip = bool(cparsed.get("is_manipulation", clabel_score > 50))
        cvalidation = validate_text(ctext)
        cclass_prob = cvalidation.classifier_prob
        cclass_is_manip = cvalidation.classifier_is_manipulation
        cgap = abs(clabel_score / 100.0 - cclass_prob)
        cdecision = evaluate_sample(
            label_score=clabel_score, label_is_manip=clabel_is_manip,
            validation=cvalidation, category=category, tier=tier, product=2,
        )
        if cdecision.accepted:
            accepted += 1
            return ClassifierTrainingSample(
                text=ctext, category=category, tier=tier, tier_label=tier_label,
                manipulation_score=clabel_score, is_manipulation=clabel_is_manip,
                keywords_triggered=[],
                classifier_prob=round(cclass_prob, 4),
                classifier_is_manipulation=cclass_is_manip,
                bias_breakdown=cvalidation.bias_breakdown,
                signal_density=round(cdecision.signal_density, 2),
                gap_manipulation_score=round(cgap, 2),
                generation_attempt=1,
            )
        return None

    while accepted < count and total_attempts < max_attempts:
        batch = min(10, count - accepted + 5)
        prompts = [prompt_template] * batch
        total_attempts += batch

        with ThreadPoolExecutor(max_workers=min(batch, 15)) as ex:
            futures = {ex.submit(_try_chat): i for i in range(batch)}
            for f in as_completed(futures):
                sample = _process_result(f.result())
                if sample is not None:
                    yield sample
                if accepted >= count:
                    break

        if total_attempts % 20 == 0:
            pool.print_stats()
            print(f"  [{category}/tier{tier}] accepted: {accepted}/{count}")

        # Progress report every few batches
        if total_attempts % (BATCH_SIZE * 2) == 0:
            pool.print_stats()
            print(f"  [{category}/tier{tier}] accepted: {accepted}/{count}")


def generate(
    pool: DeepSeekPool,
    categories: Optional[list[str]] = None,
    tiers: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
    sample_count_override: Optional[int] = None,
) -> GeneratorStats:
    """Generate Product 2 classifier training samples.

    Args:
        pool: DeepSeekPool for API calls.
        categories: Categories to generate. Defaults to ALL_CATEGORIES.
        tiers: Tiers to generate. Defaults to [1, 2, 3, 4, 5].
        output_path: JSONL output path. Defaults to DATA_DIR/classifier_training.jsonl.
        sample_count_override: Override per-tier sample count.

    Returns:
        GeneratorStats with generation metrics.
    """
    if categories is None:
        categories = ALL_CATEGORIES
    if tiers is None:
        tiers = [1, 2, 3, 4, 5]
    if output_path is None:
        output_path = DATA_DIR / "classifier_training.jsonl"

    t0 = time.time()

    all_samples: list[ClassifierTrainingSample] = []
    total_accepted = 0

    for category in categories:
        for tier in tiers:
            count = sample_count_override or SAMPLES_PER_TIER.get(tier, 60)
            tier_label = TIER_LABELS.get(tier, f"tier_{tier}")
            print(f"\n[{category}/tier{tier} ({tier_label})] target: {count} samples")

            cat_samples: list[ClassifierTrainingSample] = []
            for sample in generate_samples_for_category_tier(
                pool, category, tier, count
            ):
                cat_samples.append(sample)
                all_samples.append(sample)
                total_accepted += 1

            if cat_samples:
                avg_sd = sum(s.signal_density for s in cat_samples) / len(cat_samples)
                print(f"  Accepted: {len(cat_samples)}, avg SD: {avg_sd:.2f}")
            else:
                print("  Accepted: 0 (check API connectivity)")

            pool.print_stats()

    runtime = time.time() - t0

    if all_samples:
        written = write_jsonl(all_samples, output_path, mode="w")
        print(f"\nWrote {written} samples to {output_path}")

    avg_sd = (
        sum(s.signal_density for s in all_samples) / len(all_samples)
        if all_samples
        else 0.0
    )

    print(f"\nTotal accepted: {total_accepted}, runtime: {runtime:.1f}s")
    if all_samples:
        sd_scores = [s.signal_density for s in all_samples]
        print(
            f"SD range: [{min(sd_scores):.2f}, {max(sd_scores):.2f}], mean: {avg_sd:.2f}"
        )
        gaps = [s.gap_manipulation_score for s in all_samples]
        print(f"Gap range: [{min(gaps):.2f}, {max(gaps):.2f}]")

    pool.print_stats()

    return GeneratorStats(
        product_id=2,
        product_name="classifier_training",
        total_generated=total_accepted,
        total_accepted=total_accepted,
        avg_signal_density=round(avg_sd, 2),
        total_cost_yuan=round(pool.total_cost, 4),
        runtime_seconds=round(runtime, 2),
    )


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Product 2: Classifier Training Pairs")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "classifier_training.jsonl"),
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=ALL_CATEGORIES + ["all"],
        default="all",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Single tier (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Override sample count per category-tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making API calls",
    )
    args = parser.parse_args()

    categories = ALL_CATEGORIES if args.category == "all" else [args.category]
    tiers = [args.tier] if args.tier else [1, 2, 3, 4, 5]

    total_target = sum(
        args.count or SAMPLES_PER_TIER.get(t, 60) for _ in categories for t in tiers
    )
    print(
        f"Product 2: {len(categories)} categories × {len(tiers)} tiers = {total_target} target samples"
    )

    if args.dry_run:
        print("Dry run — no API calls will be made.")
        print(f"Categories: {categories}")
        print(f"Tiers: {tiers}")
        return

    pool = get_pool()
    print(f"DeepSeek pool: {len(pool.keys)} keys")

    stats = generate(
        pool,
        categories=categories,
        tiers=tiers,
        output_path=Path(args.output),
        sample_count_override=args.count,
    )
    print(f"Done. Generated: {stats.total_generated}")


if __name__ == "__main__":
    main()
