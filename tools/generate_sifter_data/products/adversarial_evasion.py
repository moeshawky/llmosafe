"""Product 3: Adversarial Evasion Examples — TRM flavor.

Owns:           AdversarialEvasionGenerator — text labeled manipulative but
                keyword sifter reports zero bias.
Depends on:     llm_client (DeepSeekPool), validator (validate_text, CATEGORY_KEYWORDS),
                filter (evaluate_sample), exporter (write_jsonl),
                prompts.generation (build_evasion_prompt, build_evasion_regenerate_prompt),
                schemas (AdversarialSample), config.
Provides:       Generator for 1,650 accepted evasion samples.
Invariants:     keyword_sifter_says_safe must be True (total bias ≤ 50).
                LLM label must be is_manipulation=True, score ≥ 60.
                Gap IS the training signal — evasion SD formula used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from tools.generate_sifter_data.config import (
    BATCH_SIZE,
    DATA_DIR,
    MAX_RETRIES,
    MANIPULATION_CATEGORIES,
    SD_ACCEPT,
)
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.filter import compute_signal_density
from tools.generate_sifter_data.llm_client import DeepSeekPool, LLMResponse, get_pool
from tools.generate_sifter_data.prompts.generation import (
    build_evasion_prompt,
    build_evasion_regenerate_prompt,
)
from tools.generate_sifter_data.schemas import AdversarialSample, GeneratorStats
from tools.generate_sifter_data.validator import (
    CATEGORY_KEYWORDS,
    validate_text,
)


def _parse_response(content: str) -> Optional[dict]:
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def generate_evasion_for_category(
    pool: DeepSeekPool,
    category: str,
    count: int,
) -> Iterator[AdversarialSample]:
    """Generate evasive samples for one manipulation category.

    Prompts the LLM to write manipulative text using sophisticated vocabulary
    that evades keyword-based detection. Rejects samples where keyword sifter
    still fires (total > 50).

    Args:
        pool: DeepSeekPool for API calls.
        category: Manipulation category to mimic.
        count: Target number of accepted samples.

    Yields:
        AdversarialSample with keyword_sifter_says_safe=True,
        is_manipulation=True, manipulation_score ≥ 60.
    """
    keyword_list = CATEGORY_KEYWORDS.get(category, [])
    prompt = build_evasion_prompt(category, keyword_list)

    accepted = 0
    total_attempts = 0
    max_attempts = count * (MAX_RETRIES + 1) * 3

    while accepted < count and total_attempts < max_attempts:
        batch_count = min(BATCH_SIZE // 4, count - accepted + 5)
        prompts = [prompt for _ in range(batch_count)]
        total_attempts += batch_count

        for idx, result in pool.batch_generate(
            prompts, model="fast", temperature=0.9, max_tokens=400
        ):
            if not isinstance(result, LLMResponse):
                continue

            parsed = _parse_response(result.content)
            if parsed is None:
                continue

            text = parsed.get("text", "")
            if not text or len(text) < 10:
                continue

            label_score = int(parsed.get("manipulation_score", 70))
            label_is_manip = bool(parsed.get("is_manipulation", True))
            evaded = parsed.get("evaded_keywords", [])

            validation = validate_text(text)
            sifter_total = validation.total_bias_score
            sifter_safe = sifter_total <= 50

            if not sifter_safe:
                # Retry — keyword sifter detected bias, need more evasion
                for retry in range(1, MAX_RETRIES + 1):
                    retry_msgs = build_evasion_regenerate_prompt(
                        category=category,
                        keyword_list=keyword_list,
                        previous_text=text,
                    )
                    retry_result = pool.chat(
                        retry_msgs, model="fast", temperature=0.95, max_tokens=400
                    )
                    if not isinstance(retry_result, LLMResponse):
                        continue

                    retry_parsed = _parse_response(retry_result.content)
                    if retry_parsed is None:
                        continue

                    retry_text = retry_parsed.get("text", "")
                    if not retry_text or len(retry_text) < 10:
                        continue

                    retry_label_score = int(
                        retry_parsed.get("manipulation_score", label_score)
                    )
                    retry_is_manip = bool(retry_parsed.get("is_manipulation", True))
                    retry_validation = validate_text(retry_text)
                    retry_sifter_total = retry_validation.total_bias_score

                    if retry_sifter_total <= 50 and retry_label_score >= 60:
                        retry_prob = retry_validation.classifier_prob
                        retry_is_manip_class = (
                            retry_validation.classifier_is_manipulation
                        )

                        sd = compute_signal_density(
                            label_score=retry_label_score,
                            label_is_manip=retry_is_manip,
                            classifier_prob=retry_prob,
                            classifier_is_manip=retry_is_manip_class,
                            tier=4,
                            novelty=0.90,
                            product=3,
                        )

                        if sd >= SD_ACCEPT:
                            yield AdversarialSample(
                                text=retry_text,
                                target_category=category,
                                evaded_keywords=retry_parsed.get(
                                    "evaded_keywords", evaded
                                ),
                                manipulation_score=retry_label_score,
                                is_manipulation=retry_is_manip,
                                keyword_sifter_says_safe=True,
                                keyword_sifter_total=retry_sifter_total,
                                classifier_prob=round(retry_prob, 4),
                                classifier_is_manipulation=retry_is_manip_class,
                                signal_density=round(sd, 2),
                                generation_attempt=retry + 1,
                            )
                            accepted += 1
                            break
                continue

            class_prob = validation.classifier_prob
            class_is_manip = validation.classifier_is_manipulation

            sd = compute_signal_density(
                label_score=label_score,
                label_is_manip=label_is_manip,
                classifier_prob=class_prob,
                classifier_is_manip=class_is_manip,
                tier=4,
                novelty=0.90,
                product=3,
            )

            if sd >= SD_ACCEPT and label_score >= 60:
                yield AdversarialSample(
                    text=text,
                    target_category=category,
                    evaded_keywords=evaded,
                    manipulation_score=label_score,
                    is_manipulation=label_is_manip,
                    keyword_sifter_says_safe=True,
                    keyword_sifter_total=sifter_total,
                    classifier_prob=round(class_prob, 4),
                    classifier_is_manipulation=class_is_manip,
                    signal_density=round(sd, 2),
                    generation_attempt=1,
                )
                accepted += 1
                if accepted >= count:
                    break

        # Progress
        if total_attempts % (BATCH_SIZE // 2) == 0:
            pool.print_stats()
            print(f"  [{category}] accepted: {accepted}/{count}")


def generate(
    pool: DeepSeekPool,
    categories: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    sample_count_override: Optional[int] = None,
) -> GeneratorStats:
    """Generate Product 3 adversarial evasion samples.

    Args:
        pool: DeepSeekPool for API calls.
        categories: Categories to generate. Defaults to MANIPULATION_CATEGORIES.
        output_path: JSONL output path.
        sample_count_override: Override per-category sample count.

    Returns:
        GeneratorStats with generation metrics.
    """
    if categories is None:
        categories = [c for c in MANIPULATION_CATEGORIES if c != "emphasis_typographic"]
    if output_path is None:
        output_path = DATA_DIR / "adversarial_evasion.jsonl"

    per_category = sample_count_override if sample_count_override else 185
    t0 = time.time()

    all_samples: list[AdversarialSample] = []

    for category in categories:
        if category not in CATEGORY_KEYWORDS:
            continue
        print(f"\n[{category}] target: {per_category} evasion samples")
        cat_samples: list[AdversarialSample] = []
        for sample in generate_evasion_for_category(pool, category, per_category):
            cat_samples.append(sample)
            all_samples.append(sample)

        if cat_samples:
            avg_sd = sum(s.signal_density for s in cat_samples) / len(cat_samples)
            avg_evasion = sum(s.keyword_sifter_total for s in cat_samples) / len(
                cat_samples
            )
            print(
                f"  Accepted: {len(cat_samples)}, avg SD: {avg_sd:.2f}, avg sifter total: {avg_evasion:.1f}"
            )
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

    print(f"\nTotal accepted: {len(all_samples)}, runtime: {runtime:.1f}s")
    pool.print_stats()

    return GeneratorStats(
        product_id=3,
        product_name="adversarial_evasion",
        total_generated=len(all_samples),
        total_accepted=len(all_samples),
        avg_signal_density=round(avg_sd, 2),
        total_cost_yuan=round(pool.total_cost, 4),
        runtime_seconds=round(runtime, 2),
    )


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Product 3: Adversarial Evasion Examples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "adversarial_evasion.jsonl"),
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[c for c in CATEGORY_KEYWORDS] + ["all"],
        default="all",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=185,
        help="Samples per category (default: 185)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making API calls",
    )
    args = parser.parse_args()

    cats = [c for c in CATEGORY_KEYWORDS] if args.category == "all" else [args.category]
    total = args.count * len(cats)
    print(
        f"Product 3: {len(cats)} categories × {args.count} per category = {total} target samples"
    )
    print(f"Categories: {cats}")

    if args.dry_run:
        print("Dry run — no API calls will be made.")
        return

    pool = get_pool()
    stats = generate(
        pool,
        categories=cats,
        output_path=Path(args.output),
        sample_count_override=args.count,
    )
    print(f"Done. Generated: {stats.total_generated}")


if __name__ == "__main__":
    main()
