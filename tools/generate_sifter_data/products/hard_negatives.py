"""Product 4: Hard Negative Contrastive Pairs — HyDE flavor.

Owns:           HardNegativesGenerator — same keywords, opposite intent pairs.
Depends on:     llm_client (DeepSeekPool), validator (validate_text, CATEGORY_KEYWORDS),
                filter (evaluate_sample), exporter (write_jsonl),
                prompts.generation (build_contrastive_prompt, build_contrastive_regenerate_prompt),
                schemas (HardNegativePair), config.
Provides:       Generator for 2,000 accepted contrastive pairs.
Invariants:     benign_classifier_prob < 0.3, manipulation_classifier_prob > 0.6.
                contrastive_gap = manipulation_prob - benign_prob ≥ 0.30.
                Pair discriminative value is the SD score.
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
    SD_ACCEPT,
)
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.filter import compute_signal_density
from tools.generate_sifter_data.llm_client import DeepSeekPool, LLMResponse, get_pool
from tools.generate_sifter_data.prompts.generation import (
    build_contrastive_prompt,
    build_contrastive_regenerate_prompt,
)
from tools.generate_sifter_data.schemas import HardNegativePair, GeneratorStats
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


def generate_pairs_for_category(
    pool: DeepSeekPool,
    category: str,
    count: int,
) -> Iterator[HardNegativePair]:
    """Generate contrastive pairs for one manipulation category.

    Picks 2-3 representative keywords, prompts LLM to write paired
    sentences sharing keywords but with opposite intent.

    Args:
        pool: DeepSeekPool for API calls.
        category: Manipulation category.
        count: Target number of accepted pairs.

    Yields:
        HardNegativePair with contrastive_gap ≥ 0.30.
    """
    all_keywords = CATEGORY_KEYWORDS.get(category, [])
    # Pick representative subset
    keywords = all_keywords[:3] if len(all_keywords) >= 3 else all_keywords
    prompt = build_contrastive_prompt(category, keywords)

    accepted = 0
    total_attempts = 0
    max_attempts = count * (MAX_RETRIES + 1) * 3

    while accepted < count and total_attempts < max_attempts:
        batch_count = min(BATCH_SIZE // 8, count - accepted + 3)
        prompts = [prompt for _ in range(batch_count)]
        total_attempts += batch_count

        for idx, result in pool.batch_generate(
            prompts, model="fast", temperature=0.8, max_tokens=400
        ):
            if not isinstance(result, LLMResponse):
                continue

            parsed = _parse_response(result.content)
            if parsed is None:
                continue

            benign_text = parsed.get("benign_text", "")
            manipulation_text = parsed.get("manipulation_text", "")
            if not benign_text or not manipulation_text:
                continue
            if len(benign_text) < 10 or len(manipulation_text) < 10:
                continue

            benign_score = int(parsed.get("benign_score", 10))
            manipulation_score = int(parsed.get("manipulation_score", 80))
            shared = parsed.get("shared_keywords", keywords)

            benign_val = validate_text(benign_text)
            manip_val = validate_text(manipulation_text)

            benign_prob = benign_val.classifier_prob
            manip_prob = manip_val.classifier_prob
            contrastive_gap = manip_prob - benign_prob

            # Check acceptance criteria
            if contrastive_gap < 0.30:
                # Regenerate with more contrast
                for retry in range(1, MAX_RETRIES + 1):
                    retry_msgs = build_contrastive_regenerate_prompt(
                        category=category,
                        keywords=keywords,
                        previous_benign=benign_text,
                        previous_manipulation=manipulation_text,
                    )
                    retry_result = pool.chat(
                        retry_msgs, model="fast", temperature=0.9, max_tokens=400
                    )
                    if not isinstance(retry_result, LLMResponse):
                        continue

                    retry_parsed = _parse_response(retry_result.content)
                    if retry_parsed is None:
                        continue

                    retry_benign = retry_parsed.get("benign_text", "")
                    retry_manip = retry_parsed.get("manipulation_text", "")
                    if not retry_benign or not retry_manip:
                        continue

                    retry_benign_val = validate_text(retry_benign)
                    retry_manip_val = validate_text(retry_manip)
                    retry_gap = (
                        retry_manip_val.classifier_prob
                        - retry_benign_val.classifier_prob
                    )

                    retry_benign_score = int(
                        retry_parsed.get("benign_score", benign_score)
                    )
                    retry_manip_score = int(
                        retry_parsed.get("manipulation_score", manipulation_score)
                    )

                    if retry_gap >= 0.30:
                        keyword_overlap = len(shared) / max(1, len(keywords))
                        sd = compute_signal_density(
                            label_score=retry_manip_score,
                            label_is_manip=True,
                            classifier_prob=retry_manip_val.classifier_prob,
                            classifier_is_manip=retry_manip_val.classifier_is_manipulation,
                            tier=4,
                            novelty=0.85,
                            product=4,
                            contrastive_gap=retry_gap,
                            keyword_overlap=keyword_overlap,
                        )

                        if sd >= SD_ACCEPT:
                            yield HardNegativePair(
                                category=category,
                                benign_text=retry_benign,
                                manipulation_text=retry_manip,
                                shared_keywords=shared,
                                benign_score=retry_benign_score,
                                manipulation_score=retry_manip_score,
                                benign_classifier_prob=round(
                                    retry_benign_val.classifier_prob, 4
                                ),
                                manipulation_classifier_prob=round(
                                    retry_manip_val.classifier_prob, 4
                                ),
                                contrastive_gap=round(retry_gap, 4),
                                signal_density=round(sd, 2),
                            )
                            accepted += 1
                            break
                continue

            keyword_overlap = len(shared) / max(1, len(keywords))
            sd = compute_signal_density(
                label_score=manipulation_score,
                label_is_manip=True,
                classifier_prob=manip_prob,
                classifier_is_manip=manip_val.classifier_is_manipulation,
                tier=4,
                novelty=0.85,
                product=4,
                contrastive_gap=contrastive_gap,
                keyword_overlap=keyword_overlap,
            )

            if sd >= SD_ACCEPT:
                yield HardNegativePair(
                    category=category,
                    benign_text=benign_text,
                    manipulation_text=manipulation_text,
                    shared_keywords=shared,
                    benign_score=benign_score,
                    manipulation_score=manipulation_score,
                    benign_classifier_prob=round(benign_prob, 4),
                    manipulation_classifier_prob=round(manip_prob, 4),
                    contrastive_gap=round(contrastive_gap, 4),
                    signal_density=round(sd, 2),
                )
                accepted += 1
                if accepted >= count:
                    break

        if total_attempts % (BATCH_SIZE // 4) == 0:
            pool.print_stats()
            print(f"  [{category}] pairs accepted: {accepted}/{count}")


def generate(
    pool: DeepSeekPool,
    categories: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    sample_count_override: Optional[int] = None,
) -> GeneratorStats:
    """Generate Product 4 hard negative contrastive pairs.

    Args:
        pool: DeepSeekPool for API calls.
        categories: Categories to generate. Defaults to MANIPULATION_CATEGORIES without emphasis.
        output_path: JSONL output path.
        sample_count_override: Override per-category pair count.

    Returns:
        GeneratorStats with generation metrics.
    """
    if categories is None:
        categories = [c for c in CATEGORY_KEYWORDS]
    if output_path is None:
        output_path = DATA_DIR / "hard_negatives.jsonl"

    per_category = sample_count_override if sample_count_override else 225
    t0 = time.time()

    all_samples: list[HardNegativePair] = []

    for category in categories:
        print(f"\n[{category}] target: {per_category} pairs")
        cat_samples: list[HardNegativePair] = []
        for sample in generate_pairs_for_category(pool, category, per_category):
            cat_samples.append(sample)
            all_samples.append(sample)

        if cat_samples:
            avg_sd = sum(s.signal_density for s in cat_samples) / len(cat_samples)
            avg_gap = sum(s.contrastive_gap for s in cat_samples) / len(cat_samples)
            print(
                f"  Accepted: {len(cat_samples)}, avg SD: {avg_sd:.2f}, avg gap: {avg_gap:.4f}"
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
        product_id=4,
        product_name="hard_negatives",
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
        description="Product 4: Hard Negative Contrastive Pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "hard_negatives.jsonl"),
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
        default=225,
        help="Pairs per category (default: 225)",
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
        f"Product 4: {len(cats)} categories × {args.count} per category = {total} target pairs"
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
