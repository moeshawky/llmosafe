"""Unified CLI for all 4 data products.

Owns:           CLI argument parsing and product dispatch.
Depends on:     All product generators, config, llm_client.
Provides:       `python -m tools.generate_sifter_data.cli --product {1|2|3|4|all}`.
Invariants:     --dry-run prints plan without API calls.
                --output PATH overrides default output location.
"""

from __future__ import annotations

import argparse
import sys

from tools.generate_sifter_data.config import (
    PRODUCT_OUTPUTS,
    PRODUCT_SIZES,
)

PRODUCT_NAMES = {
    1: "keyword_regression",
    2: "classifier_training",
    3: "adversarial_evasion",
    4: "hard_negatives",
}


def cmd_product1(args: argparse.Namespace) -> None:
    from tools.generate_sifter_data.products.keyword_regression import main as p1_main

    sys.argv = [
        "keyword_regression",
        f"--output={args.output or PRODUCT_OUTPUTS[1]}",
    ]
    if args.category and args.category != "all":
        sys.argv.append(f"--category={args.category}")
    if args.dry_run:
        sys.argv.append("--dry-run")
    p1_main()


def cmd_product2(args: argparse.Namespace) -> None:
    from tools.generate_sifter_data.products.classifier_training import main as p2_main

    sys.argv = [
        "classifier_training",
        f"--output={args.output or PRODUCT_OUTPUTS[2]}",
    ]
    if args.category and args.category != "all":
        sys.argv.append(f"--category={args.category}")
    if args.tier:
        sys.argv.append(f"--tier={args.tier}")
    if args.count:
        sys.argv.append(f"--count={args.count}")
    if args.dry_run:
        sys.argv.append("--dry-run")
    p2_main()


def cmd_product3(args: argparse.Namespace) -> None:
    from tools.generate_sifter_data.products.adversarial_evasion import main as p3_main

    sys.argv = [
        "adversarial_evasion",
        f"--output={args.output or PRODUCT_OUTPUTS[3]}",
    ]
    if args.category and args.category != "all":
        sys.argv.append(f"--category={args.category}")
    if args.count:
        sys.argv.append(f"--count={args.count}")
    if args.dry_run:
        sys.argv.append("--dry-run")
    p3_main()


def cmd_product4(args: argparse.Namespace) -> None:
    from tools.generate_sifter_data.products.hard_negatives import main as p4_main

    sys.argv = [
        "hard_negatives",
        f"--output={args.output or PRODUCT_OUTPUTS[4]}",
    ]
    if args.category and args.category != "all":
        sys.argv.append(f"--category={args.category}")
    if args.count:
        sys.argv.append(f"--count={args.count}")
    if args.dry_run:
        sys.argv.append("--dry-run")
    p4_main()


def cmd_all(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("DATA GENERATION FARM — All 4 Products")
    print("=" * 60)

    if args.dry_run:
        for pid in [1, 2, 3, 4]:
            print(
                f"\nProduct {pid} ({PRODUCT_NAMES[pid]}): target ~{PRODUCT_SIZES[pid]} samples"
            )
            print(f"  Output: {args.output or PRODUCT_OUTPUTS[pid]}")
        print("\nDry run complete — no API calls made.")
        return

    from tools.generate_sifter_data.llm_client import get_pool

    for pid in [1, 2, 3, 4]:
        print(f"\n{'=' * 60}")
        print(f"PRODUCT {pid}: {PRODUCT_NAMES[pid]}")
        print(f"{'=' * 60}")

        if pid == 1:
            cmd_product1(args)
        elif pid == 2:
            cmd_product2(args)
        elif pid == 3:
            cmd_product3(args)
        elif pid == 4:
            cmd_product4(args)

    print(f"\n{'=' * 60}")
    print("ALL PRODUCTS COMPLETE")
    print(f"{'=' * 60}")

    # Print final pool stats if LLM was used
    try:
        pool = get_pool()
        pool.print_stats()
    except ValueError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sifter Data Generation Farm — yield 4 training data products",
    )
    parser.add_argument(
        "--product",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Product to generate (default: all)",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Single category to generate (default: all categories)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Single tier for Product 2 (default: all tiers)",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Override sample count per category/tier",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL path (overrides default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without generating or making API calls",
    )

    args = parser.parse_args()

    dispatch = {
        "1": cmd_product1,
        "2": cmd_product2,
        "3": cmd_product3,
        "4": cmd_product4,
        "all": cmd_all,
    }
    dispatch[args.product](args)


if __name__ == "__main__":
    main()
