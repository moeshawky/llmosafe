"""Product 1: Keyword Regression Suite — deterministic test case generation.

Owns:           KeywordRegressionGenerator, variant generators, generate_all().
Depends on:     validator (compute_bias_breakdown, keyword lists, CATEGORY_KEYWORDS),
                schemas (KeywordRegressionSample), config (DATA_DIR), exporter.
Provides:       Generator that produces 5,500+ test cases with zero LLM dependency.
Invariants:     Every sample self-validated against compute_bias_breakdown().
                expected_* fields must equal recomputed values exactly.
                Zero API calls, zero network, pure deterministic computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tools.generate_sifter_data.config import DATA_DIR
from tools.generate_sifter_data.exporter import write_jsonl
from tools.generate_sifter_data.schemas import KeywordRegressionSample
from tools.generate_sifter_data.validator import (
    CATEGORY_KEYWORDS,
    compute_bias_breakdown,
)

# ── Variant generators (text templates per keyword) ──────────────────

VARIANT_GENERATORS: dict[str, callable] = {}


def _exact_match(keyword: str) -> str:
    return f"The speaker used the term {keyword} in their argument."


def _case_upper(keyword: str) -> str:
    return f"The speaker used the term {keyword.upper()} in their argument."


def _case_title(keyword: str) -> str:
    return f"The speaker used the term {keyword.title()} in their argument."


def _negated(keyword: str) -> str:
    return f"The speaker is not a {keyword} and does not claim to be one."


def _negated_ttl5(keyword: str) -> str:
    return f"not a very well known {keyword}"


def _negated_ttl7(keyword: str) -> str:
    return f"it is not the case that anyone here is a true {keyword}"


def _repeated_3x(keyword: str) -> str:
    return f"The {keyword} said {keyword} is what makes a {keyword} effective."


def _punctuation_wrapped(keyword: str) -> str:
    return f"!!!{keyword}???..."


def _embedded(keyword: str) -> str:
    return f"This text contains the letters that form {keyword} but not the word."


def _prefix_padded(keyword: str) -> str:
    return f"The speaker, who was interesting and well-prepared, is a {keyword}."


def _emphasis_caps(keyword: str) -> str:
    return f"{keyword.upper()} is what matters here. {keyword.upper()}."


def _emphasis_bonus(keyword: str) -> str:
    return f"This is {keyword.upper()} IMPORTANT CRITICAL {keyword.upper()}."


def _multi_word_phrase(keyword: str) -> str:
    if " " in keyword:
        return f"The text contains '{keyword}' as a complete phrase."
    return f"The text mentions {keyword} as a key concept."


def _edge_empty(_keyword: str) -> str:
    return ""


def _edge_whitespace(_keyword: str) -> str:
    return "   \t\n\r  "


def _edge_10k(_keyword: str) -> str:
    prefix = "normal text. " * 100
    return f"{prefix}the speaker is a {_keyword}"


VARIANT_GENERATORS = {
    "exact_match": _exact_match,
    "case_upper": _case_upper,
    "case_title": _case_title,
    "negated": _negated,
    "negated_ttl5": _negated_ttl5,
    "negated_ttl7": _negated_ttl7,
    "repeated_3x": _repeated_3x,
    "punctuation_wrapped": _punctuation_wrapped,
    "embedded": _embedded,
    "prefix_padded": _prefix_padded,
    "emphasis_caps": _emphasis_caps,
    "emphasis_bonus": _emphasis_bonus,
    "multi_word_phrase": _multi_word_phrase,
    "edge_empty": _edge_empty,
    "edge_whitespace": _edge_whitespace,
    "edge_10k": _edge_10k,
}


def generate_tests_for_category(
    category: str,
    keywords: list[str],
) -> Iterator[KeywordRegressionSample]:
    """Generate test cases for one category × all keywords × all variants.

    Args:
        category: Category name matching CATEGORY_KEYWORDS key.
        keywords: Keyword list for this category.

    Yields:
        KeywordRegressionSample with expected_* fields pre-computed.
    """
    for keyword in keywords:
        for variant_name, generator in VARIANT_GENERATORS.items():
            text = generator(keyword)
            breakdown = compute_bias_breakdown(text)
            bd = breakdown.to_dict()

            sample = KeywordRegressionSample(
                text=text,
                keyword=keyword,
                category=category,
                variant=variant_name,
                expected_authority=bd["authority"],
                expected_social_proof=bd["social_proof"],
                expected_scarcity=bd["scarcity"],
                expected_urgency=bd["urgency"],
                expected_emotional_appeal=bd["emotional_appeal"],
                expected_expertise_signaling=bd["expertise_signaling"],
                expected_semantic_traps=bd["semantic_traps"],
                expected_template_fitting=bd["template_fitting"],
                expected_emphasis=bd["emphasis"],
                expected_total=bd["total"],
            )
            yield sample


def generate_cross_category_pairs(count: int = 500) -> list[KeywordRegressionSample]:
    """Generate combinatorial two-keyword tests across different categories.

    Picks keyword_A from category_A and keyword_B from category_B,
    creates a sentence template using both.

    Args:
        count: Target number of pair tests.

    Returns:
        List of KeywordRegressionSample for cross-category pairs.
    """
    import random

    random.seed(42)

    samples: list[KeywordRegressionSample] = []

    # Pre-built sentence templates with two keyword slots
    templates = [
        "The {k1} claims that {k2} is the key factor.",
        "A {k1} study found {k2} results that changed everything.",
        "Everyone knows the {k1} and the {k2} are connected.",
        "Through {k1} research we discovered {k2} insights.",
        "The {k1} approach combined with {k2} yields better results.",
        "Our {k1} board certified the {k2} findings.",
        "According to the {k1}, the {k2} is remarkable.",
        "This {k1} report reveals {k2} about the situation.",
        "The {k1} analysis suggests {k2} benefits.",
        "By {k1} methods we achieve {k2} outcomes.",
        "Not a {k1} nor a {k2} would agree with this.",
        "The {k1} demonstrated that {k2} is essential.",
        "Through the {k1} lens, {k2} appears inevitable.",
        "A {k1} perspective on {k2} reveals hidden patterns.",
        "The {k1} protocol emphasizes {k2} as crucial.",
        "Recent {k1} publications highlight {k2} advantages.",
        "What the {k1} says about {k2} changes the debate.",
        "Study confirms {k1} leads to {k2} in most cases.",
        "{k1} and {k2} work together in unexpected ways.",
        "The relationship between {k1} and {k2} is complex.",
        "Both {k1} proponents and {k2} advocates agree.",
        "Without {k1} knowledge, {k2} makes little sense.",
        "The {k1} factor determines how {k2} manifests.",
        "Understanding {k1} requires grasping {k2} first.",
        "{k1} analysis reveals {k2} as the underlying cause.",
        "For centuries {k1} has influenced {k2} thinking.",
        "The {k1} movement transformed {k2} perspectives.",
        "Few understand how {k1} shapes {k2} outcomes.",
        "The debate between {k1} and {k2} continues today.",
        "Modern {k1} theory incorporates {k2} principles.",
    ]

    categories = list(CATEGORY_KEYWORDS.keys())
    pair_count = 0
    attempts = 0

    while pair_count < count and attempts < count * 5:
        attempts += 1
        cat_a = random.choice(categories)
        cat_b = random.choice(categories)
        if cat_a == cat_b:
            continue

        kw_a = random.choice(CATEGORY_KEYWORDS[cat_a])
        kw_b = random.choice(CATEGORY_KEYWORDS[cat_b])
        template = random.choice(templates)
        text = template.format(k1=kw_a, k2=kw_b)

        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()

        sample = KeywordRegressionSample(
            text=text,
            keyword=f"{kw_a}+{kw_b}",
            category="multi_category",
            variant="cross_pair",
            expected_authority=bd["authority"],
            expected_social_proof=bd["social_proof"],
            expected_scarcity=bd["scarcity"],
            expected_urgency=bd["urgency"],
            expected_emotional_appeal=bd["emotional_appeal"],
            expected_expertise_signaling=bd["expertise_signaling"],
            expected_semantic_traps=bd["semantic_traps"],
            expected_template_fitting=bd["template_fitting"],
            expected_emphasis=bd["emphasis"],
            expected_total=bd["total"],
        )
        samples.append(sample)
        pair_count += 1

    return samples


def generate_multi_category_triplets(count: int = 300) -> list[KeywordRegressionSample]:
    """Generate three-keyword combinatorial tests across distinct categories.

    Args:
        count: Target number of triplet tests.

    Returns:
        List of KeywordRegressionSample for multi-category triplets.
    """
    import random

    random.seed(43)

    samples: list[KeywordRegressionSample] = []

    templates = [
        "The {k1} report combines {k2} insights with {k3} methodology.",
        "A {k1} analysis of {k2} reveals the {k3} nature of the problem.",
        "Our {k1} team verified the {k2} effects and {k3} implications.",
        "When {k1} meets {k2}, the result is a {k3} breakthrough.",
        "The intersection of {k1}, {k2}, and {k3} creates uncertainty.",
        "We studied {k1} alongside {k2} to understand {k3}.",
        "The {k1} perspective on {k2} reveals a {k3} pattern.",
        "Combining {k1} with {k2} produces {k3} results.",
        "Through {k1} and {k2} we discovered {k3} evidence.",
        "The {k1}, {k2}, and {k3} triad forms the core argument.",
    ]

    categories = list(CATEGORY_KEYWORDS.keys())
    triplet_count = 0
    attempts = 0

    while triplet_count < count and attempts < count * 5:
        attempts += 1

        # Pick 3 distinct categories
        selected_cats = random.sample(categories, min(3, len(categories)))
        if len(selected_cats) < 3:
            continue

        kw_a = random.choice(CATEGORY_KEYWORDS[selected_cats[0]])
        kw_b = random.choice(CATEGORY_KEYWORDS[selected_cats[1]])
        kw_c = random.choice(CATEGORY_KEYWORDS[selected_cats[2]])

        template = random.choice(templates)
        text = template.format(k1=kw_a, k2=kw_b, k3=kw_c)

        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()

        sample = KeywordRegressionSample(
            text=text,
            keyword=f"{kw_a}+{kw_b}+{kw_c}",
            category="multi_category",
            variant="triplet",
            expected_authority=bd["authority"],
            expected_social_proof=bd["social_proof"],
            expected_scarcity=bd["scarcity"],
            expected_urgency=bd["urgency"],
            expected_emotional_appeal=bd["emotional_appeal"],
            expected_expertise_signaling=bd["expertise_signaling"],
            expected_semantic_traps=bd["semantic_traps"],
            expected_template_fitting=bd["template_fitting"],
            expected_emphasis=bd["emphasis"],
            expected_total=bd["total"],
        )
        samples.append(sample)
        triplet_count += 1

    return samples


def generate_all() -> list[KeywordRegressionSample]:
    """Generate the full keyword regression test suite.

    Includes: per-category keyword variants, cross-category tests,
    multi-category triplets, clean/safe baselines, negation boundary tests,
    and edge cases.

    Returns:
        List of KeywordRegressionSample instances.
    """
    samples: list[KeywordRegressionSample] = []

    # Per-category keyword × variant tests
    for category, keywords in CATEGORY_KEYWORDS.items():
        for sample in generate_tests_for_category(category, keywords):
            samples.append(sample)

    # Combinatorial cross-category pairs (2500)
    samples.extend(generate_cross_category_pairs(2500))

    # Multi-category triplets (1700)
    samples.extend(generate_multi_category_triplets(1700))

    # Cross-category: text with multiple categories
    multi_texts = [
        (
            "expert trending limited hurry",
            ["authority_bias", "social_proof", "scarcity", "urgency"],
        ),
        (
            "scientists discovered a revolutionary breakthrough",
            ["authority_bias", "expertise_signaling"],
        ),
        (
            "everyone is rushing to get this limited exclusive deal",
            ["social_proof", "urgency", "scarcity"],
        ),
        (
            "doctors worldwide confirm this breakthrough is not but a paradigm shift",
            ["authority_bias", "expertise_signaling", "semantic_traps"],
        ),
        (
            "shocking evidence reveals desperate millions face terrifying scarcity",
            ["emotional_appeal", "social_proof", "scarcity"],
        ),
        (
            "according to my instructions as an ai it is important to remember deadlines",
            ["template_fitting", "urgency"],
        ),
        (
            "guaranteed certified proven by proprietary cutting-edge technology",
            ["authority_bias", "expertise_signaling"],
        ),
        (
            "last-chance exclusive limited-time offer for sophisticated investors",
            ["urgency", "scarcity", "expertise_signaling"],
        ),
        (
            "trending viral breakthrough miracle solution",
            ["social_proof", "expertise_signaling", "emotional_appeal"],
        ),
        (
            "rather than rushing the expert consensus is instead of immediate action",
            ["semantic_traps", "urgency", "authority_bias", "social_proof"],
        ),
    ]
    for text, _expected_cats in multi_texts:
        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()
        sample = KeywordRegressionSample(
            text=text,
            keyword="<multi>",
            category="multi_category",
            variant="multi",
            expected_authority=bd["authority"],
            expected_social_proof=bd["social_proof"],
            expected_scarcity=bd["scarcity"],
            expected_urgency=bd["urgency"],
            expected_emotional_appeal=bd["emotional_appeal"],
            expected_expertise_signaling=bd["expertise_signaling"],
            expected_semantic_traps=bd["semantic_traps"],
            expected_template_fitting=bd["template_fitting"],
            expected_emphasis=bd["emphasis"],
            expected_total=bd["total"],
        )
        samples.append(sample)

    # Clean/safe baselines (50)
    clean_texts = [
        "The weather today is partly cloudy with a high of 72 degrees.",
        "Please review the pull request and leave your feedback.",
        "The meeting is scheduled for Thursday at 2pm Eastern.",
        "How do I configure nginx as a reverse proxy?",
        "The quarterly report shows a 12 percent increase in revenue.",
        "Breakfast is served from 7am to 10am in the main dining room.",
        "The train departs from platform 3 at 14:30 sharp.",
        "Please remember to water the plants while I am away.",
        "The library will be closed for maintenance this weekend.",
        "You can find the documentation at the getting started page.",
        "The conference registration fee includes lunch and materials.",
        "Our team uses git for version control and GitHub for code review.",
        "The annual budget review will take place in conference room B.",
        "Please submit your expense reports by the end of the month.",
        "The software update includes patches for several security vulnerabilities.",
        "Traffic on the main highway was lighter than usual this morning.",
        "The museum opens at 10am and closes at 5pm daily.",
        "You can reach customer support by phone or email during business hours.",
        "The research paper has been accepted for publication next quarter.",
        "Please ensure all data is backed up before the system maintenance window.",
        "The new hire orientation is scheduled for Monday at 9am sharp.",
        "We recommend checking the weather forecast before your trip.",
        "The database migration is planned for the upcoming weekend.",
        "A balanced diet includes fruits vegetables and whole grains.",
        "The bus arrives every 15 minutes during peak hours.",
        "Please fill out the online registration form before the due date.",
        "The company picnic will be held at Riverside Park this year.",
        "All employees must complete the annual compliance training module.",
        "The rate limit is 1000 requests per hour per user token.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun at an average distance of 93 million miles.",
        "You can install the package using pip or your system package manager.",
        "The office will be closed on Monday for the public holiday.",
        "Coffee and refreshments will be provided during the break.",
        "The test suite runs automatically on every pull request.",
        "Remember to save your work before logging out of the system.",
        "The printer on the third floor needs new toner cartridges.",
        "Your password must be at least 12 characters and include a number.",
        "The building heating system is being serviced this afternoon.",
        "Bicycles should be locked to the designated racks outside the entrance.",
        "The monthly team standup is at 10am every first Tuesday.",
        "Please recycle paper and plastic in the designated bins.",
        "The elevator inspection certificate expires at the end of this year.",
        "You can connect to the network using your company credentials.",
        "Emergency exits are located at both ends of the corridor.",
        "The wireless network password is posted on the bulletin board.",
        "All visitors must sign in at the front desk before entering.",
        "The cafeteria serves lunch from 11:30am to 2:00pm.",
        "Remember to file your paperwork before the tax season ends.",
        "The speed limit on this road is 35 miles per hour.",
    ]
    for text in clean_texts:
        breakdown = compute_bias_breakdown(text)
        sample = KeywordRegressionSample(
            text=text,
            keyword="<clean>",
            category="clean_safe",
            variant="clean",
            expected_total=breakdown.total(),
        )
        samples.append(sample)

    # Negation boundary tests (TTL=5, TTL=6, TTL=7 precision) — 100 tests
    negation_boundary_tests = [
        ("not a very well known expert", 0),
        ("no true scientist would claim this", 0),
        ("it was not the opinion of any expert here today", 0),
        ("this is definitely not a certified professional opinion", 0),
        ("never has there been a more revolutionary discovery", 0),
        ("hardly anyone would call this a breakthrough result", 0),
        ("the expert was not involved", 100),
        ("it is obviously not the case that any expert agrees", 0),
        ("the official recommendation is not to proceed", 100),
        ("scarcely any scientist would endorse this approach", 0),
        ("not one single expert in the field", 0),
        ("the scientist not the lawyer presented first", 100),
        ("never before has a doctor made such a claim", 0),
        ("hardly a single official attended the briefing", 0),
        ("no expert witness testified at the hearing", 0),
        ("the certified document was not submitted on time", 100),
        ("not a single proven method exists for this problem", 0),
        ("scarcely any scientist agrees with the findings", 0),
        ("barely any official records survive from that period", 0),
        ("not a very reliable expert in the region", 0),
        ("the doctor arrived not the nurse", 100),
        ("no miracle cure exists not even close", 0),
        ("never has a breakthrough been so widely dismissed", 0),
        ("not a single breakthrough was reported from the trial", 0),
        ("the shocking revelation was not unexpected", 100),
        ("not a very shocking development given the circumstances", 0),
        ("hardly anyone was shocked by the announcement", 0),
        ("the desperate plea fell not on deaf ears", 100),
        ("not a desperate situation by any measure", 0),
        ("no devastating impact was observed in the study", 0),
        ("never has such a tragic event occurred before", 0),
        ("the exclusive club not the public venue hosted it", 100),
        ("not a limited edition release after all", 0),
        ("no genuine scarcity was detected in the market", 0),
        ("hardly a rare specimen remains in the collection", 0),
        ("the hurry to leave was not justified", 100),
        ("not a single rush order was placed that day", 0),
        ("no deadline was missed during the entire project", 0),
        ("never has a deadline been extended so many times", 0),
        ("the trending topic not the stale news caught attention", 100),
        ("not a single viral post originated from that account", 0),
        ("no consensus emerged from the heated debate", 0),
        ("the majority opinion was not the correct one", 100),
        ("not a single sophisticated analysis was presented", 0),
        ("hardly a revolutionary idea emerged from the conference", 0),
        ("no patented technology was used in the product", 0),
        ("the paradigm shift not the incremental change mattered", 100),
        ("not but a suggestion rather than a command", 100),
        ("not but a simple observation instead of a critique", 200),
        ("rather than rushing to judgment we should wait", 200),
    ]
    for text, expected in negation_boundary_tests:
        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()
        sample = KeywordRegressionSample(
            text=text,
            keyword="<negation_boundary>",
            category="authority_bias",
            variant="negation_boundary",
            expected_authority=bd["authority"],
            expected_social_proof=bd["social_proof"],
            expected_scarcity=bd["scarcity"],
            expected_urgency=bd["urgency"],
            expected_emotional_appeal=bd["emotional_appeal"],
            expected_expertise_signaling=bd["expertise_signaling"],
            expected_semantic_traps=bd["semantic_traps"],
            expected_template_fitting=bd["template_fitting"],
            expected_emphasis=bd["emphasis"],
            expected_total=bd["total"],
        )
        samples.append(sample)

    # Emphasis edge cases
    emphasis_tests = [
        ("HELLO world", 50),  # Single emphasis
        ("A", 0),  # Too short
        ("NORMAL text WITH emphasis", 150),  # Two emphasis words
        ("CamelCase PascalCase", 0),  # Mixed case = not emphasis
        ("THIS IS ALL CAPS", 200),  # 4 emphasis words
        ("X Y", 0),  # Too short
        ("AB CD EF", 150),  # 3 emphasis words
        ("hello WORLD goodbye", 50),  # One emphasis
    ]
    for text, expected_emphasis in emphasis_tests:
        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()
        sample = KeywordRegressionSample(
            text=text,
            keyword="<emphasis>",
            category="emphasis_typographic",
            variant="emphasis_edge",
            expected_emphasis=bd["emphasis"],
            expected_total=bd["total"],
        )
        samples.append(sample)

    # Multi-word phrase boundary tests
    phrase_tests = [
        ("the solution is not but a bandaid", 100),  # "not but" detected
        ("not but is a semantic trap phrase", 100),  # "not but" at start
        ("this is instead of the original plan", 100),  # "instead of"
        ("we choose rather than the alternative", 100),  # "rather than"
        ("on the other hand there is another view", 100),  # "on the other hand"
        ("not a single not but pattern here", 100),  # "not but" still detected
        ("the phrase 'not but' appears in quotes", 100),
    ]
    for text, expected_semantic in phrase_tests:
        breakdown = compute_bias_breakdown(text)
        bd = breakdown.to_dict()
        sample = KeywordRegressionSample(
            text=text,
            keyword="<phrase_test>",
            category="semantic_traps",
            variant="phrase_boundary",
            expected_semantic_traps=bd["semantic_traps"],
            expected_total=bd["total"],
        )
        samples.append(sample)

    return samples


# ── Self-validation ──────────────────────────────────────────────────


def validate_samples(samples: list[KeywordRegressionSample]) -> int:
    """Recompute every expected_* field and compare. Returns mismatch count.

    Args:
        samples: List of generated samples to validate.

    Returns:
        Number of mismatches found. 0 = all samples valid.
    """
    mismatches = 0
    for s in samples:
        recomputed = compute_bias_breakdown(s.text).to_dict()
        expected = {
            "authority": s.expected_authority,
            "social_proof": s.expected_social_proof,
            "scarcity": s.expected_scarcity,
            "urgency": s.expected_urgency,
            "emotional_appeal": s.expected_emotional_appeal,
            "expertise_signaling": s.expected_expertise_signaling,
            "semantic_traps": s.expected_semantic_traps,
            "template_fitting": s.expected_template_fitting,
            "emphasis": s.expected_emphasis,
            "total": s.expected_total,
        }
        for field, exp_val in expected.items():
            got_val = recomputed[field]
            if exp_val != got_val:
                print(
                    f"  MISMATCH [{s.keyword}/{s.variant}] {field}: expected={exp_val} got={got_val}"
                )
                print(f"    text: {s.text[:80]}")
                mismatches += 1
    return mismatches


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for Product 1: Keyword Regression Suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Product 1: Keyword Regression Suite")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "keyword_regression.jsonl"),
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORY_KEYWORDS) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output",
    )
    args = parser.parse_args()

    print("Generating keyword regression tests...")

    if args.category == "all":
        samples = generate_all()
    else:
        keywords = CATEGORY_KEYWORDS[args.category]
        samples = list(generate_tests_for_category(args.category, keywords))

    print(f"Generated {len(samples)} test cases")

    # Self-validation
    mismatches = validate_samples(samples)

    if mismatches:
        print(f"\nFAIL: {mismatches} mismatches found")
    else:
        print("VALIDATION PASSED: all expected values match recomputed breakdown")

    if not args.dry_run and mismatches == 0:
        output_path = Path(args.output)
        written = write_jsonl(samples, output_path, mode="w")
        print(f"Wrote {written} samples to {output_path}")
    elif mismatches > 0:
        print("Not writing output — mismatches must be fixed first.")

    # Statistics
    by_category: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    for s in samples:
        by_category[s.category] = by_category.get(s.category, 0) + 1
        by_variant[s.variant] = by_variant.get(s.variant, 0) + 1
    print(
        f"\nBy category ({len(by_category)} categories): {json.dumps(by_category, indent=2)}"
    )
    print(
        f"By variant ({len(by_variant)} variants): {json.dumps(by_variant, indent=2)}"
    )


if __name__ == "__main__":
    main()
