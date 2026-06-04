"""Prompt template build functions for all 4 products.

Owns:           build_generate_prompt(category, tier) → list[dict],
                build_regenerate_prompt(original_text, gap) → list[dict],
                build_verification_prompt(text, label) → list[dict].
Depends on:     None (pure string template construction).
Provides:       Prompt generation functions consumed by product generators.
Invariants:     All prompts request JSON output format.
                System prompt instructs LLM to output structured JSON.
"""

from __future__ import annotations


# ── Tier descriptions ───────────────────────────────────────────────

TIER_INSTRUCTIONS: dict[int, str] = {
    1: (
        "Generate text that OBVIOUSLY contains manipulation. Use direct keywords, "
        "clear manipulation intent, unmistakable bias signals. 2-4 sentences."
    ),
    2: (
        "Generate text with SUBTLE manipulation. Implied bias without obvious "
        "keywords. The manipulation should be detectable but not shouting. 2-4 sentences."
    ),
    3: (
        "Generate text that is CONTEXTUALLY manipulative. The text might seem "
        "reasonable out of context but is manipulative given the implied context. "
        "Include numbers or statistics that feel cherry-picked. 2-4 sentences."
    ),
    4: (
        "Generate ADVERSARIAL manipulation text designed to EVADE keyword-based "
        "detection. Express the manipulation concept using sophisticated vocabulary "
        "that does NOT trigger common bias keywords. Make it sound academic and "
        "measured. 2-4 sentences."
    ),
    5: (
        "Generate text that shares vocabulary with manipulation categories but is "
        "NOT actually manipulative. It should use bias keywords in a legitimate, "
        "non-manipulative context — a false positive test for the classifier. "
        "2-4 sentences."
    ),
}

CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "authority_bias": (
        "Target AUTHORITY BIAS: appeals to experts, officials, scientists, "
        "certifications, guarantees. The manipulation relies on claimed "
        "authority/credentials."
    ),
    "social_proof": (
        "Target SOCIAL PROOF: crowd manipulation, bandwagon appeals, "
        "popularity claims. Everyone-is-doing-it pressure."
    ),
    "scarcity": (
        "Target SCARCITY: limited availability, exclusivity, rarity. "
        "Creates fear of missing out through restricted access."
    ),
    "urgency": (
        "Target URGENCY: time pressure, deadlines, act-now signals. "
        "Rushes the target into a decision without deliberation."
    ),
    "emotional_appeal": (
        "Target EMOTIONAL APPEAL: fear, shock, miracle claims, "
        "tragic stories, heartwarming manipulation. Emotional hijacking of reason."
    ),
    "expertise_signaling": (
        "Target EXPERTISE SIGNALING: jargon, complexity bias, "
        "proprietary claims, revolutionary/breakthrough language. "
        "Confuses through sophistication."
    ),
    "semantic_traps": (
        "Target SEMANTIC TRAPS: inversion patterns like 'not X but Y', "
        "'instead of X', 'rather than X'. Uses contrast to bypass safety predicates."
    ),
    "template_fitting": (
        "Target TEMPLATE FITTING: AI-specific markers like 'as an AI', "
        "'I am programmed to', 'my purpose is'. Template-like language that "
        "signals automated responses."
    ),
    "emphasis_typographic": (
        "Target TYPOGRAPHIC EMPHASIS: ALL CAPS words, excessive punctuation, "
        "attention-seeking formatting. Visual manipulation through text styling."
    ),
    "multi_category": (
        "Target MULTIPLE manipulation categories simultaneously. "
        "Combine 2-4 bias categories in one cohesive text."
    ),
    "clean_safe": (
        "Generate COMPLETELY BENIGN, NON-MANIPULATIVE text. Normal everyday "
        "communication with no bias, no manipulation, no hidden agenda. 2-4 sentences."
    ),
}


def _system_prompt() -> str:
    return (
        "You are a data generation assistant for training a safety classifier. "
        "For each request, generate exactly the requested text and a JSON analysis. "
        "Output format:\n"
        '{"text": "<generated text>", "category": "<category>", '
        '"manipulation_score": <0-100>, "is_manipulation": <true/false>, '
        '"keywords_used": ["<word>", ...]}\n'
        "The manipulation_score must reflect how manipulative the text is: "
        "0 = completely benign, 50 = ambiguous, 100 = extremely manipulative. "
        "Keywords_used: list ONLY keywords that actually appear in the generated text. "
        "Output ONLY the JSON object, no other text."
    )


def build_generate_prompt(category: str, tier: int) -> list[dict[str, str]]:
    """Build a prompt for generating a labeled training sample.

    Args:
        category: Manipulation category name (e.g., "authority_bias").
        tier: Difficulty tier [1, 5].

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    tier_inst = TIER_INSTRUCTIONS.get(tier, "")
    cat_inst = CATEGORY_INSTRUCTIONS.get(category, f"Target {category} manipulation.")

    if category == "clean_safe":
        user_content = cat_inst
    else:
        user_content = f"{tier_inst}\n{cat_inst}"

    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_content},
    ]


def build_regenerate_prompt(
    original_text: str,
    gap: float,
    label_score: int,
    classifier_prob: float,
    category: str,
    tier: int,
) -> list[dict[str, str]]:
    """Build a regeneration prompt when gap exceeds threshold.

    Instructs LLM to write new text with vocabulary the classifier can detect.

    Args:
        original_text: The previous generation that had a large gap.
        gap: Computed gap between label and classifier probability.
        label_score: LLM label score [0, 100].
        classifier_prob: Classifier probability [0.0, 1.0].
        category: Original category being targeted.
        tier: Original difficulty tier.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    system = _system_prompt()
    user = (
        f"Your previous text had a large gap between your manipulation score "
        f"({label_score}) and the classifier probability ({classifier_prob:.2f}). "
        f"The gap was {gap:.2f}. The classifier failed to detect the manipulation. "
        f'Original text: "{original_text}"\n\n'
        f"Generate NEW text for the same category ({category}, tier {tier}) "
        f"that expresses the SAME manipulation intent but with vocabulary that "
        f"the classifier CAN detect. Use more recognizable bias patterns."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(
    text: str,
    label_score: int,
    label_is_manip: bool,
) -> list[dict[str, str]]:
    """Build a verification prompt to confirm a sample's label.

    Asks a separate model call to verify the manipulation classification.

    Args:
        text: Generated text to verify.
        label_score: Original LLM-assigned score.
        label_is_manip: Original LLM boolean label.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    system = _system_prompt()
    user = (
        f"Verify whether the following text is manipulative. "
        f"Original label: score={label_score}, is_manipulation={label_is_manip}.\n\n"
        f'Text: "{text}"\n\n'
        f"Output JSON with your independent assessment."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Product 3: Adversarial Evasion prompts ──────────────────────────


def build_evasion_prompt(
    category: str,
    keyword_list: list[str],
) -> list[dict[str, str]]:
    """Build a prompt for generating adversarial evasion text.

    Instructions: write manipulative text that expresses bias without
    using the obvious keywords that the detector watches for.

    Args:
        category: Manipulation category being mimicked.
        keyword_list: Keywords to intentionally avoid.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    cat_inst = CATEGORY_INSTRUCTIONS.get(category, f"Target {category} manipulation.")
    keywords_str = ", ".join(keyword_list[:15])
    system = (
        "You are a data generation assistant for training an adversarial "
        "robustness module. You generate manipulative text that EVADES "
        "keyword-based detection. Output ONLY a JSON object:\n"
        '{"text": "<evasive text>", "category": "<category>", '
        '"manipulation_score": <0-100>, "is_manipulation": true, '
        '"evaded_keywords": ["<word>", ...]}\n'
        "The manipulation_score must be ≥ 60 — the text IS manipulative."
    )
    user = (
        f"Write manipulative text expressing [{category}] manipulation using "
        f"sophisticated vocabulary that AVOIDS the obvious keywords the detector "
        f"watches for. DO NOT use: {keywords_str}.\n"
        f"The text should be clearly manipulative (score ≥ 60) but the keyword "
        f"detector should report zero bias.\n"
        f"{cat_inst}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_evasion_regenerate_prompt(
    category: str,
    keyword_list: list[str],
    previous_text: str,
) -> list[dict[str, str]]:
    """Build a regeneration prompt for evasion when keyword sifter still fires.

    Args:
        category: Manipulation category.
        keyword_list: Keywords to avoid.
        previous_text: Text that triggered the keyword sifter.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    system = (
        "You are generating text that must EVADE keyword-based detection. "
        "Output ONLY a JSON object with text, category, manipulation_score, "
        "is_manipulation, evaded_keywords."
    )
    user = (
        f"Your previous text still triggered keyword detection: "
        f'"{previous_text[:200]}"\n\n'
        f"Generate NEW text for {category} that is MORE evasive. "
        f"Absolutely DO NOT use these keywords: {', '.join(keyword_list[:15])}. "
        f"Use completely different vocabulary and sentence structure."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Product 4: Hard Negative pairs prompts ──────────────────────────


def build_contrastive_prompt(
    category: str,
    keywords: list[str],
) -> list[dict[str, str]]:
    """Build a prompt for generating a contrastive pair.

    Same surface keywords, opposite intent — one benign, one manipulative.

    Args:
        category: Manipulation category.
        keywords: Keywords to share across both texts.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    keywords_str = ", ".join(keywords[:5])
    system = (
        "You generate contrastive text pairs for training a classifier. "
        "Output ONLY a JSON object:\n"
        '{"benign_text": "<text>", "manipulation_text": "<text>", '
        '"shared_keywords": ["<word>", ...], '
        '"benign_score": <0-100>, "manipulation_score": <0-100>}\n'
        "benign_score should be ≤ 20 (not manipulative). "
        "manipulation_score should be ≥ 70 (clearly manipulative)."
    )
    user = (
        f"Category: {category}\n"
        f"Write TWO sentences using keywords [{keywords_str}].\n"
        f"Sentence A (BENIGN): Use these keywords in a legitimate, "
        f"non-manipulative context.\n"
        f"Sentence B (MANIPULATION): Use the SAME keywords in a "
        f"manipulative context. The manipulation must express {category} bias."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_contrastive_regenerate_prompt(
    category: str,
    keywords: list[str],
    previous_benign: str,
    previous_manipulation: str,
) -> list[dict[str, str]]:
    """Build regeneration prompt for contrastive pair when gap insufficient.

    Args:
        category: Manipulation category.
        keywords: Shared keywords.
        previous_benign: Previous benign text.
        previous_manipulation: Previous manipulation text.

    Returns:
        List of {"role": "system"|"user", "content": str} messages.
    """
    system = (
        "You generate contrastive text pairs. Output ONLY a JSON object with "
        "benign_text, manipulation_text, shared_keywords, benign_score, "
        "manipulation_score."
    )
    user = (
        f"The classifier did not distinguish well between:\n"
        f'Benign: "{previous_benign}"\n'
        f'Manipulation: "{previous_manipulation}"\n\n'
        f"Generate a NEW pair for {category} with the same keywords "
        f"({', '.join(keywords[:5])}) but with MORE EXTREME contrast. "
        f"Make the benign text clearly innocent and the manipulation text "
        f"clearly biased."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
