#!/usr/bin/env python3
"""Hardened serial farm — survives disconnects, resumable, incremental writes."""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.llm_client import get_pool, LLMResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt
from tools.generate_sifter_data.validator import validate_text
from tools.generate_sifter_data.filter import evaluate_sample
from tools.generate_sifter_data.schemas import ClassifierTrainingSample
from tools.generate_sifter_data.config import ALL_CATEGORIES

OUTPUT = os.path.join(os.path.dirname(__file__), "data", "classifier_training.jsonl")
STATE = os.path.join(os.path.dirname(__file__), "data", ".farm_state.json")
PER_TIER = 30
TIERS = [1, 2, 3]

def parse_json(content):
    c = content.strip()
    if c.startswith("```"):
        c = c.split("\n", 1)[-1] if "\n" in c else c[3:]
        if c.endswith("```"): c = c[:-3]
    try:
        s = c.find("{"); e = c.rfind("}")
        return json.loads(c[s:e+1]) if s >= 0 and e > s else None
    except: return None

def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f: return json.load(f)
    return {"cat_idx": 0, "tier_idx": 0, "count": 0}

def save_state(cat_idx, tier_idx, count):
    with open(STATE, "w") as f:
        json.dump({"cat_idx": cat_idx, "tier_idx": tier_idx, "count": count}, f)

def append_sample(sample):
    with open(OUTPUT, "a") as f:
        f.write(json.dumps({
            "text": sample.text, "category": sample.category, "tier": sample.tier,
            "tier_label": sample.tier_label, "manipulation_score": sample.manipulation_score,
            "is_manipulation": sample.is_manipulation, "classifier_prob": sample.classifier_prob,
            "classifier_is_manipulation": sample.classifier_is_manipulation,
            "bias_breakdown": sample.bias_breakdown, "signal_density": sample.signal_density,
            "gap_manipulation_score": sample.gap_manipulation_score,
        }) + "\n")

def main():
    state = load_state()
    nv = get_nvpool()
    ds = get_pool()
    cats = ALL_CATEGORIES

    print(f"RESUMING from cat_idx={state['cat_idx']} tier_idx={state['tier_idx']} count={state['count']}")
    print(f"  NVIDIA: {len(nv.keys)} keys | DeepSeek: {len(ds.keys)} keys")
    print(f"  Target: {len(cats)} cats x {len(TIERS)} tiers x {PER_TIER} = {len(cats)*len(TIERS)*PER_TIER}")
    sys.stdout.flush()

    total = state["count"]
    for ci in range(state["cat_idx"], len(cats)):
        cat = cats[ci]
        for ti in range(state["tier_idx"] if ci == state["cat_idx"] else 0, len(TIERS)):
            tier = TIERS[ti]
            prompt = build_generate_prompt(cat, tier)
            target = PER_TIER
            acc = 0
            attempts = 0
            t0 = time.time()

            while acc < target and attempts < target * 6:
                attempts += 1
                # Alternate pools each attempt
                if attempts % 2 == 0:
                    r = nv.chat(prompt, max_tokens=300)
                    if not isinstance(r, NvResponse):
                        time.sleep(0.3); continue
                    content = r.content
                else:
                    r = ds.chat(prompt, model="fast", temperature=0.8, max_tokens=300)
                    if not isinstance(r, LLMResponse):
                        time.sleep(0.3); continue
                    content = r.content

                p = parse_json(content)
                if not p: continue
                t = p.get("text", "")
                if not t or len(t) < 10: continue
                s = int(p.get("manipulation_score", 50))
                ism = bool(p.get("is_manipulation", s > 50))

                v = validate_text(t)
                d = evaluate_sample(label_score=s, label_is_manip=ism, validation=v,
                                    category=cat, tier=tier, product=2)
                if d.accepted:
                    sample = ClassifierTrainingSample(
                        text=t, category=cat, tier=tier, tier_label=f"tier{tier}",
                        manipulation_score=s, is_manipulation=ism, keywords_triggered=[],
                        classifier_prob=round(v.classifier_prob, 4),
                        classifier_is_manipulation=v.classifier_is_manipulation,
                        bias_breakdown=v.bias_breakdown,
                        signal_density=round(d.signal_density, 2),
                        gap_manipulation_score=round(d.gap_score, 2),
                        generation_attempt=1)
                    append_sample(sample)
                    acc += 1
                    total += 1

            elapsed = time.time() - t0
            rate = acc / elapsed if elapsed > 0 else 0
            save_state(ci, ti + 1, total)
            print(f"  [{cat}/tier{tier}] {acc}/{target} ({rate:.1f}/s) | total={total}")
            sys.stdout.flush()

    print(f"\nDONE: {total} samples")
    ds.print_stats(); nv.print_stats()
    # Remove state file on completion
    if os.path.exists(STATE): os.remove(STATE)

if __name__ == "__main__":
    main()
