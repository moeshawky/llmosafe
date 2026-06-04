#!/usr/bin/env python3
"""Fixed generator — exponential backoff, rate-limit aware, JSON-parse resilient."""
import sys, json, time, os
sys.path.insert(0, "/workspace/llmosafe")
from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt

OUT = "/workspace/llmosafe/tools/generate_sifter_data/data/classifier_training.jsonl"
pool = get_nvpool()
category = sys.argv[1] if len(sys.argv) > 1 else "authority_bias"
tier = int(sys.argv[2]) if len(sys.argv) > 2 else 1
target = int(sys.argv[3]) if len(sys.argv) > 3 else 30

prompt = build_generate_prompt(category, tier)
acc = 0; miss_streak = 0; t0 = time.time()
print(f"[{category}/tier{tier}] target={target}", flush=True)

while acc < target:
    r = pool.chat(prompt, max_tokens=300)
    if not isinstance(r, NvResponse):
        miss_streak += 1
        backoff = min(2 ** miss_streak, 30)
        time.sleep(backoff)
        continue
    miss_streak = 0

    c = r.content.strip()
    if c.startswith("```"):
        after = c.split("\n", 1)
        c = after[1] if len(after) > 1 else c[3:]
        if c.endswith("```"): c = c[:-3]
    s = c.find("{"); e = c.rfind("}")
    if s < 0 or e <= s:
        time.sleep(0.5)
        continue
    try:
        p = json.loads(c[s:e+1])
        t = p.get("text", "")
        if not t or len(t) < 10:
            time.sleep(0.5)
            continue
    except:
        time.sleep(0.5)
        continue

    score = int(p.get("manipulation_score", 50))
    ism = bool(p.get("is_manipulation", score > 50))
    rec = {"text": t, "category": category, "tier": tier,
           "manipulation_score": score, "is_manipulation": ism,
           "source": "nvidia-mistral-small"}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    acc += 1
    if acc % 5 == 0:
        elapsed = time.time() - t0
        print(f"  {acc}/{target} ({acc/elapsed:.1f}/s)", flush=True)

elapsed = time.time() - t0
print(f"DONE {acc}/{target} in {elapsed:.0f}s ({acc/elapsed:.2f}/s)", flush=True)
