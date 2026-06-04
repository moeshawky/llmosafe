import sys, json, time
sys.path.insert(0, "/workspace/llmosafe")
from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.llm_client import get_pool, LLMResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt
from tools.generate_sifter_data.validator import validate_text
from tools.generate_sifter_data.filter import evaluate_sample

nv = get_nvpool(); ds = get_pool()
prompt = build_generate_prompt("authority_bias", 1)
acc = 0; t0 = time.time()

def parse_j(content):
    c = content.strip()
    if c.startswith("```"):
        c = c.split("\n", 1)[-1] if "\n" in c else c[3:]
        if c.endswith("```"): c = c[:-3]
    s = c.find("{"); e = c.rfind("}")
    return json.loads(c[s:e+1]) if s >= 0 and e > s else None

for attempt in range(1, 20):
    if attempt % 2 == 0:
        r = nv.chat(prompt, max_tokens=300)
        ok = isinstance(r, NvResponse)
        src = "NV"
    else:
        r = ds.chat(prompt, model="fast", temperature=0.8, max_tokens=300)
        ok = isinstance(r, LLMResponse)
        src = "DS"

    if not ok:
        print(f"  [{attempt}] {src} miss")
        time.sleep(0.3)
        continue

    p = parse_j(r.content)
    if not p:
        print(f"  [{attempt}] {src} bad_json: {r.content[:50]}")
        continue

    t = p.get("text", "")
    if not t or len(t) < 10:
        print(f"  [{attempt}] {src} short_text")
        continue

    sc = int(p.get("manipulation_score", 50))
    ism = bool(p.get("is_manipulation", sc > 50))
    v = validate_text(t)
    d = evaluate_sample(label_score=sc, label_is_manip=ism, validation=v,
                        category="authority_bias", tier=1, product=2)
    print(f"  [{attempt}] {src} SD={d.signal_density:.2f} acc={d.accepted} | {t[:60]}...")
    if d.accepted:
        acc += 1
        if acc >= 3: break

print(f"DONE: {acc} samples in {time.time()-t0:.1f}s")
nv.print_stats(); ds.print_stats()
