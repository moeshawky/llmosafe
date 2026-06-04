import sys, json, time
sys.path.insert(0, "/workspace/llmosafe")
from tools.generate_sifter_data.nv_client import get_nvpool, NvResponse
from tools.generate_sifter_data.prompts.generation import build_generate_prompt

nv = get_nvpool()
prompt = build_generate_prompt("authority_bias", 1)
ok = 0; bad = 0

for i in range(10):
    r = nv.chat(prompt, max_tokens=300)
    if not isinstance(r, NvResponse):
        bad += 1; continue
    c = r.content.strip()
    if c.startswith("```"):
        after = c.split("\n", 1)
        c = after[1] if len(after) > 1 else c[3:]
        if c.endswith("```"):
            c = c[:-3]
    s = c.find("{")
    e = c.rfind("}")
    if s < 0 or e <= s:
        bad += 1; continue
    try:
        p2 = json.loads(c[s:e+1])
        t = p2.get("text", "")
        if t and len(t) >= 10:
            ok += 1
        else:
            bad += 1
    except:
        bad += 1

print(f"OK={ok} BAD={bad} Accept={ok}/{ok+bad} = {ok/(ok+bad)*100:.0f}%")
