#!/bin/bash
cd /workspace/llmosafe
DATA="tools/generate_sifter_data/data/classifier_training.jsonl"

for cat in authority_bias social_proof scarcity urgency emotional_appeal expertise_signaling semantic_traps template_fitting emphasis_typographic multi_category clean_safe; do
  for tier in 1 2 3; do
    existing=$(python3 -c "
import json
n=0
with open('$DATA') as f:
    for l in f:
        d=json.loads(l)
        if d.get('category')=='$cat' and d.get('tier')==$tier: n+=1
print(n)
" 2>/dev/null)
    if [ "$existing" -ge 30 ] 2>/dev/null; then
      echo "SKIP $cat/tier$tier (has $existing)"
      continue
    fi
    need=$((30 - existing))
    echo "=== $cat / tier $tier (need $need) ==="
    python tools/generate_sifter_data/gen.py "$cat" "$tier" "$need"
  done
done
echo "DONE: $(wc -l < $DATA) samples"
