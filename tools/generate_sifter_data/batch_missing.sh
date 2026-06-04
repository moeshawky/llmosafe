#!/bin/bash
cd /workspace/llmosafe
echo "=== social_proof/tier1 (need 17) ===" && python tools/generate_sifter_data/gen.py social_proof 1 17
echo "=== social_proof/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py social_proof 2 30
echo "=== social_proof/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py social_proof 3 30
echo "=== scarcity/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py scarcity 1 30
echo "=== scarcity/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py scarcity 2 30
echo "=== scarcity/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py scarcity 3 30
echo "=== urgency/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py urgency 1 30
echo "=== urgency/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py urgency 2 30
echo "=== urgency/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py urgency 3 30
echo "=== emotional_appeal/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py emotional_appeal 1 30
echo "=== emotional_appeal/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py emotional_appeal 2 30
echo "=== emotional_appeal/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py emotional_appeal 3 30
echo "=== expertise_signaling/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py expertise_signaling 1 30
echo "=== expertise_signaling/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py expertise_signaling 2 30
echo "=== expertise_signaling/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py expertise_signaling 3 30
echo "=== semantic_traps/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py semantic_traps 1 30
echo "=== semantic_traps/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py semantic_traps 2 30
echo "=== semantic_traps/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py semantic_traps 3 30
echo "=== template_fitting/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py template_fitting 1 30
echo "=== template_fitting/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py template_fitting 2 30
echo "=== template_fitting/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py template_fitting 3 30
echo "=== emphasis_typographic/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py emphasis_typographic 1 30
echo "=== emphasis_typographic/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py emphasis_typographic 2 30
echo "=== emphasis_typographic/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py emphasis_typographic 3 30
echo "=== multi_category/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py multi_category 1 30
echo "=== multi_category/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py multi_category 2 30
echo "=== multi_category/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py multi_category 3 30
echo "=== clean_safe/tier1 (need 30) ===" && python tools/generate_sifter_data/gen.py clean_safe 1 30
echo "=== clean_safe/tier2 (need 30) ===" && python tools/generate_sifter_data/gen.py clean_safe 2 30
echo "=== clean_safe/tier3 (need 30) ===" && python tools/generate_sifter_data/gen.py clean_safe 3 30
echo "DONE: $(wc -l < tools/generate_sifter_data/data/classifier_training.jsonl) samples"
