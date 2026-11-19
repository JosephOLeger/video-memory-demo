"""bench30 (29/10): RECONSTRUCTED BABILong 4k for qa11-qa20 (the official HF release has these tasks only at 0k).
Official 0k TEST samples (RMT-team/babilong data/qaN/0k.json, 100 each), facts inserted in order at PG19 sentence
boundaries drawn uniformly over the whole context (booydar/babilong NoiseInjection recipe), total ~3650 Qwen tokens
like the official 4k files. Background = PG19 VALIDATION (the official files use PG19 test; not on the boxes).
Output records keep task names qaN, so the official prompt and metric are used (gen_babilong delegation)."""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_babilong as GB
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
T = 3650
out = open("evalsets/bl1120_4k_reconstructed.jsonl", "w")
for t in range(11, 21):
    d = json.load(open(f"ext/bl0k/qa{t}.json"))
    for i, ex in enumerate(d):
        facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ex["input"].strip()) if s.strip()]
        ftok = sum(len(tok.encode(" " + f)) for f in facts)
        st = {"split": "eval", "bg_seed": 1120_000 + t * 1000 + i}
        bg = GB.background(st, tok, T - ftok)
        r = random.Random(t * 1000 + i)
        pos = sorted(r.randrange(len(bg) + 1) for _ in facts)
        sents, k = [], 0
        for j in range(len(bg) + 1):
            while k < len(pos) and pos[k] == j:
                sents.append(facts[k]); k += 1
            if j < len(bg):
                sents.append(tok.decode(bg[j]).strip())
        out.write(json.dumps({"type": f"qa{t}", "question": ex["question"].strip(), "answer": ex["target"],
                              "input": " ".join(sents), "i": i}) + "\n")
print("done")
