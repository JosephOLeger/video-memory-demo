"""bench30 plumbing check (30/10): what the sup / orc oracle writes on dev records (same rules as train_b30.orc_pos
at eval: cache = end state of each span (oracle_sent_ends), deep = every token of each span, content_ids unused)."""
import json, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
R = [json.loads(l) for l in open("evalsets/b30_dev_4k_1ep_sp.jsonl")]
st = collections.Counter()
endtok = collections.Counter()
for k, r in enumerate(R):
    ids, P = r["ids"], r["P"]
    past = ids[:P]
    sup = r["sup_spans"]
    sup_txt = " ".join(tok.decode(past[a:b]) for a, b in sup)
    ans_in = r["answer"].lower() in sup_txt.lower()
    st["sup_has_answer"] += ans_in
    st["sup_nonempty"] += bool(sup)
    st["n"] += 1
    for a, b in r["fact_spans"]:
        endtok[repr(tok.decode(past[b - 1:b]))] += 1
    if k < 5:
        print(f"== {r['type']} [{r['form']}] Q: {r['question']} | gold {r['answer']} | answer in sup spans: {ans_in}")
        for a, b in sup:
            print(f"   sup span {a}-{b} deep writes {b - a} tokens: {tok.decode(past[a:b])!r}")
            print(f"   cache writes position {b - 1}: token {tok.decode(past[b - 1:b])!r}, preceded by {tok.decode(past[b - 6:b - 1])!r}")
        print(f"   orc: {len(r['fact_spans'])} spans, cache writes the last 8 ends; deep writes {sum(b - a for a, b in r['fact_spans'])} tokens")
print(dict(st))
print("end tokens of fact spans:", endtok.most_common(10))
