import sys, os, time, torch
sys.argv = ["x", "--arm", "A", "--out", "/dev/null"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "pylib"))
import train_text_ab as T, gen_story2 as G
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.qwen import graft, disabled
from titans.graft.train_load import write_stream
host = T.TextHost("Qwen/Qwen2.5-VL-3B-Instruct", "cuda")
tok = host.tok
st = G.make_story(1_000_123, "train"); lays = G.layout(st, tok, P=8192, W=256, D=3000); win = lays[0]["window"]
for f in G.FAMILIES:
    it = st["items"][f][0]
    facts = [x for s, e in G.facts_of(lays[0], f) for x in lays[0]["past"][s:e]]
    q, ans = T.build_teacher(host, facts, win, it)
    x = torch.tensor([q], device="cuda")
    out = host.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=10, do_sample=False)
    print(f, "| gold:", repr(it["answer_text"]), "| teacher:", repr(tok.decode(out[0, len(q):], skip_special_tokens=True)))
pca = torch.load(os.path.join(HERE, "basis_l24_text_pca128.pt"), weights_only=False)["pca"]
br = MemoryBranch(2048, 128, chunk_size=8, write_stride=8, down_pca=pca, **memory_kwargs_from_args({})).cuda()
w = graft(host.model, br, layer=24); w.writes_enabled = False
ids = torch.tensor([lays[0]["past"], lays[1]["past"]], device="cuda")
for rep in range(2):
    torch.cuda.synchronize(); t = time.time()
    with disabled(w): h = host.layer_states(ids)
    torch.cuda.synchronize(); t1 = time.time()
    z = br.down(h)[:, ::8]; state, S, _, _ = write_stream(br, z, 8)
    torch.cuda.synchronize(); t2 = time.time()
    loss = sum(v.float().pow(2).mean() for v in state.values()); loss.backward()
    torch.cuda.synchronize(); t3 = time.time()
    print(f"encode {t1-t:.2f}s write {t2-t1:.2f}s backward-through-write {t3-t2:.2f}s; peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
