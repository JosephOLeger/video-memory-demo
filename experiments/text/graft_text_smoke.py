"""Text-only path through the existing graft code: identity at init, memory writes, delta moves logits,
streaming in two chunks with carry (sliding window)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_text import Host
import gen_story as G
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.qwen import identity_check, graft, find_decoder_layers
h = Host()
items = G.make_group("count_end", 10000, "eval")
L = G.layout(items, h.tok, P=1024, W=256, D=0)[0]
ids = torch.tensor([L["past"][:512]], device="cuda")
inp = {"input_ids": ids}
with torch.no_grad():
    b0 = h.model(**inp, use_cache=False).logits.float()
print("base finite:", bool(torch.isfinite(b0).all()))
br = MemoryBranch(2048, 128, chunk_size=8, write_stride=8, **memory_kwargs_from_args({})).cuda()
print("n_layers", len(find_decoder_layers(h.model)))
print("identity:", identity_check(h.model, br, inp, layer=24))
torch.manual_seed(0)
with torch.no_grad():
    br.up.weight.normal_(0, 0.02)
w = graft(h.model, br, layer=24)
w.reset()
with torch.no_grad():
    base_on = h.model(input_ids=ids, use_cache=False).logits.float()
    w.enabled = False
    base_off = h.model(input_ids=ids, use_cache=False).logits.float()
    w.enabled = True
print("finite on:", bool(torch.isfinite(base_on).all()), "delta moves logits (max abs):", float((base_on - base_off).abs().max()))
# streaming: two chunks with carry, window-only forward
w.reset()
with torch.no_grad():
    for k in range(2):
        h.model(input_ids=ids[:, 256 * k:256 * (k + 1)], use_cache=False)
print("streamed tokens:", w.token_offset, "carry set:", w.carry[0] is not None)
print("GRAFT-TEXT-OK")
