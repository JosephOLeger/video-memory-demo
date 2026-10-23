import time, torch, sys, os, glob
sys.path.insert(0, ".")
import core as C
h = C.Host()
f = torch.load(sorted(glob.glob("cache/fv_*.pt"))[0])
hi, lo = f["hi"], f["lo"]
p = "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer with the option\x27s letter from the given choices directly."
def t(mem, win, n=5):
    h.score(mem, win, p, 4); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): _, nt = h.score(mem, win, p, 4)
    torch.cuda.synchronize(); return (time.time() - t0) / n, nt
for name, mem, win in [("key22+win8 @448", hi[:22], hi[:8]), ("fifo80@224+win8", lo[:80], hi[:8]),
                       ("oracle18+win8", hi[:18], hi[:8]), ("win8 only", None, hi[:8]),
                       ("key15+win4", hi[:15], hi[:4]), ("all@224: key22+win8", lo[:22], lo[:8]),
                       ("all@224: fifo80+win8", lo[:80], lo[:8])]:
    s, nt = t(mem, win); print(f"{name:28s} {nt:6d} tok  {s*1000:6.0f} ms  {s/nt*1e6:.1f} us/tok", flush=True)
print("peak GB", torch.cuda.max_memory_allocated() / 2**30)
t0=time.time(); e=h.encode(C.pad_even(__import__("numpy").zeros((40,448,448,3),"uint8"))); torch.cuda.synchronize(); print("encode 20 groups @448", time.time()-t0)
