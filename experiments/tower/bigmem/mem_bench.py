"""21/10 : memoire GPU de l ecriture + retropropagation seule (sans Qwen), avec et sans recalcul
par segments. Lot 4, NC clips de passe x 45 tokens, tailles d etat 64 Ko a 32 Mo.
A ajouter : environ 9 Go pour Qwen (smoke d128)."""
import sys, time, torch, types
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.train_load import write_stream
from titans.graft.write_ckpt import write_stream_ckpt

b = torch.load("/home/josephleger/work/iclr/stream/C1v_color_s1_at4000.pt", map_location="cpu", weights_only=False)
a = types.SimpleNamespace(**b["args"])
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 32
for d, h in ((128, 64), (512, 256), (1024, 512), (2048, 2048)):
    pca = torch.load(f"cache/qwen_l24_stream_pca{d}.pt", map_location="cpu", weights_only=False)["pca"]
    a.d_mem, a.memory_hidden = d, h
    br = MemoryBranch(2048, d, chunk_size=a.chunk, write_stride=a.write_stride, down_pca=pca,
                      **memory_kwargs_from_args(a)).cuda()
    z = torch.randn(4, NC * 45, d, device="cuda")
    for seg in (0, 360, 90):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); br.zero_grad()
        t = time.time()
        try:
            st, S, _, _ = write_stream_ckpt(br, z, a.chunk, seg) if seg else write_stream(br, z, a.chunk)
            sum((x.float() ** 2).mean() for x in st.values()).backward()
            torch.cuda.synchronize()
            r = f"{torch.cuda.max_memory_allocated() / 2**30:6.2f} Go {time.time() - t:6.1f} s"
        except torch.cuda.OutOfMemoryError:
            r = "OOM"
        print(f"etat d{d}/h{h} ({2 * d * h * 4 / 2**20:6.2f} Mo) segments {seg or 'aucun'} : {r}", flush=True)
    del br, z
print("MEM-BENCH-FINI")
