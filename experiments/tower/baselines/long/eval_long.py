"""23/10 : eval des memoires (profonde C1 s0/s1/s2, pool, pool a P adapte, fifo, mean) sur les
flux de prep.py. Aucun entrainement. Meme protocole que train_stream.evaluate : lettre forcee,
memoire de la video / memoire d'une AUTRE question (autre video cible) ; fenetre seule et
enseignant viennent de prep.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src:~/work/iclr/baselines \
        python eval_long.py --data data --conds orig,L8 --out res
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import bl
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.check_qwen import load_host
from titans.graft.distill import forced_inputs
from titans.graft.qwen import default_layer, find_decoder_layers, graft
from titans.graft.train_load import write_stream
from titans.graft.stream_targets import LET

HOME = os.path.expanduser("~")
CK = {"deep_s0": f"{HOME}/work/iclr/stream/C1_color_s0_at4000.pt",
      "deep_s1": f"{HOME}/work/iclr/stream/C1_color_s1_at4000.pt",
      "deep_s2": f"{HOME}/work/iclr/stream/C1_color_s2_at4000.pt",
      "pool_s0": f"{HOME}/work/iclr/baselines/pool_s0_at4000.pt",
      "poolP_s0": f"{HOME}/work/iclr/baselines/pool_s0_at4000.pt",
      "fifo_s0": f"{HOME}/work/iclr/baselines/fifo_s0_at4000.pt",
      "mean_s0": f"{HOME}/work/iclr/baselines/mean_s0_at4000.pt"}
TYPE = {"pool_s0": "pool", "poolP_s0": "pool", "fifo_s0": "fifo", "mean_s0": "mean"}


def retry(fn):
    """La carte 1 est partagee : sur OOM, on vide le cache, on attend et on recommence."""
    def g(*x, **k):
        for t in range(60):
            try:
                return fn(*x, **k)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM ({fn.__name__}), nouvel essai dans 30 s", flush=True)
                time.sleep(30)
        raise RuntimeError("OOM persistant")
    return g


def build(arm, pca, device):
    blob = torch.load(CK[arm], map_location="cpu", weights_only=False)
    args = blob["args"]
    kw = dict(chunk_size=args["chunk"], write_stride=args["write_stride"], down_pca=pca,
              **memory_kwargs_from_args(args))
    if arm in TYPE:
        bl.CFG.update(mem_type=TYPE[arm], slots=128, pool=3)
        br = bl.BaselineBranch(2048, args["d_mem"], **kw)
    else:
        br = MemoryBranch(2048, args["d_mem"], **kw)
    br.load_state_dict(blob["branch"])
    return br.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conds", default="orig,L8,L32,L80,D2,D3")
    ap.add_argument("--arms", default="deep_s0,deep_s1,deep_s2,pool_s0,poolP_s0,fifo_s0,mean_s0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--early", type=int, default=60, help="arret si tout est au hasard apres n questions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import AutoProcessor

    os.makedirs(a.out, exist_ok=True)
    arms = a.arms.split(",")
    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    pca = torch.load(f"{HOME}/work/titans-mac/cache/qwen_l24_pca128.pt", map_location="cpu",
                     weights_only=False)["pca"]
    B = {arm: build(arm, pca, a.device) for arm in arms}
    wrapper = graft(model, B[arms[0]], layer=default_layer(len(find_decoder_layers(model))))
    wrapper.writes_enabled = False
    bank = torch.load(f"{a.data}/bank.pt", weights_only=False)["bank"]

    def past_z(q):
        N = len(q["past"])
        ks = {N - nd["d"]: k for k, nd in enumerate(q["needles"])}
        cl = [q["needle_z"][ks[i]] if i in ks else bank[v][j] for i, (v, j) in enumerate(q["past"])]
        return torch.cat(cl, 0).float().to(a.device)[None]

    @retry
    @torch.no_grad()
    def write(arm, z):
        br = B[arm]
        if arm in TYPE:
            if arm == "poolP_s0":
                br.memory.pool = max(3, math.ceil(z.shape[1] / 128))
            st, S, _ = br.memory.write(z)[0], None, None
            return st, {}
        st, S, _, _ = write_stream(br, z, 8)
        return st, S

    @retry
    @torch.no_grad()
    def student(arm, inp, n0, carry):
        wrapper.branch = B[arm]
        wrapper.carry, wrapper.token_offset = carry, 0
        out = model(**inp, use_cache=False, logits_to_keep=int(inp["input_ids"].shape[1]) - n0 + 1)
        return out.logits[0, 0, lid].float()

    for cond in a.conds.split(","):
        qs = torch.load(f"{a.data}/{cond}.pt", weights_only=False)["qs"][:a.limit]
        path = f"{a.out}/{cond}.jsonl"
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["i"] for l in open(path)}
        f = open(path, "a")
        t0, hits = time.time(), {arm: 0 for arm in arms}
        print(f"== {cond}: {len(qs)} questions, {len(done)} deja faites", flush=True)
        for i, q in enumerate(qs):
            if i in done:
                continue
            o = next((qs[(i + k) % len(qs)] for k in range(1, len(qs))
                      if qs[(i + k) % len(qs)]["vid"] != q["vid"]), q)
            z, zo = past_z(q), past_z(o)
            row = {"i": i, "cond": cond, "vid": q["vid"], "bin": q["bin"], "which": q["which"],
                   "dist": [nd["d"] for nd in q["needles"]], "gold": q["answer"],
                   "options": q["options"], "target": q["target"], "lures": q.get("lures", []),
                   "other_colors": [nd["color"] for nd in o["needles"]], "n_writes": int(z.shape[1]),
                   "window": int(q["lt_window"].argmax()), "teacher": int(q["lt_teacher"].argmax())}
            inp, n0 = forced_inputs(proc, q["window"].float(), q["prompt"], q["y"].long(), 1.0, a.device)
            tw = time.time()
            for arm in arms:
                if arm == "poolP_s0" and "pool_s0" in row and max(z.shape[1], zo.shape[1]) <= 384:
                    for sfx in ("", "_wrong", "_p"):           # P = 3 : identique a pool_s0
                        row[arm + sfx] = row["pool_s0" + sfx]
                    hits[arm] += row[arm] == q["answer"]
                    continue
                lm = student(arm, inp, n0, write(arm, z))
                lw = student(arm, inp, n0, write(arm, zo))
                row[arm] = int(lm.argmax())
                row[arm + "_wrong"] = int(lw.argmax())
                row[arm + "_p"] = float(F.softmax(lm, -1)[q["answer"]])
                hits[arm] += row[arm] == q["answer"]
            row["sec"] = time.time() - tw
            f.write(json.dumps(row) + "\n"); f.flush()
            n = i + 1
            if n % 20 == 0:
                print(f"  {cond} {n}/{len(qs)} " + " ".join(f"{arm} {100 * hits[arm] / n:.0f}" for arm in arms)
                      + f"  {(time.time() - t0) / n:.1f} s/q", flush=True)
            if a.early and n == a.early and not done and cond != "orig" and \
                    max(hits.values()) / n <= 0.35:
                print(f"  {cond}: ARRET PRECOCE, tous les bras <= 35 % apres {n}", flush=True)
                break
        f.close()
    print("EVAL-LONG-FINI", flush=True)


if __name__ == "__main__":
    main()
