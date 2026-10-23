"""22/10 : graft training / eval on scene benchmark v3 (adapted copy of titans.graft.train_stream).

Data: prep_v3.py (ladder3 layout, 2 fps, 448 px, past P=160 s cut into 8 s clips, window W=4 s).
Student = frozen Qwen on the WINDOW only, in the ladder's ``empty`` format
("The most recent seconds of the video stream:" + window video + the generator's question),
read at the last position on the option letters, plus the grafted memory written with the
past clips (own variant). Loss per item:

    L = CE(letters | own memory)
      + lam_con * relu(margin + CE(gold | own memory) - CE(gold | OTHER-VARIANT memory))

The other-variant memory comes from the SAME counterfactual group (same filler, window,
prompt, options; only the tableau clip differs, and its answer is another option), so the
contrastive term can only be lowered by reading the memory's CONTENT.

Eval (held-out split, ladder groups): per item own / other-variant (+ donor rate) / window
(graft off) / oracle (graft off, ladder oracle format: tableau as a memory video + window);
item and GROUP accuracy (all 4 variants right), per family and per D.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src python train_v3.py \\
        --train 'data/train/*.pt' --eval 'data/eval/*.pt' --init <ckpt> --steps 20 --out runs/smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_v3 as PV                 # noqa: E402

from titans.graft.branch import MemoryBranch, memory_kwargs_from_args   # noqa: E402
from titans.graft.fit_basis import check_temporal                       # noqa: E402
from titans.graft.qwen import default_layer, disabled, find_decoder_layers, graft  # noqa: E402
from titans.graft.train_load import pr, take, write_stream              # noqa: E402
from titans.graft.write_ckpt import write_stream_ckpt                   # noqa: E402

MEM_TXT = "Memory of earlier parts of the video stream:"
WIN_TXT = "The most recent seconds of the video stream:"
LET = "ABCD"


def load_groups(pattern):
    out = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        out += torch.load(f, map_location="cpu", weights_only=False)
    return out


class Fill:
    """Filler clip encodings (RAM) + window frames (decoded on demand, small LRU)."""

    def __init__(self, root):
        self.root, self.z, self.win = root, {}, {}

    def clips(self, rec):
        d, k = f"{self.root}/fill_{rec['split']}", rec["fill_key"]
        if k not in self.z:
            self.z[k] = torch.load(f"{d}/{k}.pt", map_location="cpu")["z"]
        return self.z[k]

    def window(self, rec):
        key = (rec["fill_key"], rec["start"])
        if key not in self.win:
            if len(self.win) >= 64:
                self.win.pop(next(iter(self.win)))
            self.win[key] = PV.filler_frames(rec, rec["W"], rec["P"])
        return self.win[key]


MEM_STRIDE = 1


def past(fill, rec, D, v, max_past, device):
    """[1, N, 128] memory input: the P s of filler clips, tableau clip(s) of variant v patched.

    ``MEM_STRIDE`` > 1 keeps 1 stored token in MEM_STRIDE (cost knob: writes / MEM_STRIDE)."""
    ncl = rec["P"] // rec["clip_s"]
    z = list(fill.clips(rec)[rec["j0"]:rec["j0"] + ncl])
    for k, zk in rec["patches"][D][v].items():
        z[k] = zk
    lo = max(0, ncl - max_past)
    return torch.cat([x[::MEM_STRIDE] for x in z[lo:]], 0).float().to(device)[None]


class Timer:
    """Wall time per phase (cuda-synchronised), for the s/step breakdown."""

    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(self):
                torch.cuda.synchronize()
                self.t0 = time.time()

            def __exit__(self, *e):
                torch.cuda.synchronize()
                tm.t[name] += time.time() - self.t0
        return _C()


_INP = {}


def inputs(proc, pieces, prompt, device):
    """pieces: [(label, frames uint8 [F,448,448,3])]; real 2 fps metadata, spg asserted."""
    content, vids = [], []
    for lab, fr in pieces:
        content += [{"type": "text", "text": lab}, {"type": "video"}]
        vids.append(fr)
    content.append({"type": "text", "text": prompt})
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True)
    inp = proc(text=[text], videos=vids, return_tensors="pt", do_sample_frames=False,
               video_metadata=[{"fps": PV.FPS, "total_num_frames": len(v)} for v in vids])
    check_temporal(inp, PV.FPS, proc, n_videos=len(vids))
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}


def win_inputs(proc, fill, rec, device):
    key = (rec["fill_key"], rec["start"], rec["prompt"])
    if key not in _INP:
        if len(_INP) >= 32:
            _INP.pop(next(iter(_INP)))
        _INP[key] = inputs(proc, [(WIN_TXT, fill.window(rec))], rec["prompt"], device)
    return _INP[key]


def letters(model, wrapper, inp, carry, lid):
    """Option-letter logits at the last position; carry=None -> graft off."""
    if carry is None:
        with disabled(wrapper):
            out = model(**inp, use_cache=False, logits_to_keep=1)
    else:
        wrapper.carry, wrapper.token_offset = carry, 0
        out = model(**inp, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1, lid].float()


def write(branch, z, a):
    st, S, _, g3 = (write_stream_ckpt(branch, z, a.chunk, a.ckpt_seg) if a.ckpt_seg
                    else write_stream(branch, z, a.chunk))
    return st, S, g3


@torch.no_grad()
def evaluate(model, proc, branch, wrapper, fill, ev, lid, a, oracle=True):
    branch.eval()
    rows = []
    for rec in ev:
        n = len(rec["answers"])
        lid_n = lid[:n]
        winp = win_inputs(proc, fill, rec, a.device)
        lw = letters(model, wrapper, winp, None, lid_n)              # identical across variants
        for D in rec["dists"]:
            if any(min(p) < rec["P"] // rec["clip_s"] - a.max_past for p in
                   (list(x.keys()) for x in rec["patches"][D])):
                continue                                              # tableau outside max_past
            # the n variant memories in ONE batched write (same length N for all)
            st, S, _ = write(branch, torch.cat([past(fill, rec, D, v, a.max_past, a.device)
                                                for v in range(n)]), a)
            mems = [(take(st, v), take(S, v)) for v in range(n)]
            for v in range(n):
                o = (v + 1) % n
                lm = letters(model, wrapper, winp, mems[v], lid_n)
                lo = letters(model, wrapper, winp, mems[o], lid_n)
                r = {"family": rec["family"], "kind": rec["kind"], "group": rec["group"],
                     "variant": v, "D": D, "gold": rec["answers"][v], "donor": rec["answers"][o],
                     "mem": int(lm.argmax()), "other": int(lo.argmax()), "window": int(lw.argmax()),
                     "lg_mem": [round(x, 3) for x in lm.tolist()]}
                if oracle:
                    it = PV.make_group(rec["family"], rec["group"], rec["split"])[v]
                    oinp = inputs(proc, [(MEM_TXT, PV.tableau_frames(it)), (WIN_TXT, fill.window(rec))],
                                  rec["prompt"], a.device)
                    r["oracle"] = int(letters(model, wrapper, oinp, None, lid_n).argmax())
                rows.append(r)
    branch.train()
    return rows


def summarise(rows):
    conds = [c for c in ("mem", "other", "window", "oracle") if rows and c in rows[0]]

    def cell(sel):
        out = {"n": len(sel)}
        for c in conds:
            out[c] = 100 * sum(r[c] == r["gold"] for r in sel) / len(sel)
            g = defaultdict(list)
            for r in sel:
                g[(r["family"], r["group"], r["D"])].append(r[c] == r["gold"])
            out[c + "_grp"] = 100 * sum(all(x) for x in g.values()) / len(g)
        out["donor"] = 100 * sum(r["other"] == r["donor"] for r in sel) / len(sel)
        return out
    res = {"all": cell(rows)} if rows else {}
    for key in ("family", "D", "kind"):
        for val in sorted({r[key] for r in rows}, key=str):
            res[f"{key}={val}"] = cell([r for r in rows if r[key] == val])
    return res


def show(tag, res):
    for k, s in res.items():
        print(f"  [{tag}] {k:22s} n {s['n']:4d} | own {s['mem']:5.1f} (grp {s['mem_grp']:5.1f}) | "
              f"other {s['other']:5.1f} (grp {s['other_grp']:5.1f}, donor {s['donor']:5.1f}) | "
              f"window {s['window']:5.1f} (grp {s['window_grp']:5.1f})"
              + (f" | oracle {s['oracle']:5.1f} (grp {s['oracle_grp']:5.1f})" if "oracle" in s else ""),
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default=f"{HOME}/work/iclr/graft_v3/data")
    ap.add_argument("--train", default="train/*.pt")
    ap.add_argument("--eval", default="eval/*.pt")
    ap.add_argument("--families", default=",".join(PV.FAMS))
    ap.add_argument("--basis", default=f"{HOME}/work/titans-mac/cache/qwen_l24_pca128.pt")
    ap.add_argument("--lam-con", type=float, default=1.0, dest="lam_con")
    ap.add_argument("--margin", type=float, default=1.0, help="nats, CE(other) - CE(own)")
    ap.add_argument("--max-past", type=int, default=20, dest="max_past", help="clips of memory at most")
    ap.add_argument("--mem-stride", type=int, default=1, dest="mem_stride",
                    help="keep 1 stored memory token in N (256/N writes per 8 s clip)")
    ap.add_argument("--ckpt-seg", type=int, default=512, dest="ckpt_seg")
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--memory-hidden", type=int, default=64, dest="memory_hidden")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--write-stride", type=int, default=8, dest="write_stride")
    ap.add_argument("--alpha-init", type=float, default=1.27e-4, dest="alpha_init")
    ap.add_argument("--muon", default="unit")
    ap.add_argument("--omega-window", type=int, default=1, dest="omega_window")
    ap.add_argument("--omega-stride", type=int, default=1, dest="omega_stride")
    ap.add_argument("--feature-degree", type=int, default=1, dest="feature_degree")
    ap.add_argument("--init", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    ap.add_argument("--eval-groups", type=int, default=None, dest="eval_groups",
                    help="groups per family in the periodic eval")
    ap.add_argument("--no-oracle", action="store_true", dest="no_oracle")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only")
    ap.add_argument("--save-every", type=int, default=500, dest="save_every")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from titans.graft.check_qwen import load_host

    global MEM_STRIDE
    MEM_STRIDE = a.mem_stride
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    fams = set(a.families.split(","))
    fill = Fill(a.data)
    tr = [] if a.eval_only else [r for r in load_groups(f"{a.data}/{a.train}") if r["family"] in fams]
    tr = [r for r in tr if min(r["patches"][r["dists"][0]][0]) >= r["P"] // r["clip_s"] - a.max_past]
    ev = [r for r in load_groups(f"{a.data}/{a.eval}") if r["family"] in fams]
    if a.eval_groups:
        seen = defaultdict(int)
        ev = [r for r in ev if (seen.__setitem__(r["family"], seen[r["family"]] + 1) or seen[r["family"]]) <= a.eval_groups]
    print(f"train {len(tr)} groups, eval {len(ev)} groups ({sorted(fams)})", flush=True)

    model = load_host(a.model, torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained(a.model)
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        for k in ("d_mem", "memory_hidden", "alpha_init", "muon", "write_stride",
                  "omega_window", "omega_stride", "feature_degree"):
            if k in blob["args"]:
                setattr(a, k, blob["args"][k])
        assert os.path.basename(blob["args"].get("basis", a.basis)) == os.path.basename(a.basis), \
            "the checkpoint's down projection is another basis"
    branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem,
                          chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **memory_kwargs_from_args(a)).to(a.device)
    if a.init:
        branch.load_state_dict(blob["branch"])
    branch.train()
    wrapper = graft(model, branch, layer=site)
    wrapper.writes_enabled = False
    params = list(branch.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def lr_scale(step):
        if a.warmup and step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    hist, glog, t0 = [], [], time.time()

    def report(step):
        te = time.time()
        rows = evaluate(model, proc, branch, wrapper, fill, ev, lid, a, oracle=not a.no_oracle)
        res = summarise(rows)
        hist.append({"step": step, **res})
        show(f"eval {step}", res)
        print(f"  (eval {len(rows)} items, {time.time() - te:.0f} s)", flush=True)
        with open(f"{a.out}_eval{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    with torch.no_grad():                         # batched write == one write per stream
        r0 = ev[0]
        D0 = r0["dists"][0]
        zz = torch.cat([past(fill, r0, D0, v, a.max_past, a.device) for v in range(2)])
        sb, _, _ = write(branch, zz, a)
        s1, _, _ = write(branch, zz[1:2], a)
        err = max(float((sb[k][1:2] - s1[k]).abs().max() / (s1[k].abs().max() + 1e-12)) for k in s1)
        print(f"check batched write vs single: max rel err {err:.1e}", flush=True)
        assert err < 1e-3, err
    if a.eval_only:
        report(0)
        json.dump({"args": vars(a), "history": hist}, open(a.out + ".json", "w"), indent=1)
        return
    run, tstep = {}, []
    tm = Timer()
    for step in range(a.steps):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        bs = rng.sample(tr, a.batch)
        try:
            opt.zero_grad(set_to_none=True)
            tot = {"ce": 0.0, "con": 0.0, "hit": 0, "gap": 0.0}
            # the batch's own memories in ONE batched write (with graph), the other-variant
            # memories in another (no graph); the students then read slices of the carry.
            with tm("data"):
                pick = []
                for rec in bs:
                    n = len(rec["answers"])
                    v = rng.randrange(n)
                    pick.append((v, (v + rng.randrange(1, n)) % n, rec["dists"][0]))
                zv = torch.cat([past(fill, r, D, v, a.max_past, a.device) for r, (v, _, D) in zip(bs, pick)])
                zo = torch.cat([past(fill, r, D, o, a.max_past, a.device) for r, (_, o, D) in zip(bs, pick)])
            with tm("write_own"):
                st, S, gates = write(branch, zv, a)
            pr_w = pr(st["w1"][0].detach())
            if a.lam_con > 0:
                with tm("write_other"), torch.no_grad():
                    so, So, _ = write(branch, zo, a)
            # students: gradients stop at a detached copy of the carry, then ONE backward
            # through the batched write (instead of one per item)
            std = {k: t.detach().requires_grad_() for k, t in st.items()}
            Sd = {k: t.detach() for k, t in S.items()}
            for i, (rec, (v, o, D)) in enumerate(zip(bs, pick)):
                n = len(rec["answers"])
                gold = torch.tensor([rec["answers"][v]], device=a.device)
                with tm("data"):
                    winp = win_inputs(proc, fill, rec, a.device)
                with tm("student_fwd"):
                    lm = letters(model, wrapper, winp, (take(std, i), take(Sd, i)), lid[:n])
                ce = F.cross_entropy(lm[None], gold)
                loss = ce
                if a.lam_con > 0:
                    with tm("student_fwd"):
                        lo = letters(model, wrapper, winp, (take(so, i), take(So, i)), lid[:n])
                    ce_o = F.cross_entropy(lo[None], gold)
                    con = F.relu(a.margin + ce - ce_o)
                    loss = loss + a.lam_con * con
                    tot["con"] += float(con)
                    tot["gap"] += float(ce_o - ce)
                with tm("student_bwd"):
                    (loss / len(bs)).backward()
                tot["ce"] += float(ce)
                tot["hit"] += int(lm.argmax()) == rec["answers"][v]
            with tm("write_bwd"):
                ks = [k for k in st if std[k].grad is not None]
                torch.autograd.backward([st[k] for k in ks], [std[k].grad for k in ks])
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  OOM at step {step}", flush=True)
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        tstep.append(time.time() - ts)
        for k, val in tot.items():
            x = val / len(bs)
            run[k] = x if k not in run else 0.95 * run[k] + 0.05 * x
        if step % 5 == 0 or step + 1 == a.steps:
            al, et, th = gates
            gl = {"step": step, "eta_med": float(et.median()), "alpha_med": float(al.median()),
                  "theta_med": float(th.median()), "pr_w1": pr_w, "s_step": tstep[-1],
                  "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30, **run}
            glog.append(gl)
            print(f"  {step:5d} ce {run['ce']:.3f} con {run['con']:.3f} gap(o-own) {run['gap']:+.3f} "
                  f"acc~ {100 * run['hit']:.0f}% |g| {gn:.2f} eta {gl['eta_med']:.3f} "
                  f"alpha {gl['alpha_med']:.1e} theta {gl['theta_med']:.3f} pr {pr_w:.1f} "
                  f"{tstep[-1]:.2f} s/step (mean {sum(tstep) / len(tstep):.2f}) peak {gl['peak_gib']:.1f} GiB"
                  f" | per step: " + " ".join(f"{k} {v / (step + 1):.1f}s" for k, v in tm.t.items()),
                  flush=True)
        last = step + 1 == a.steps
        if a.save_every and ((step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)},
                       f"{a.out}_at{step + 1}.pt")
        if a.eval_every and ((step + 1) % a.eval_every == 0):
            report(step + 1)
        if step % 50 == 0 or last:
            json.dump({"args": vars(a), "history": hist, "gates": glog,
                       "s_per_step": sum(tstep) / max(1, len(tstep)),
                       "phase_s_per_step": {k: v / (step + 1) for k, v in tm.t.items()},
                       "elapsed_s": time.time() - t0}, open(a.out + ".json", "w"), indent=1)
    print(f"TRAIN-V3-FINI mean {sum(tstep) / max(1, len(tstep)):.2f} s/step", flush=True)


if __name__ == "__main__":
    main()
