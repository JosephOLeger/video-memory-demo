"""Les memoires greffees a Qwen ont-elles collapse ? pr(W1) le long d'un flux reel.

Meme mesure que le papier : on ecrit un long flux dans une memoire FRAICHE et on
suit le participation ratio de W1 (et de son increment W1 - W1_0) en fonction du
nombre d'ecritures. On y joint les portes apprises (alpha, eta, theta) sur ce flux :
le collapse du papier passait par un eta appris ~0 ([[eta-zero-collapse-mechanism]]).

Le flux : ``--frames`` images uniformes de videos tenues a l'ecart, l'hote en blocs
de 112 images (le regime de ``reach.py``). ``down`` est la base PCA gelee, commune a
tous les checkpoints, donc z est calcule une fois par video et partage.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/diag_graft_pr.py \\
        --dataset finevideo --data data/finevideo --dur-min 400 --dur-max 800 \\
        --basis cache/qwen_l24_pca128.pt --ckpts runs/cf_s8_lr1e2_s1.pt,... \\
        --out ~/work/iclr/graft_pr.json
"""
from __future__ import annotations

import argparse
import json

import torch

from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.fit_basis import resize_frames
from titans.graft.qwen import default_layer, find_decoder_layers
from titans.graft.train_color import host_states, video_pool


def pr(w: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(w.double()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def build(ckpt, d_model, pca, device):
    if ckpt == "untrained":
        t = {"d_mem": 128, "chunk": 8, "write_stride": 8, "muon": "unit"}
        torch.manual_seed(0)
        b = MemoryBranch(d_model, 128, chunk_size=8, write_stride=8, down_pca=pca,
                         **memory_kwargs_from_args(t))
    else:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        t = blob["args"]
        b = MemoryBranch(d_model, t["d_mem"], chunk_size=t["chunk"],
                         write_stride=t["write_stride"], down_pca=pca,
                         **memory_kwargs_from_args(t))
        b.load_state_dict(blob["branch"])
    return b.to(device).eval(), t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="finevideo")
    ap.add_argument("--split", default="lvb_val.json")
    ap.add_argument("--duration-group", type=int, default=600, dest="duration_group")
    ap.add_argument("--dur-min", type=float, default=400, dest="dur_min")
    ap.add_argument("--dur-max", type=float, default=800, dest="dur_max")
    ap.add_argument("--num-videos", type=int, default=6, dest="num_videos")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--ckpts", required=True, help="liste ; 'untrained' = branche fraiche")
    ap.add_argument("--frames", type=int, default=896)
    ap.add_argument("--host-block", type=int, default=112, dest="host_block")
    ap.add_argument("--checkpoints", default="64,128,330,1000,2500",
                    help="nombres d'ecritures ou mesurer pr")
    ap.add_argument("--max-side", type=int, default=256, dest="max_side")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor

    from titans.graft.check_qwen import load_host
    from titans.worldmodel.data import load_video_frames

    marks = [int(x) for x in a.checkpoints.split(",")]
    band = (a.dur_min, a.dur_max) if a.dataset == "finevideo" else None
    vids = video_pool(a.data, a.split, a.duration_group, a.num_videos, a.dataset, band)
    model = load_host(a.model, torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained(a.model)
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    d_model = int(model.config.text_config.hidden_size)
    names = [c for c in a.ckpts.split(",") if c]
    branches = {c: build(c, d_model, pca, a.device) for c in names}
    down = next(iter(branches.values()))[0].down

    acc = {c: {"pr_w1": {m: [] for m in marks}, "pr_dw1": {m: [] for m in marks},
               "gates": []} for c in names}
    for vi, vid in enumerate(vids):
        allf = resize_frames(load_video_frames(vid["path"], num_frames=a.frames), a.max_side)
        fps = a.frames / vid["dur"]
        with torch.no_grad():
            z = torch.cat([down(host_states(model, proc, allf[i:i + a.host_block], fps,
                                            a.device, site))
                           for i in range(0, allf.shape[0], a.host_block)], 1)
        for c, (b, t) in branches.items():
            zs = z[:, ::t["write_stride"]]
            mem = b.memory
            with torch.no_grad():
                _, _, _, al, et, th = mem._project(zs)
                acc[c]["gates"].append({k: [float(v.float().median()), float(v.float().quantile(.1)),
                                            float(v.float().quantile(.9))]
                                        for k, v in (("alpha", al), ("eta", et), ("theta", th))})
                s0, _ = b.init_carry(1)
                w0 = s0["w1"][0].clone()
                for m in marks:
                    if m > zs.shape[1]:
                        continue
                    st, S = b.init_carry(1)
                    st, S = mem.update(zs[:, :m], st, S, min(t["chunk"], m))
                    acc[c]["pr_w1"][m].append(pr(st["w1"][0]))
                    acc[c]["pr_dw1"][m].append(pr(st["w1"][0] - w0))
        print(f"video {vi + 1}/{len(vids)} ({z.shape[1]} tokens)", flush=True)

    mean = lambda x: sum(x) / len(x) if x else float("nan")
    out = {}
    for c in names:
        b, t = branches[c]
        g = acc[c]["gates"]
        shape = tuple(b.init_carry(1)[0]["w1"][0].shape)
        out[c] = {"muon": t.get("muon"), "stride": t["write_stride"], "w1_shape": shape,
                  "pr_w1": {m: mean(v) for m, v in acc[c]["pr_w1"].items()},
                  "pr_dw1": {m: mean(v) for m, v in acc[c]["pr_dw1"].items()},
                  "gates_median": {k: mean([x[k][0] for x in g]) for k in ("alpha", "eta", "theta")},
                  "gates_p10_p90": {k: [mean([x[k][1] for x in g]), mean([x[k][2] for x in g])]
                                    for k in ("alpha", "eta", "theta")}}
        o = out[c]
        print(f"\n{c}  muon={o['muon']} stride={o['stride']} W1{shape}")
        print("   pr(W1)     " + "  ".join(f"{m}:{v:5.1f}" for m, v in o["pr_w1"].items()))
        print("   pr(W1-W0)  " + "  ".join(f"{m}:{v:5.1f}" for m, v in o["pr_dw1"].items()))
        print("   portes med " + "  ".join(f"{k}={v:.3g}" for k, v in o["gates_median"].items())
              + "   p10-p90 " + "  ".join(f"{k}=[{v[0]:.3g},{v[1]:.3g}]"
                                         for k, v in o["gates_p10_p90"].items()))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("GRAFT-PR-FINIE", flush=True)


if __name__ == "__main__":
    main()
