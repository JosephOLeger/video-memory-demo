"""23/10 : flux d'eval LONGS et a LEURRES pour l'aiguille couleur (eval seule, aucun entrainement).

Chemin d'entree CORRIGE (fit_basis.video_inputs avec video_metadata, spg = 2/fps) pour tout :
clips du passe, clips a aiguille, fenetre seule, enseignant. Videos = les 124 videos d'eval de
stream/tgc (jamais vues a l'entrainement), memes clips de 16 s a 1 img/s, 1 jeton sur 8.

Conditions (fichier long/data/<cond>.pt) :
  orig   : les 300 questions d'eval d'origine (meme tirage que train_stream), recalculees.
  L8, L32, L80 : passe de N clips (videos d'eval enchainees, la video cible en dernier), une
           aiguille a distance d (en clips depuis la fenetre), bins stratifies.
  D2, D3 : N = 8, 2 ou 3 ecrans unis de couleurs distinctes ; question « premier » ou
           « dernier » ; les couleurs leurres sont parmi les options.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/work/titans-mac/src python prep.py --out data
"""
import argparse
import glob
import math
import os
import random
import time

import torch

import titans.graft.fit_basis as FB
from titans.graft.branch import fold_pca
from titans.graft.check_qwen import load_host
from titans.graft.distill import forced_inputs
from titans.graft.qwen import default_layer, find_decoder_layers
from titans.graft.stream_color_targets import Q_COLOR, fill
from titans.graft.stream_targets import COLORS, LET, TAIL, is_eval, read_fps

CLEN, NEEDLE, STRIDE = 16, 4, 8
Q_FIRST = ("Earlier in this video, the whole screen briefly turned a single color several times, "
           "each time a different color. Which color was it the first time?")
Q_LAST = ("Earlier in this video, the whole screen briefly turned a single color several times, "
          "each time a different color. Which color was it the last time?")


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


def fit(fr, ref):
    """Enseignant seulement : clip a aiguille d'une autre video remis a la taille de la fenetre."""
    if fr.shape[1:3] == ref.shape[1:3]:
        return fr
    x = torch.nn.functional.interpolate(fr.permute(0, 3, 1, 2).float(), size=tuple(ref.shape[1:3]),
                                        mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).to(fr.dtype).permute(0, 2, 3, 1).contiguous()


def make_specs(meta, seed, n_long, n_lure):
    """Questions neuves. meta: vid -> nclip. Une question = past [(vid, clip, k|None)], window."""
    rng = random.Random(seed)
    vids = sorted(meta)
    specs = {}

    def assemble(N):
        v = rng.choice(vids)
        c = rng.randrange(2, meta[v])
        past = [(v, j) for j in range(c)]
        used = {v}
        while len(past) < N:
            u = rng.choice([x for x in vids if x not in used])
            used.add(u)
            past = [(u, j) for j in range(meta[u])] + past
        return v, c, past[-N:]

    bins = {"L8": (8, [(2, 8)]), "L32": (32, [(2, 8), (9, 16), (17, 32)]),
            "L80": (80, [(2, 8), (9, 24), (25, 48), (49, 80)])}
    for cond, (N, bb) in bins.items():
        out = []
        per = n_long // len(bb)
        for lo, hi in bb:
            for _ in range(per):
                v, c, past = assemble(N)
                d = rng.randint(lo, hi)
                co = rng.choice(list(COLORS))
                opts = [co] + rng.sample([x for x in COLORS if x != co], 3)
                rng.shuffle(opts)
                out.append({"cond": cond, "N": N, "vid": v, "c": c, "past": past,
                            "needles": [{"d": d, "color": co, "f0": rng.randrange(0, CLEN - NEEDLE + 1)}],
                            "prompt": "\n".join([Q_COLOR] + [f"{LET[i]}. {o}" for i, o in enumerate(opts)] + [TAIL]),
                            "options": opts, "answer": opts.index(co), "target": co, "which": "only",
                            "bin": f"{lo}-{hi}"})
        rng.shuffle(out)
        specs[cond] = out
    for K in (2, 3):
        out = []
        for i in range(n_lure):
            v, c, past = assemble(8)
            ds = sorted(rng.sample(range(2, 9), K), reverse=True)      # du plus ancien au plus recent
            cols = rng.sample(list(COLORS), K)
            which = "first" if i % 2 == 0 else "last"
            co = cols[0] if which == "first" else cols[-1]
            lures = [x for x in cols if x != co]
            opts = [co] + lures + rng.sample([x for x in COLORS if x not in cols], 4 - K)
            rng.shuffle(opts)
            q = Q_FIRST if which == "first" else Q_LAST
            out.append({"cond": f"D{K}", "N": 8, "vid": v, "c": c, "past": past,
                        "needles": [{"d": d, "color": cl, "f0": rng.randrange(0, CLEN - NEEDLE + 1)}
                                    for d, cl in zip(ds, cols)],
                        "prompt": "\n".join([q] + [f"{LET[j]}. {o}" for j, o in enumerate(opts)] + [TAIL]),
                        "options": opts, "answer": opts.index(co), "target": co, "which": which,
                        "lures": lures, "bin": f"d{ds[0] if which == 'first' else ds[-1]}"})
        specs[f"D{K}"] = out
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgc", default=os.path.expanduser("~/work/iclr/stream/tgc"))
    ap.add_argument("--data", default=os.path.expanduser("~/work/titans-mac/data/finevideo"))
    ap.add_argument("--basis", default=os.path.expanduser("~/work/titans-mac/cache/qwen_l24_pca128.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-long", type=int, default=240, dest="n_long")
    ap.add_argument("--n-lure", type=int, default=240, dest="n_lure")
    ap.add_argument("--max-eval", type=int, default=300, dest="max_eval")
    ap.add_argument("--limit-videos", type=int, default=None, dest="limit_videos")
    ap.add_argument("--seed", type=int, default=2310)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoProcessor

    t0 = time.time()
    W, b = fold_pca(torch.load(a.basis, map_location="cpu", weights_only=False)["pca"])
    W, b = W.float(), b.float()
    proj = lambda h: (h.float() @ W.T + b).half()               # noqa: E731

    files = sorted(glob.glob(f"{a.tgc}/*.pt"))
    files = [f for f in files if is_eval(os.path.basename(f)[:-3])][:a.limit_videos]
    D = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert d["eval"]
        D[d["vid"]] = {"fps": d["fps"], "nclip": len(d["clips"]), "qs": d["qs"], "windows": d["windows"]}
    print(f"{len(D)} videos d'eval, {sum(v['nclip'] for v in D.values())} clips ({time.time() - t0:.0f} s)", flush=True)
    assert all(v["fps"] == 1.0 for v in D.values())

    # 300 questions d'origine : meme tirage que train_stream (ordre des fichiers, Random(1), [:300])
    ev = [(v, i) for v in sorted(D) for i in range(len(D[v]["qs"]))]
    random.Random(1).shuffle(ev)
    ev = ev[:a.max_eval]

    specs = make_specs({v: D[v]["nclip"] for v in D}, a.seed, a.n_long, a.n_lure)
    need = {}                                     # (vid, clip) -> list of (cond, qi, k)
    wins = {}                                     # (vid, c) -> None
    for cond, ss in specs.items():
        for qi, s in enumerate(ss):
            N = s["N"]
            for k, nd in enumerate(s["needles"]):
                need.setdefault(s["past"][N - nd["d"]], []).append((cond, qi, k))
            wins[(s["vid"], s["c"])] = None

    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a.device)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    layers = find_decoder_layers(model)
    got = []
    layers[default_layer(len(layers))].register_forward_hook(
        lambda _m, _i, out: got.append((out[0] if isinstance(out, tuple) else out).detach()))
    vtok = model.config.video_token_id
    lid = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    fps = 1.0

    @retry
    @torch.no_grad()
    def encode(fr):
        got.clear()
        inp = FB.video_inputs(proc, fr.float(), "", fps)
        assert abs(float(inp["second_per_grid_ts"][0]) - 2.0 / fps) < 1e-4
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        model(**inp, use_cache=False)
        return proj(got[0][0][inp["input_ids"][0] == vtok][::STRIDE].half().cpu())

    @retry
    @torch.no_grad()
    def letters(fr, prompt):
        inp = FB.video_inputs(proc, fr.float(), prompt, fps)
        inp = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        return model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1, lid].float().cpu()

    @retry
    @torch.no_grad()
    def forced(fr, prompt, y):
        inp, n0 = forced_inputs(proc, fr.float(), prompt, y, fps, a.device)
        # logits_to_keep = T + 1 : la position 0 gardee est n0 - 1 (memoire : pas de [L, 151k] en fp32)
        out = model(**inp, use_cache=False, logits_to_keep=int(inp["input_ids"].shape[1]) - n0 + 1)
        return out.logits[0, 0, lid].float().cpu()

    bank, needled, winfr = {}, {}, {}
    orig_by_vid = {}
    for v, i in ev:
        orig_by_vid.setdefault(v, []).append(i)
    orig = {}
    have_bank = os.path.exists(f"{a.out}/bank.pt") and os.path.exists(f"{a.out}/orig.pt")
    if have_bank:
        bank = torch.load(f"{a.out}/bank.pt", weights_only=False)["bank"]
        print("banque et orig deja calculees : seuls les clips a aiguille sont encodes", flush=True)
    todo = {cond for cond in specs if not os.path.exists(f"{a.out}/{cond}.pt")}
    for n, v in enumerate(sorted(D)):
        if have_bank and not any(cv == v for (cv, _c) in list(need) + list(wins)
                                 if True):
            continue
        frames, _ = read_fps(os.path.join(a.data, "videos", v + ".mp4"), fps, 256)
        nclip = D[v]["nclip"]
        assert frames.shape[0] // CLEN >= nclip, (v, frames.shape, nclip)
        clip = lambda c: frames[c * CLEN:(c + 1) * CLEN]          # noqa: E731
        if not have_bank:
            bank[v] = [encode(clip(c)) for c in range(nclip)]
        for c in range(nclip):
            for (cond, qi, k) in need.get((v, c), []):
                if cond not in todo:
                    continue
                nd = specs[cond][qi]["needles"][k]
                fr = fill(clip(c), nd["color"], nd["f0"], nd["f0"] + NEEDLE)
                needled[(cond, qi, k)] = (fr, encode(fr))
            if (v, c) in wins:
                winfr[(v, c)] = clip(c).clone()
        for i in ([] if have_bank else orig_by_vid.get(v, [])):
            q = D[v]["qs"][i]
            m = q["meta"]
            ev_fr = fill(clip(q["p"]), m["color"], m["f0"], m["f0"] + NEEDLE)
            win = D[v]["windows"][q["c"]]
            assert torch.equal(win, clip(q["c"])), "fenetre en cache != clip relu"
            orig[(v, i)] = {"cond": "orig", "vid": v, "c": q["c"], "N": None,
                            "past": [(v, j) for j in range(max(0, q["c"] - 8), q["c"])],
                            "needles": [{"d": q["c"] - q["p"], "color": m["color"], "f0": m["f0"]}],
                            "prompt": q["prompt"], "y": q["y"], "options": q["options"],
                            "answer": q["answer"], "target": m["color"], "which": "only",
                            "bin": f"d{q['c'] - q['p']}", "needle_z": [encode(ev_fr)],
                            "window": win.clone(), "lt_window": letters(win, q["prompt"]),
                            "lt_teacher": forced(torch.cat([ev_fr, win]), q["prompt"], q["y"]),
                            "lt_window_cached": q["lt_window"], "lt_teacher_cached": q["lt_teacher"]}
        if (n + 1) % 10 == 0:
            print(f"  [{n + 1}/{len(D)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    if not have_bank:
        torch.save({"bank": bank}, f"{a.out}/bank.pt")
        torch.save({"qs": [orig[k] for k in ev if k in orig]}, f"{a.out}/orig.pt")
    print(f"banque + orig : {(time.time() - t0) / 60:.1f} min", flush=True)

    for cond, ss in specs.items():
        if cond not in todo:
            continue
        rows = []
        for qi, s in enumerate(ss):
            co = s["target"]
            y = proc.tokenizer(f"{LET[s['answer']]}. {co}", add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
            win = winfr[(s["vid"], s["c"])]
            nfr = [fit(needled[(cond, qi, k)][0], win) for k in range(len(s["needles"]))]
            rows.append(dict(s, y=y, window=win,
                             needle_z=[needled[(cond, qi, k)][1] for k in range(len(s["needles"]))],
                             lt_window=letters(win, s["prompt"]),
                             lt_teacher=forced(torch.cat(nfr + [win]), s["prompt"], y)))
        torch.save({"qs": rows}, f"{a.out}/{cond}.pt")
        acc = lambda key: 100 * sum(int(r[key].argmax()) == r["answer"] for r in rows) / len(rows)  # noqa
        print(f"{cond}: {len(rows)} q, fenetre {acc('lt_window'):.0f} %, enseignant {acc('lt_teacher'):.0f} % "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("PREP-FINI", flush=True)


if __name__ == "__main__":
    main()
