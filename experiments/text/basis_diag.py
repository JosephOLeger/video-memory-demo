"""Does the layer-24 PCA basis keep the bAbI fact states? (26/10, after two extinctions of the oracle write)

Streaming states (slide 256) of BABILong-like train stories. For each basis: share of the (centred) variance
of the FACT token states that it explains, participation ratio of the projected fact states, and the same for
the LOCATION / NAME tokens inside facts (the content the memory must keep). Optionally fits a new basis on a
mix of fact and background states (--fit-out), fact states weighted --fact-share.
"""
import argparse
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pylib"))
os.environ.setdefault("TEXT_GEN", "gen_babilong")
import gen_babilong as G  # noqa: E402
import train_bl as T  # noqa: E402


def pr(X):
    s = torch.linalg.svdvals(X.double()) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=60)
    ap.add_argument("--bases", default="basis_l24_bl_pca128.pt,basis_l24_text_pca128.pt")
    ap.add_argument("--fit-out", default="")
    ap.add_argument("--fact-share", type=float, default=0.5)
    a = ap.parse_args()
    T.TextHost.past_chunk, T.TextHost.past_mode = 256, "slide"
    host = T.TextHost("Qwen/Qwen2.5-VL-3B-Instruct", "cuda")
    tok = host.tok
    content = {tok.encode(" " + w, add_special_tokens=False)[0] for w in G.LOCS + G.PEOPLE + ["Bill", "Fred", "Jeff"]}
    F_, B_, C_ = [], [], []
    for i in range(a.stories):
        st = G.make_story(7_000_000 + i, "train")
        L = G.layout(st, tok, P=4096, W=256, D=300)[0]
        h = host.layer_states(torch.tensor([L["past"]], device="cuda"))[0].cpu()
        fpos = [p for s, e, k in L["bounds"] if k != "scen" for p in range(s, e)]
        bpos = [p for s, e, k in L["bounds"] if k == "scen" for p in range(s, e)]
        cpos = [p for p in fpos if L["past"][p] in content]
        F_.append(h[fpos]); C_.append(h[cpos]); B_.append(h[random.Random(i).sample(bpos, min(len(bpos), 400))])
    Fm, Cm, Bm = torch.cat(F_), torch.cat(C_), torch.cat(B_)
    print(f"states: fact {Fm.shape[0]}, content tokens {Cm.shape[0]}, background {Bm.shape[0]}; "
          f"raw PR fact {pr(Fm - Fm.mean(0)):.1f}, content {pr(Cm - Cm.mean(0)):.1f}, background {pr(Bm - Bm.mean(0)):.1f}")
    bases = a.bases.split(",")
    if a.fit_out:
        from titans.worldmodel.bottleneck import fit_pca
        nf = int(a.fact_share * 30000)
        X = torch.cat([Fm[torch.randperm(Fm.shape[0])[:nf]], Bm[torch.randperm(Bm.shape[0])[:30000 - nf]]])
        p = fit_pca(X, 128)
        torch.save({"pca": p, "layer": 24, "n": X.shape[0], "source": f"gen_babilong train, fact share {a.fact_share}",
                    "past_chunk": 256, "past_mode": "slide"}, a.fit_out)
        print(f"fitted {a.fit_out}: explained on its own mix {p['explained']:.3f}")
        bases.append(a.fit_out)
    for b in bases:
        p = torch.load(os.path.join(HERE, b), map_location="cpu", weights_only=False)["pca"]
        keys = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in p.items()}
        mean, std, comp, sv = p["mean"].float(), p["std"].float(), p["components"].float(), p["sv"].float()
        out = {}
        for name, X in (("fact", Fm), ("content", Cm), ("background", Bm)):
            Xs = (X - mean) / std                      # the basis works in per-channel standardised space
            Xc = Xs - Xs.mean(0)
            Z = Xs @ comp.T
            Zc = Z - Z.mean(0)
            out[name] = (round(float((Zc ** 2).sum() / (Xc ** 2).sum()), 3), round(pr(Zc), 1), round(pr(Zc / sv), 1))
        print(f"{b}: explained share of centred standardised variance / PR projected / PR whitened: {out}  ({keys if b == bases[0] else ''})")


if __name__ == "__main__":
    main()
