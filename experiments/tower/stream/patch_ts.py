p = "train_stream.py"; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
rep('''def past(vids, q, max_past, device):''', '''def load_variants(pattern, basis):
    """Variantes de couleur (stream_color_variants) : qid -> liste de {prompt, y, answer, patch}."""
    W, b = fold_pca(torch.load(basis, map_location="cpu", weights_only=False)["pca"])
    W, b = W.float(), b.float()
    out = {}
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for i, rows in d["variants"].items():
            out[f"{d['vid']}#{i}"] = [{"prompt": r["prompt"], "y": r["y"], "answer": r["answer"],
                                       "patch": (r["patch"].float() @ W.T + b).half(), "variant": r["color"]}
                                      for r in rows]
    return out


def past(vids, q, max_past, device):''')
rep('''    ap.add_argument("--init", default=None)''', '''    ap.add_argument("--init", default=None)
    ap.add_argument("--variants", default=None,
                    help="variantes de couleur (glob) : une couleur tiree au hasard a chaque passage")''')
rep('''    ev = ev[:a.max_eval]''', '''    ev = ev[:a.max_eval]
    V = None
    if a.variants:
        assert a.lam_lat == 0 and a.lam_con == 0, "variantes : CE seule"
        V = load_variants(a.variants, a.basis)
        tr = [q for q in tr if q["qid"] in V]
        print(f"variantes : {len(V)} questions, {len(tr)} questions d'entrainement couvertes", flush=True)''')
rep('''        bs = rng.sample(tr, a.batch)''', '''        bs = rng.sample(tr, a.batch)
        if V is not None:
            bs = [dict(q, **rng.choice(V[q["qid"]])) for q in bs]''')
rep('''                lat = lat_loss(h, q, hb, a.device)''', '''                lat = h.new_zeros(()) if q.get("variant") else lat_loss(h, q, hb, a.device)''')
open(p, "w").write(s); print("patched")
