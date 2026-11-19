"""21/10 : bilan des trois filtres sur les cibles CG-Bench (sortie de cgb_targets.py), CPU.

Pour chaque exemple (question, instant T) : bonne reponse de Qwen
  sans video (blind), fenetre seule (window), preuve remise dans la fenetre (teacher).
Un exemple est UTILE pour le module si blind faux, window faux, teacher juste.
Bilan global, par tranche de distance, par sous-categorie et par partition, et liste des
exemples utiles (keep.jsonl : vid, qid, T) pour l'entrainement / l'evaluation.

    python cgb_filter.py --tg runs/cgb/v1/tg --out runs/cgb/v1/filtre
"""
import argparse, collections, glob, json, os
import torch

BINS = [(0, 60), (60, 300), (300, 900), (900, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc = collections.defaultdict(lambda: collections.Counter())
    keep = []
    for f in sorted(glob.glob(os.path.join(a.tg, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        split = "eval" if d["eval"] else "train"
        for q in d["qs"]:
            ok = {k: int(int(q[f"lt_{k}"].argmax()) == q["answer"]) for k in ("blind", "window", "teacher")}
            m = q["meta"]
            useful = ok["blind"] == 0 and ok["window"] == 0 and ok["teacher"] == 1
            b = next(f"{lo}-{hi:.0f}s" for lo, hi in BINS if lo <= m["dist_s"] < hi)
            for key in ("tout", f"split:{split}", f"dist:{b}", f"cat:{m['sub_category']}"):
                c = acc[key]
                c["n"] += 1; c["utile"] += useful
                for k, v in ok.items():
                    c[k] += v
            if useful:
                keep.append({"vid": d["vid"], "qid": m["qid"], "T": m["T"], "split": split, "dist_s": m["dist_s"]})
    rows = {k: {"n": c["n"], **{x: round(c[x] / c["n"], 3) for x in ("blind", "window", "teacher", "utile")}}
            for k, c in sorted(acc.items())}
    json.dump(rows, open(os.path.join(a.out, "bilan.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "keep.jsonl"), "w") as fo:
        for k in keep:
            fo.write(json.dumps(k) + "\n")
    print(f"{'tranche':34s} {'n':>6s} {'aveugle':>8s} {'fenetre':>8s} {'plafond':>8s} {'utile':>6s}")
    for k, r in rows.items():
        print(f"{k:34s} {r['n']:6d} {r['blind']:8.3f} {r['window']:8.3f} {r['teacher']:8.3f} {r['utile']:6.3f}")


if __name__ == "__main__":
    main()
