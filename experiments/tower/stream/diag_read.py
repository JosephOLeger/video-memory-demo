"""20/10 soir : ou bloque la couleur ? ecriture ou lecture.

Questions d'ENTRAINEMENT qui ont des variantes (meme video, meme question, aiguille dans les 6
couleurs). Pour chaque variante : on ecrit le passe dans une memoire fraiche, puis
  A (ecriture) : l'etat de la memoire (poids rapides), projection aleatoire fixe ;
  B (lecture)  : la contribution de la branche d = up(norm(retrieve)) au dernier token du prompt ;
  C (reponse)  : la lettre choisie (couleur correcte ?).
A et B sont CENTRES par question sur les 6 variantes (on retire l'identite de la video : seule
la couleur de l'aiguille change). Sonde lineaire (ridge, un contre tous) entrainee sur la moitie
des videos, testee sur l'autre moitie. Hasard = 1/6.
  A haut, B bas  -> l'ecriture garde la couleur, la lecture (q_proj) ne la trouve pas ;
  A bas          -> l'ecriture ne garde pas la couleur (portes, force de l'aiguille) ;
  A et B hauts, C au hasard -> l'information arrive dans le flux, l'hote ne la lit pas.

    PYTHONPATH=src python diag_read.py --ckpts a.pt b.pt --targets 'tgc/*.pt' --variants 'tgv/*.pt' --basis ... --out x.jsonl
"""
import argparse, json, random, types
import torch
from transformers import AutoProcessor
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.qwen import default_layer, find_decoder_layers, graft
from titans.graft.train_load import write_stream
from titans.graft.train_stream import LET, load, load_variants, past, student
from titans.graft.check_qwen import load_host


def flat(p):
    if isinstance(p, dict):
        p = [p[k] for k in sorted(p)]
    return torch.cat([t.reshape(-1).float() for t in p])


def ridge_acc(X, y, vid, lam=1.0):
    """X [n, D] centres, y [n] classes 0..5, vid [n] : entraine sur les videos paires, teste impaires (et inverse)."""
    accs = []
    for half in (0, 1):
        tr, te = vid % 2 == half, vid % 2 != half
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Y = torch.nn.functional.one_hot(y[tr], 6).float()
        A = Xtr @ Xtr.T + lam * len(Xtr) * torch.eye(len(Xtr), device=X.device)   # forme duale
        W = Xtr.T @ torch.linalg.solve(A, Y)
        accs.append(((Xte @ W).argmax(1) == y[te]).float().mean().item())
    return round(sum(accs) / 2, 3)


@torch.no_grad()
def run(model, processor, branch, wrapper, vids, qs, V, colors, lid, a, G):
    cap = {}
    orig = branch.delta
    def delta(h, *x, **k):
        d, s, S = orig(h, *x, **k); cap["d"] = d.detach(); return d, s, S
    branch.delta = delta
    FA, FB, Y, VID, ok = [], [], [], [], 0
    for i, q in enumerate(qs):
        fa, fb = [], []
        for r in V[q["qid"]]:
            qq = dict(q, **r)
            st, S, _, _ = write_stream(branch, past(vids, qq, a.max_past, a.device), a.chunk)
            lg, _, _ = student(model, processor, wrapper, vids, qq, (st, S), lid, a.device)
            T = qq["y"].shape[0]
            fa.append(flat(st) @ G if G is not None else flat(st))
            fb.append(cap["d"][0, -T - 1].float())
            ok += int(lg.argmax()) == qq["answer"]
            Y.append(colors.index(r["variant"])); VID.append(i)
        fa, fb = torch.stack(fa), torch.stack(fb)
        FA.append(fa - fa.mean(0)); FB.append(fb - fb.mean(0))
        if i == 0:
            print(f"  premiere question ok : {len(V[q['qid']])} variantes, dim A {fa.shape[1]}, dim B {fb.shape[1]}", flush=True)
    branch.delta = orig
    X_A, X_B = torch.cat(FA), torch.cat(FB)
    y, vid = torch.tensor(Y, device=X_A.device), torch.tensor(VID, device=X_A.device)
    return {"ecriture_A": ridge_acc(X_A, y, vid), "lecture_B": ridge_acc(X_B, y, vid),
            "reponse_C": round(ok / len(Y), 3), "n_questions": len(qs), "n": len(Y),
            "norme_B": round(X_B.norm(dim=1).mean().item(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--proj", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a0 = ap.parse_args()
    vids, qs = load(a0.targets, a0.basis, 8)
    V = load_variants(a0.variants, a0.basis)
    colors = sorted({r["variant"] for rows in V.values() for r in rows})
    assert len(colors) == 6, colors
    qs = [q for q in qs if q["kind"] == "color" and not q["eval"] and q["qid"] in V and len(V[q["qid"]]) == 6]
    random.Random(3).shuffle(qs)
    seen, sel = set(), []
    for q in qs:                                   # une question par video
        if q["vid"] not in seen:
            seen.add(q["vid"]); sel.append(q)
    sel = sel[:a0.n]
    print(f"{len(sel)} questions (une par video), couleurs {colors}", flush=True)
    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a0.device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    lid = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a0.basis, map_location="cpu", weights_only=False)["pca"]
    wrapper, G = None, None
    with open(a0.out, "a") as fo:
        for path in a0.ckpts:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            a = types.SimpleNamespace(**blob["args"]); a.device = a0.device
            branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem, chunk_size=a.chunk,
                                  write_stride=a.write_stride, down_pca=pca, **memory_kwargs_from_args(a)).to(a0.device)
            branch.load_state_dict(blob["branch"]); branch.eval()
            if wrapper is None:
                wrapper = graft(model, branch, layer=site); wrapper.writes_enabled = False
            else:
                wrapper.branch = branch
            if G is None:
                D = flat(branch.init_carry(1)[0]).numel()
                G = torch.randn(D, a0.proj, generator=torch.Generator().manual_seed(0)).to(a0.device) / a0.proj ** 0.5 \
                    if D > a0.proj else None
            row = {"ckpt": path.split("/")[-1], **run(model, processor, branch, wrapper, vids, sel, V, colors, lid, a, G)}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            print(json.dumps(row), flush=True)
    print("DIAG-READ-FINI", flush=True)


if __name__ == "__main__":
    main()
