"""20/10 : pourquoi les graines couleur 1 et 2 n'apprennent pas (s0 : 100 %).

Pour chaque checkpoint : precision sur des questions d'ENTRAINEMENT et d'EVAL, avec la bonne
memoire, la memoire d'une autre video, et la fenetre seule (precalculee). Si l'entrainement
monte avec la bonne memoire ET avec celle d'une autre video, le modele apprend par coeur
(la fenetre identifie la video) au lieu de lire la memoire. Aussi : taille de la contribution
de la branche (||d|| / ||h||) et ecart des lectures bonne/autre memoire.

    PYTHONPATH=src python diag_color.py --ckpts a.pt b.pt --out x.jsonl
"""
import argparse, json, random, types
import torch
from transformers import AutoProcessor
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.qwen import default_layer, find_decoder_layers, graft
from titans.graft.train_load import write_stream
from titans.graft.train_stream import LET, load, past, student
from titans.graft.check_qwen import load_host


@torch.no_grad()
def run(model, processor, branch, wrapper, vids, qs, lid, a):
    acc = {"mem": 0, "wrong": 0, "window": 0, "agree": 0}
    for i, q in enumerate(qs):
        st, S, _, _ = write_stream(branch, past(vids, q, a.max_past, a.device), a.chunk)
        lm, _, _ = student(model, processor, wrapper, vids, q, (st, S), lid, a.device)
        o = next(qs[(i + k) % len(qs)] for k in range(1, len(qs)) if qs[(i + k) % len(qs)]["vid"] != q["vid"])
        so, So, _, _ = write_stream(branch, past(vids, o, a.max_past, a.device), a.chunk)
        lw, _, _ = student(model, processor, wrapper, vids, q, (so, So), lid, a.device)
        m, w = int(lm.argmax()), int(lw.argmax())
        acc["mem"] += m == q["answer"]; acc["wrong"] += w == q["answer"]
        acc["window"] += int(q["lt_window"].argmax()) == q["answer"]; acc["agree"] += m == w
    return {k: round(v / len(qs), 3) for k, v in acc.items()} | {"n": len(qs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a0 = ap.parse_args()
    vids, qs = load(a0.targets, a0.basis, 8)
    qs = [q for q in qs if q["kind"] == "color"]
    tr = [q for q in qs if not q["eval"] and q["teacher_ok"]]
    ev = [q for q in qs if q["eval"]]
    random.Random(1).shuffle(ev); random.Random(2).shuffle(tr)
    ev, tr = ev[:a0.n], tr[:a0.n]
    model = load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, a0.device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    lid = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a0.basis, map_location="cpu", weights_only=False)["pca"]
    wrapper = None
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
            row = {"ckpt": path.split("/")[-1], "train": run(model, processor, branch, wrapper, vids, tr, lid, a),
                   "eval": run(model, processor, branch, wrapper, vids, ev, lid, a)}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            print(json.dumps(row), flush=True)
    print("DIAG-COLOR-FINI", flush=True)


if __name__ == "__main__":
    main()
