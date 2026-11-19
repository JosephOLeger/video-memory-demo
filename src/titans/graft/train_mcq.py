"""QCM sur des elements PRECIS du passe : la memoire doit rappeler ce que la fenetre ne voit pas.

Donnees : les QCM de ``make_mcq.py`` (annotations FineVideo : activites et objets
horodates ; la bonne reponse est terminee AVANT la fenetre, les trois distracteurs
viennent de la MEME video et ne commencent qu'APRES), et les clips / fenetres deja
encodes par ``distill_targets`` (meme echantillonnage : 256 images, clips de 16).

Eleve : Qwen gele sur la fenetre (clip ``end-1``) + la memoire ecrite avec les clips
d'avant. Perte : entropie croisee sur les logits des quatre lettres, a la premiere
position de la reponse (la notation de LongVideoBench).

Mesures a l'eval (videos tenues a l'ecart, memes QCM a chaque fois) :
  fenetre  = la greffe coupee (Qwen sur la fenetre seule)
  memoire  = la memoire de la MEME video
  mauvaise = la memoire d'une AUTRE video (meme longueur de passe) : ce que la branche
             apprend sans contenu (a priori, style des options) ; memoire - mauvaise
             est la part due au CONTENU.

    PYTHONPATH=src python -m titans.graft.train_mcq --targets 'runs/distill/targets_*.pt' \\
        --mcq mcq3.jsonl --basis cache/qwen_l24_pca128.pt --init dist_unit_s0_at2000.pt --out run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from .branch import MemoryBranch, memory_kwargs_from_args, add_memory_flags
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_distill import load_targets
from .train_load import pr, take, write_stream

LET = "ABCD"


def prompt(m):
    lines = [m["question"]] + [f"{LET[i]}. {o}" for i, o in enumerate(m["options"])]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def letter_logits(model, processor, frames, text, fps, ids, device):
    from .fit_basis import video_inputs
    inp = video_inputs(processor, frames, text, fps)
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    return model(**inp, use_cache=False).logits[0, -1, ids].float()


def past_z(vids, m, device):
    return torch.cat(vids[m["vid"]]["clips"][:m["end"] - 1], 0).float().to(device)


def batches_by_end(items, B, rng):
    by = defaultdict(list)
    for m in items:
        by[m["end"]].append(m)
    ends = list(by)
    while True:
        e = rng.choices(ends, [len(by[x]) for x in ends])[0]
        yield rng.sample(by[e], min(B, len(by[e])))


@torch.no_grad()
def evaluate(model, processor, branch, wrapper, vids, ev, ids, a):
    branch.eval()
    by = defaultdict(list)
    for m in ev:
        by[m["end"]].append(m)
    rows = []
    for e, ms in sorted(by.items()):
        for b0 in range(0, len(ms), a.batch):
            bs = ms[b0:b0 + a.batch]
            z = torch.stack([past_z(vids, m, a.device) for m in bs])
            state, S, _, _ = write_stream(branch, z, a.chunk)
            for j, m in enumerate(bs):
                v = vids[m["vid"]]
                win, p = v["windows"][e].float(), prompt(m)
                with disabled(wrapper):
                    base = int(letter_logits(model, processor, win, p, v["fps"], ids, a.device).argmax())
                wrapper.carry, wrapper.token_offset = (take(state, j), take(S, j)), 0
                mem = int(letter_logits(model, processor, win, p, v["fps"], ids, a.device).argmax())
                # mauvaise memoire : celle d'un autre QCM du lot, sur une autre video
                jj = next((i for i in range(1, len(bs)) if bs[(j + i) % len(bs)]["vid"] != m["vid"]), None)
                if jj is None:
                    wrong = mem
                else:
                    k = (j + jj) % len(bs)
                    wrapper.carry, wrapper.token_offset = (take(state, k), take(S, k)), 0
                    wrong = int(letter_logits(model, processor, win, p, v["fps"], ids, a.device).argmax())
                rows.append({"kind": m["kind"], "end": e, "gold": m["answer"],
                             "base": base, "mem": mem, "wrong": wrong})
    branch.train()
    out = {}
    for key, sel in [("all", rows)] + [(k, [r for r in rows if r["kind"] == k])
                                       for k in ("activity", "prop")]:
        if sel:
            out[key] = {"n": len(sel), **{f: sum(r[f] == r["gold"] for r in sel) / len(sel)
                                          for f in ("base", "mem", "wrong")}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--targets", required=True, help="glob des shards de distill_targets")
    ap.add_argument("--mcq", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--init", default=None, help="checkpoint de depart (branche entiere)")
    add_memory_flags(ap, write_stride=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--save-every", type=int, default=250, dest="save_every")
    ap.add_argument("--max-eval", type=int, default=None, dest="max_eval",
                    help="plafonne le nombre de QCM d'eval (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import AutoProcessor

    from .check_qwen import load_host

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    vids, _, L = load_targets(a.targets)
    # load_targets ne tronque au mode que les videos qu'il garde : on ecarte les autres
    # (memes videos que la distillation ; le batch empile des passes de meme longueur)
    vids = {k: v for k, v in vids.items() if all(c.shape[0] == L for c in v["clips"])}
    items = [json.loads(line) for line in open(a.mcq)]
    items = [m for m in items if m["vid"] in vids and m["end"] in vids[m["vid"]]["windows"]]
    tr = [m for m in items if not m["eval"]]
    ev = [m for m in items if m["eval"]][:a.max_eval]
    print(f"{len(vids)} videos, {len(tr)} QCM d'entrainement, {len(ev)} d'eval", flush=True)

    model = load_host(a.model, torch.bfloat16, a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in LET]
    site = default_layer(len(find_decoder_layers(model)))
    pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
    if a.init:                                   # la memoire de depart fixe l'architecture
        blob = torch.load(a.init, map_location="cpu", weights_only=False)
        for k in ("d_mem", "memory_hidden", "alpha_init", "muon", "write_stride",
                  "omega_window", "omega_stride", "feature_degree"):
            if k in blob["args"]:
                setattr(a, k, blob["args"][k])
    mem_kw = memory_kwargs_from_args(a)
    branch = MemoryBranch(int(model.config.text_config.hidden_size), a.d_mem,
                          chunk_size=a.chunk, write_stride=a.write_stride,
                          down_pca=pca, **mem_kw).to(a.device)
    if a.init:
        branch.load_state_dict(blob["branch"])
        print(f"depart : {a.init} (pas {blob.get('step')})", flush=True)
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

    def report(step):
        r = evaluate(model, processor, branch, wrapper, vids, ev, ids, a)
        hist.append({"step": step, **r})
        s = r["all"]
        print(f"  == eval {step}: fenetre {100 * s['base']:.1f} | memoire {100 * s['mem']:.1f} | "
              f"mauvaise {100 * s['wrong']:.1f}  || " + "  ".join(
                  f"{k} {100 * v['base']:.0f}/{100 * v['mem']:.0f}/{100 * v['wrong']:.0f}"
                  for k, v in r.items() if k != "all"), flush=True)

    hist, gate_log, t0, run = [], [], time.time(), None
    report(0)
    gen = batches_by_end(tr, a.batch, rng)
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        bs = next(gen)
        try:
            z = torch.stack([past_z(vids, m, a.device) for m in bs])
            state, S, _, (al, et, th) = write_stream(branch, z, a.chunk)
            loss, hit = 0.0, 0
            for j, m in enumerate(bs):
                v = vids[m["vid"]]
                wrapper.carry, wrapper.token_offset = (take(state, j), take(S, j)), 0
                lg = letter_logits(model, processor, v["windows"][m["end"]].float(), prompt(m),
                                   v["fps"], ids, a.device)
                loss = loss + F.cross_entropy(lg[None], torch.tensor([m["answer"]], device=a.device))
                hit += int(lg.argmax()) == m["answer"]
            loss = loss / len(bs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad))
        opt.step()
        acc = hit / len(bs)
        run = acc if run is None else 0.95 * run + 0.05 * acc
        if step % 10 == 0:
            gl = {"step": step, "eta_med": float(et.median()), "alpha_med": float(al.median()),
                  "theta_med": float(th.median()), "pr_w1": pr(state["w1"][0].detach()),
                  "loss": float(loss), "acc_run": run}
            gate_log.append(gl)
            print(f"  {step:5d} loss {float(loss):.3f} acc~ {100 * run:.1f}% |g| {gn:.2f} "
                  f"eta {gl['eta_med']:.3f} alpha {gl['alpha_med']:.1e} theta {gl['theta_med']:.3f} "
                  f"pr {gl['pr_w1']:.1f} {(time.time() - t0) / (step + 1):.2f} s/pas", flush=True)
        last = step + 1 == a.steps
        if a.save_every and ((step + 1) % a.save_every == 0 or last):
            torch.save({"branch": branch.state_dict(), "step": step + 1, "args": vars(a)},
                       f"{a.out}_at{step + 1}.pt")
        if (step + 1) % a.eval_every == 0 or last:
            report(step + 1)
        if step % 50 == 0 or last:
            with open(a.out + ".json", "w") as f:
                json.dump({"args": vars(a), "history": hist, "gates": gate_log,
                           "elapsed_s": time.time() - t0}, f, indent=1)
    print("TRAIN-MCQ-FINIE", flush=True)


if __name__ == "__main__":
    main()
