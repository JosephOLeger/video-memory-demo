"""Mesure OMBRE des gates : que predisent les deux jeux de tetes sur
les MEMES entrees ?

Le swap gates (vague 1) rescousse unit — mais les tetes sont des
fonctions de l'entree du bloc. Ici on rejoue l'artefact de base
normalement et, a chaque appel de _project, on evalue AUSSI les tetes
du donneur sur le meme x. On rapporte les distributions par gate
(alpha/eta/theta) pour base et donneur, par bloc.

Env : DONOR = artefact donneur. Sortie stats sur stdout (log) + le
jsonl du probe reste celui du rejeu de base (ignorable).
"""
import importlib.util
import json
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "diag_pr_collapse_probe.py"),
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

_DONOR = os.environ["DONOR"]
_orig_load = probe.load_frozen
_acc = {}  # (block, who, gate) -> list of 1D tensors


def _push(bi, who, al, et, th):
    for g, t in (("alpha", al), ("eta", et), ("theta", th)):
        _acc.setdefault((bi, who, g), []).append(
            t.detach().flatten().float().cpu())


def _load_with_shadow(path, device):
    model, info = _orig_load(path, device)
    donor, _ = _orig_load(_DONOR, device)
    n = 0
    for bi, (blk, dblk) in enumerate(zip(model.blocks, donor.blocks)):
        mem = getattr(blk, "memory", None)
        dmem = getattr(dblk, "memory", None)
        if mem is None or dmem is None:
            continue
        orig = mem._project

        def wrapped(x, _orig=orig, _dmem=dmem, _bi=bi):
            k, v, q, al, et, th = _orig(x)
            _push(_bi, "base", al, et, th)
            with torch.no_grad():
                _, _, _, dal, det, dth = _dmem._project(x)
            _push(_bi, "donor", dal, det, dth)
            return k, v, q, al, et, th

        mem._project = wrapped
        n += 1
    print(f"[shadow] tetes de {_DONOR.split('/')[-1]} evaluees en ombre "
          f"sur {n} blocs ({path})", flush=True)
    return model, info


probe.load_frozen = _load_with_shadow


def _report():
    rows = []
    for (bi, who, g) in sorted(_acc):
        v = torch.cat(_acc[(bi, who, g)])
        q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9]))
        row = {"block": bi, "who": who, "gate": g,
               "mean": float(v.mean()), "p10": float(q[0]),
               "med": float(q[1]), "p90": float(q[2])}
        if g == "eta":
            row["frac_lt_0.9"] = float((v < 0.9).float().mean())
        rows.append(row)
        print(f"SHADOW b{bi} {who:5s} {g:5s} "
              f"moy {row['mean']:.3e} p10 {row['p10']:.3e} "
              f"med {row['med']:.3e} p90 {row['p90']:.3e}"
              + (f" frac<0.9 {row['frac_lt_0.9']:.3f}" if g == "eta"
                 else ""), flush=True)
    out = os.environ.get("SHADOW_OUT")
    if out:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    probe.main()
    _report()
