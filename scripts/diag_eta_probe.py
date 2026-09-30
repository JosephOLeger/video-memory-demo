"""Dose d'eta au rejeu (question Josep : eta n'est-il pas trop grand ?).

Env : ETA_OVERRIDE = valeur constante forcee du gate de momentum au
rejeu (le _project de chaque bloc est enveloppe). eta natif median
~0.995/token -> les ecritures successives partagent ~96% de leur masse
de momentum ; on mesure ce que la correlation mecanique de eta coute
en diversite d'ecritures et en pr d'etat.
"""
import importlib.util
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "diag_pr_collapse_probe.py"),
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

_ETA = float(os.environ["ETA_OVERRIDE"])
_orig_load = probe.load_frozen


def _load_with_eta(path, device):
    model, info = _orig_load(path, device)
    n = 0
    for blk in model.blocks:
        mem = getattr(blk, "memory", None)
        if mem is None:
            continue
        orig = mem._project

        def wrapped(x, _orig=orig):
            k, v, q, al, et, th = _orig(x)
            return k, v, q, al, torch.full_like(et, _ETA), th

        mem._project = wrapped
        n += 1
    print(f"[eta] override {_ETA} sur {n} blocs ({path})", flush=True)
    return model, info


probe.load_frozen = _load_with_eta

if __name__ == "__main__":
    probe.main()
