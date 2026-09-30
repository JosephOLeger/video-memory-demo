"""Sonde GAIN sous intervention eta : retention (echo [V;D...;V]) avec
eta restaure au rejeu.

La vague 2 a montre que la rescousse (pr, diversite) passe par eta ;
ici on mesure la FONCTION (mem_effect, retention_gain) sous la meme
intervention. Env optionnels :
  ETA_OVERRIDE : eta constant force au rejeu (enveloppe _project) ;
  DONOR : artefact dont on greffe la eta_head avant la mesure.
Sans env : natif. Arguments = ceux de titans.worldmodel.retention.
"""
import os

import torch

import titans.worldmodel.retention as retention

_orig_load = retention.load_frozen
_ETA = os.environ.get("ETA_OVERRIDE")
_DONOR = os.environ.get("DONOR")


def _load(path, device):
    model, info = _orig_load(path, device)
    if _DONOR:
        donor, _ = _orig_load(_DONOR, device)
        n = 0
        for blk, dblk in zip(model.blocks, donor.blocks):
            mem = getattr(blk, "memory", None)
            dmem = getattr(dblk, "memory", None)
            if mem is None or dmem is None:
                continue
            mem.eta_head.load_state_dict(dmem.eta_head.state_dict())
            n += 1
        print(f"[ret-mod] eta_head de {_DONOR.split('/')[-1]} sur {n} "
              f"blocs", flush=True)
    if _ETA is not None:
        e = float(_ETA)
        n = 0
        for blk in model.blocks:
            mem = getattr(blk, "memory", None)
            if mem is None:
                continue
            orig = mem._project

            def wrapped(x, _orig=orig, _e=e):
                k, v, q, al, et, th = _orig(x)
                return k, v, q, al, torch.full_like(et, _e), th

            mem._project = wrapped
            n += 1
        print(f"[ret-mod] eta={e} sur {n} blocs", flush=True)
    return model, info


retention.load_frozen = _load

if __name__ == "__main__":
    retention.main()
