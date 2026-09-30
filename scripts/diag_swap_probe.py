"""Chimeres au rejeu : echange de composants entre deux artefacts geles.

Env : DONOR = chemin de l'artefact donneur ; SWAP =
  rule  : regle d'ecriture seule (_orth_mode + orthogonalize_clip du
          donneur) — poids identiques, mecanique pure ;
  proj  : k_proj/v_proj/q_proj du donneur (hypothese k de Josep) ;
  gates : alpha_head/eta_head/theta_head du donneur ;
  eta / theta / alpha : UNE seule tete du donneur (decomposition du
          resultat gates de la vague 1) ;
  minit : init apprise de la memoire (memory.params) du donneur.
Le reste = diag_pr_collapse_probe inchange.

Garde-fou d'interpretation : les composants sont co-adaptes — verifier
l'engagement (normes d'ecriture) avant de lire le pr d'une chimere.
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

_DONOR = os.environ["DONOR"]
_SWAP = os.environ["SWAP"]
_orig_load = probe.load_frozen


def _load_with_swap(path, device):
    model, info = _orig_load(path, device)
    donor, _ = _orig_load(_DONOR, device)
    n = 0
    for blk, dblk in zip(model.blocks, donor.blocks):
        mem = getattr(blk, "memory", None)
        dmem = getattr(dblk, "memory", None)
        if mem is None or dmem is None:
            continue
        if _SWAP == "rule":
            mem._orth_mode = dmem._orth_mode
            mem.orthogonalize_clip = dmem.orthogonalize_clip
        elif _SWAP == "proj":
            for name in ("k_proj", "v_proj", "q_proj"):
                getattr(mem, name).load_state_dict(
                    getattr(dmem, name).state_dict())
        elif _SWAP == "gates":
            for name in ("alpha_head", "eta_head", "theta_head"):
                getattr(mem, name).load_state_dict(
                    getattr(dmem, name).state_dict())
        elif _SWAP in ("eta", "theta", "alpha"):
            name = f"{_SWAP}_head"
            getattr(mem, name).load_state_dict(
                getattr(dmem, name).state_dict())
        elif _SWAP == "minit":
            with torch.no_grad():
                for key, t in mem.memory.params.items():
                    t.data.copy_(dmem.memory.params[key].data)
        else:
            raise SystemExit(f"SWAP inconnu : {_SWAP}")
        n += 1
    print(f"[swap] {_SWAP} depuis {_DONOR.split('/')[-1]} sur {n} blocs "
          f"({path})", flush=True)
    return model, info


probe.load_frozen = _load_with_swap

if __name__ == "__main__":
    probe.main()
