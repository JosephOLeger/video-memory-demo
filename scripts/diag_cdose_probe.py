"""Dose de c au REJEU sur artefacts usq geles (mecanique vs appris).

Charge diag_pr_collapse_probe et force orthogonalize_clip = CLIP_OVERRIDE
(variable d'environnement) sur chaque bloc memoire apres load_frozen.
Au rejeu, baisser c sature psi (toutes les ecritures pleine taille =
unit mecanique) ; monter c les amortit proportionnellement.

Prediction P-CD-1 : pr essentiellement PLAT en c-override — la diversite
directionnelle des ecritures d'usq est APPRISE, pas fabriquee par psi au
rejeu. Si baisser c a 35 reproduit le collapse d'unit, le levier est
mecanique — a consigner tel quel.

Usage : CLIP_OVERRIDE=35 diag_cdose_probe.py <args de la sonde>.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "diag_pr_collapse_probe.py"),
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

_C = float(os.environ["CLIP_OVERRIDE"])
_orig_load = probe.load_frozen


def _load_with_c(path, device):
    model, info = _orig_load(path, device)
    n = 0
    for blk in model.blocks:
        mem = getattr(blk, "memory", None)
        if mem is not None and hasattr(mem, "orthogonalize_clip"):
            mem.orthogonalize_clip = _C
            n += 1
    print(f"[cdose] orthogonalize_clip={_C} sur {n} blocs ({path})",
          flush=True)
    return model, info


probe.load_frozen = _load_with_c

if __name__ == "__main__":
    probe.main()
