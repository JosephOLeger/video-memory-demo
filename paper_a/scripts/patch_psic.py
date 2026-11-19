"""Greffe de la regle de magnitude usq dans lucidrains/titans-pytorch.

Leur pipeline par poids (neural_memory.py store_memories) :
  surprise -> scan momentum -> newtonschulz5(update) -> scan decay.
Leur NEURAL_MEM_SPEC_NORM_SURPRISES aplatit la magnitude comme notre
`unit` — et leur stack collapse en rang quand meme (volet R : pr 63.7 ->
1-3 en ~1150 ecritures). La greffe transplante NOTRE levier : la sortie
NS est multipliee par psi_c(||update pre-NS||) = (2/pi) arctan(pi n / 2c)
par ecriture (dims [bh, n] preservees, norme sur les dims matricielles).

c est auto-calibre : EMA (0.99) de la mediane des normes pre-NS, par
FORME de poids — la regle anti-piege-du-c (psi vaut 0.64 a la mediane
courante, mi-transition, quelle que soit l'echelle du tenseur). Les
tenseurs sans structure matricielle passent inchanges, comme chez nous.

S'installe AVANT la construction du modele :
    import patch_psic; patch_psic.install()
Meme patch pour les deux volets : R2a (inference seule, ckpt existant)
et R2b (reentrainement complet avec la regle active).
"""
import math

import torch

import titans_pytorch.neural_memory as nm

_ema = {}
_orig_ns5 = None


def _usq_ns5(t, *args, **kwargs):
    u = _orig_ns5(t, *args, **kwargs)
    if t.dim() < 4:
        # pas de structure [bh, n, d1, d2] : biais/vecteurs inchanges,
        # comme les biais qui traversent NS-5 tels quels chez nous.
        return u
    dims = tuple(range(2, t.dim()))
    n = t.norm(dim=dims, keepdim=True)
    key = tuple(t.shape[2:])
    med = float(n.detach().median())
    c = _ema.get(key)
    c = med if c is None else 0.99 * c + 0.01 * med
    _ema[key] = c
    if not (c > 0.0):
        return u
    return u * (2.0 / math.pi) * torch.atan(n * (math.pi / (2.0 * c)))


def install():
    global _orig_ns5
    if _orig_ns5 is not None:
        return
    _orig_ns5 = nm.newtonschulz5
    nm.newtonschulz5 = _usq_ns5
    print("[psic] regle de magnitude usq installee sur newtonschulz5",
          flush=True)
