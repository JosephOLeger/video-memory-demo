"""21/10 : ecriture du passe avec recalcul par segments (gradient checkpointing).

Meme calcul que ``train_load.write_stream`` sans perte dense, mais la recurrence est decoupee en
segments de ``seg`` tokens : pendant l'aller on ne garde que l'etat (poids rapides + tampon) a
chaque frontiere de segment ; a la retropropagation, chaque segment est recalcule. La memoire
GPU passe de O(nombre d'ecritures) a O(nombre de segments + un segment), pour ~1 aller de plus.
Le gradient est identique (verifie par ``check``).
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
import inspect

from ..memory import NeuralMemory as _NM
_TAKES_X = "x" in inspect.signature(_NM._scan_chunked).parameters


def write_stream_ckpt(branch, z, chunk, seg):
    mem = branch.memory
    k, v, _q, al, et, th = mem._project(z)
    gam = mem._gamma(z)
    state, S = branch.init_carry(z.shape[0])
    ks, kS = sorted(state), sorted(S)
    seg = max(chunk, seg - seg % chunk)
    N = z.shape[1]

    def run(lo, hi, *flat):
        st = dict(zip(ks, flat[:len(ks)]))
        sS = dict(zip(kS, flat[len(ks):len(ks) + len(kS)]))
        kk, vv, aa, ee, tt, xx = flat[len(ks) + len(kS):len(ks) + len(kS) + 6]
        gg = flat[-1] if gam is not None else None
        kw = {"x": xx} if _TAKES_X else {}  # x n'existe que dans l'ancien code (tour)
        st, sS, _ = mem._scan_chunked(kk, vv, aa, ee, tt, st, sS, min(chunk, hi - lo), q=None, gamma=gg, **kw)
        return tuple(st[key] for key in ks) + tuple(sS[key] for key in kS)

    for lo in range(0, N, seg):
        hi = min(lo + seg, N)
        xs = [k[:, lo:hi], v[:, lo:hi], al[:, lo:hi], et[:, lo:hi], th[:, lo:hi], z[:, lo:hi]]
        if gam is not None:
            xs.append(gam[:, lo:hi])
        flat = [state[key] for key in ks] + [S[key] for key in kS] + xs
        if torch.is_grad_enabled():
            out = checkpoint(run, lo, hi, *flat, use_reentrant=True)
        else:
            out = run(lo, hi, *flat)
        state = dict(zip(ks, out[:len(ks)]))
        S = dict(zip(kS, out[len(ks):]))
    return state, S, None, (al.detach(), et.detach(), th.detach())


def check(branch, z, chunk, seg):
    """Gradient de la meme perte (lecture de l'etat final) avec et sans recalcul."""
    from titans.graft.train_load import write_stream
    out = []
    for f in (lambda: write_stream(branch, z, chunk), lambda: write_stream_ckpt(branch, z, chunk, seg)):
        branch.zero_grad()
        st, S, _, _ = f()
        loss = sum((t.float() ** 2).mean() for t in st.values()) + sum((t.float() ** 2).mean() for t in S.values())
        loss.backward()
        out.append((loss.item(), {n: p.grad.clone() for n, p in branch.named_parameters() if p.grad is not None}))
    (l0, g0), (l1, g1) = out
    err = max(((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)).item() for n in g0)
    return l0, l1, err, len(g0)
