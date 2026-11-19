"""Sonde de la loi pr sur artefacts geles — mesure les ecritures reelles.

Rejoue un flux (loop/cycle/shuffle, carry longue duree) sur un artefact
gele et, par monkeypatch de ``mem.update``, capture w1 apres CHAQUE appel
d'update, pour TOUS les blocs. Les ecritures U_t = w1_t - w1_{t-1}
(la decroissance (1-alpha) est negligeable a l'echelle d'un appel avec
alpha ~ 1e-4 ; sous --alpha-override elle est corrigee explicitement).

Sorties par (artefact, bloc), en JSONL :
  - trajectoire pr(L) mesuree (une valeur par clip),
  - q_d = E_t ||Un_t^T Un_{t+d}||_F^2 (recouvrement cote cles/lignes),
  - p_d = E_t ||Un_t Un_{t+d}^T||_F^2 (cote valeurs/colonnes),
  - r_eff par ecriture (participation du spectre), normes, alpha
    effectif estime, forme de w1.

La prediction pr(L) par la formule a somme finie se fait hors ligne
(scripts/analyze_pr_law.py) — la sonde ne fait que mesurer.

Predictions pre-enregistrees : results/predictions_collapse_2026-08-28.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import torch

from titans.worldmodel.frozen import load_frozen
from titans.worldmodel.train import load_latent_cache
from titans.worldmodel.long_run import build_stream, state_diagnostics


def pr_of(sv2: torch.Tensor) -> float:
    return float(sv2.sum() ** 2 / (sv2 ** 2).sum())


@torch.no_grad()
def probe_one(path: str, va: torch.Tensor, a) -> list[dict]:
    model, info = load_frozen(path, a.device)
    model.eval()
    nb = len(model.blocks)
    snaps: list[list[torch.Tensor]] = [[] for _ in range(nb)]

    for bi, blk in enumerate(model.blocks):
        mem = blk.memory
        orig = mem.update

        def wrapped(x, state, S, chunk_size, _o=orig, _bi=bi):
            state, S = _o(x, state, S, chunk_size)
            snaps[_bi].append(state["w1"][0].detach().float().cpu().clone())
            return state, S

        mem.update = wrapped
        if a.alpha_override is not None:
            mem.alpha_override = a.alpha_override

    w1_init = [blk.memory.init_memory(1)[0]["w1"][0].float().cpu().clone()
               for blk in model.blocks]

    clips = build_stream(va, reps=a.clips, mode=a.mode, seed=a.stream_seed,
                         pool=a.pool)
    carry = None
    pr_traj: list[dict] = []
    stopped = None
    for r, z in enumerate(clips):
        pred, carry = model(z.to(a.device), carry=carry, return_carry=True)
        if not bool(torch.isfinite(pred).all()):
            stopped = f"pred non finie au clip {r}"
        d = state_diagnostics(model, carry)
        pr_traj.append({"clip": r,
                        **{f"pr_b{i}": d[f"b{i}"]["pr"] for i in range(nb)},
                        **{f"w1n_b{i}": d[f"b{i}"]["w1_norm"] for i in range(nb)}})
        if stopped or any(not d[f"b{i}"]["finite"] for i in range(nb)):
            stopped = stopped or f"etat non fini au clip {r}"
            break

    rows = []
    for bi in range(nb):
        if not snaps[bi]:
            continue
        W = torch.stack([w1_init[bi]] + snaps[bi])          # [L+1, d1, d2]
        U = W[1:] - W[:-1]
        if a.alpha_override:
            # alpha s'applique PAR TOKEN dans _step : sur un appel de C tokens,
            # w1_t = (1-alpha)^C w1_{t-1} + U_t ; on retire la part decroissance.
            U = W[1:] - (1 - a.alpha_override) ** a.tokens_per_call * W[:-1]
        U = U.to(a.device)
        norms = U.flatten(1).norm(dim=1)
        keep = norms > 0
        Un = (U[keep] / norms[keep, None, None])
        L = Un.shape[0]
        sv = torch.linalg.svdvals(Un)                        # [L, min(d1,d2)]
        sv2 = sv ** 2
        r_eff = (sv2.sum(1) ** 2 / (sv2 ** 2).sum(1))        # [L]
        D = min(a.max_lag, L - 1)
        q_d, p_d = [], []
        for d in range(D + 1):
            A, B = Un[: L - d], Un[d:]
            q = torch.einsum("tij,tik->tjk", A, B).flatten(1).pow(2).sum(1)
            p = torch.einsum("tij,tkj->tik", A, B).flatten(1).pow(2).sum(1)
            q_d.append(float(q.mean()))
            p_d.append(float(p.mean()))
        # L'etat part de l'init APPRIS W_init (norme ~14, pr eleve) qui ne
        # decroit jamais a alpha ~1e-4 : la prediction pr(L) doit inclure
        # les moments croises init <-> ecritures.
        Wi = w1_init[bi].to(a.device)
        # --- loi DC (decouverte du 28/08, proto_v2_dump.py) : la moyenne
        # des ecritures m, accumulee en rampe coherente, reproduit pr(L)
        # exactement : pr(L) = pr(dec^L W_i + (somme des poids) m).
        m = U.mean(0)
        m_norm = float(m.flatten().norm())
        m_hat = m / max(m_norm, 1e-12)
        sv_m = torch.linalg.svdvals(m) ** 2
        dec = (1 - (a.alpha_override or 0.0)) ** a.tokens_per_call
        pr_dc = []
        for Lc in range(8, U.shape[0] + 1, 8):
            wsum = Lc if dec == 1.0 else (1 - dec ** Lc) / (1 - dec)
            A = (dec ** Lc) * Wi + wsum * m
            sva = torch.linalg.svdvals(A) ** 2
            pr_dc.append(round(pr_of(sva), 3))
        win = Wi / Wi.norm()
        svi = torch.linalg.svdvals(Wi) ** 2
        cos_init = torch.einsum("ij,tij->t", win, Un)          # signe garde
        qi = torch.einsum("ij,tik->tjk", win, Un).flatten(1).pow(2).sum(1)
        pi = torch.einsum("ij,tkj->tik", win, Un).flatten(1).pow(2).sum(1)
        rows.append({
            "frozen": path, "block": bi, "mode": a.mode,
            "alpha_override": a.alpha_override, "clips": a.clips,
            "stream_seed": a.stream_seed, "w1_shape": list(W.shape[1:]),
            "n_writes": int(L), "stopped": stopped,
            "write_norm_mean": float(norms.mean()),
            "write_norm_last50_mean": float(norms[-50:].mean()),
            "r_eff_mean": float(r_eff.mean()),
            "r_eff_median": float(r_eff.median()),
            "q_d": q_d, "p_d": p_d,
            "init_norm": float(Wi.norm()), "init_pr": pr_of(svi),
            "qi_mean": float(qi.mean()), "pi_mean": float(pi.mean()),
            "m_norm": m_norm, "m_pr": pr_of(sv_m),
            "m_cos_init": float((m_hat * (Wi / Wi.norm())).sum()),
            "pr_dc": pr_dc,
            "norms": [round(float(x), 5) for x in norms],
            "cos_init": [round(float(x), 6) for x in cos_init],
            "cos_m": [round(float(x), 6) for x in
                      torch.einsum("ij,tij->t", m_hat, Un)],
            "pr_final": pr_traj[-1][f"pr_b{bi}"] if pr_traj else None,
        })
    rows.append({"frozen": path, "traj": pr_traj, "mode": a.mode,
                 "alpha_override": a.alpha_override})
    if a.dump_writes:
        base = a.dump_writes + "_" + path.split("/")[-1].replace(".pt", "")
        torch.save({"w1_init": w1_init,
                    "snaps": [torch.stack(s) if s else None for s in snaps]},
                   base + f"_{a.mode}_a{a.alpha_override or 0}.pt")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--mode", default="cycle",
                    choices=["loop", "cycle", "shuffle"])
    ap.add_argument("--clips", type=int, default=100)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--stream-seed", type=int, default=0, dest="stream_seed")
    ap.add_argument("--alpha-override", type=float, default=None,
                    dest="alpha_override")
    ap.add_argument("--tokens-per-call", type=int, default=8,
                    dest="tokens_per_call",
                    help="tokens par appel d'update (= segment_size)")
    ap.add_argument("--max-lag", type=int, default=64, dest="max_lag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-writes", default="", dest="dump_writes")
    ap.add_argument("--diag-out", required=True, dest="out")
    a = ap.parse_args()

    va, _ = load_latent_cache(a.val)
    files = sorted(glob.glob(a.frozen))
    if not files:
        raise SystemExit(f"aucun artefact ne matche {a.frozen}")
    with open(a.out, "a") as f:
        f.write(json.dumps({"header": True, "frozen": a.frozen,
                            "mode": a.mode, "clips": a.clips,
                            "alpha_override": a.alpha_override}) + "\n")
        for path in files:
            print(f"--- {path}", flush=True)
            try:
                rows = probe_one(path, va, a)
            except Exception as e:  # un artefact casse ne tue pas la serie
                rows = [{"frozen": path, "error": repr(e)}]
                print(f"    ERREUR {e!r}", flush=True)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
    print("PRLAW-PROBE-FINIE", flush=True)


if __name__ == "__main__":
    main()
