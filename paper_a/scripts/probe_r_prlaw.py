"""Volet R : la sonde de loi pr sur la pile lucidrains (titans-pytorch).

Reconstruit le modele de train_r_prlaw.py, charge un checkpoint, feed des
sequences enwik8 val LONGUES (16384 octets -> 128 updates memoire par
couche), et capture par hook sur NeuralMemory.forward les etats de poids
par mini-batch (state.updates). Pour chaque couche a memoire (2, 4, 6) et
le plus grand poids 2D du MemoryMLP :

  U_t = W_t - W_{t-1}, stats identiques a diag_pr_collapse_probe
  (normes, r_eff, q_d/p_d), trajectoire pr(W_t), et loi DC
  pr(W0 + L m) avec W0 = les poids appris du MemoryMLP.

Sortie JSONL compatible avec analyze_pr_law (champs identiques).

Usage : probe_r_prlaw.py --ckpt ckpt_prlaw_1000.pt --seqs 3 \
        --seq-len 16384 --diag-out results_prlaw_r.jsonl
"""
import argparse
import gzip
import json
import math

import numpy as np
import torch

from titans_pytorch import (
    MemoryAsContextTransformer, MemoryMLP,
)
from titans_pytorch.neural_memory import NeuralMemory


def build_model():
    neural_memory_model = MemoryMLP(dim=64, depth=4)
    return MemoryAsContextTransformer(
        num_tokens=256, dim=384, depth=8, segment_len=32,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(2, 4, 6),
        neural_memory_segment_len=4, neural_memory_batch_size=128,
        neural_mem_gate_attn_output=False, neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False, sliding_window_attn=True,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=64, heads=4, attn_pool_chunks=True, qk_rmsnorm=True,
            momentum=True, momentum_order=1,
            default_step_transform_max_lr=1e-1, use_accelerated_scan=False,
            per_parameter_lr_modulation=True, spectral_norm_surprises=True,
            store_with_lookahead_value=False,
        ),
    )


def pr_of_np(M):
    s2 = np.linalg.svd(M, compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--enwik", default="./data/enwik8.gz")
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=16384, dest="seq_len")
    ap.add_argument("--max-lag", type=int, default=48, dest="max_lag")
    ap.add_argument("--diag-out", required=True, dest="out")
    a = ap.parse_args()

    model = build_model().cuda()
    blob = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    with gzip.open(a.enwik) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    val = torch.from_numpy(data[int(90e6):])

    # hooks : capture des etats de poids par mini-batch pour chaque memoire
    mems = [m for m in model.modules() if isinstance(m, NeuralMemory)]
    grabbed = [[] for _ in mems]

    def make_wrap(orig, idx):
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            state = out[1]
            upd = state.updates
            if upd is not None and len(list(upd.keys())) > 0:
                keys = [k for k in upd.keys()
                        if hasattr(upd[k], "numel") and upd[k].ndim >= 3]
                if keys:
                    key = max(keys, key=lambda k: upd[k][0].numel())
                    grabbed[idx].append((key, upd[key].detach().float().cpu()))
            return out
        return wrapped

    for i, m in enumerate(mems):
        m.forward = make_wrap(m.forward, i)

    g = torch.Generator().manual_seed(0)
    rows = []
    with torch.no_grad():
        for s in range(a.seqs):
            for gr in grabbed:
                gr.clear()
            start = int(torch.randint(0, len(val) - a.seq_len - 1, (1,),
                                      generator=g))
            seq = val[start:start + a.seq_len].long().cuda()[None]
            model(seq, return_loss=False)
            for mi, gr in enumerate(grabbed):
                if not gr:
                    continue
                key = gr[0][0]
                # updates : [bh, n_upd, ...] -> serie temporelle de poids
                Ws = torch.cat([t for _, t in gr], dim=1)[0]  # tete 0
                Ws = Ws.reshape(Ws.shape[0], Ws.shape[-2], Ws.shape[-1])
                W = Ws.numpy().astype(np.float64)
                # init = les poids appris du memory model (meme cle)
                sd = mems[mi].memory_model.state_dict()
                k0 = key.split(".")[-1]
                Wi = None
                for kk, vv in sd.items():
                    if vv.ndim >= 2 and vv.shape[-2:] == tuple(W.shape[-2:]):
                        if kk.endswith(k0) or Wi is None:
                            Wi = vv.detach().float().cpu().numpy().astype(np.float64)
                series = np.concatenate([Wi[None], W], axis=0)
                U = np.diff(series, axis=0)
                L = U.shape[0]
                n = np.linalg.norm(U.reshape(L, -1), axis=1)
                keep = n > 1e-12
                Un = U[keep] / n[keep, None, None]
                Lk = Un.shape[0]
                sv2 = np.linalg.svd(Un, compute_uv=False) ** 2
                r_eff = sv2.sum(1) ** 2 / (sv2 ** 2).sum(1)
                D = min(a.max_lag, Lk - 1)
                q_d, p_d = [], []
                Ut = torch.from_numpy(Un)
                for dd in range(D + 1):
                    A, B = Ut[: Lk - dd], Ut[dd:]
                    q_d.append(float(torch.einsum("tij,tik->tjk", A, B)
                                     .flatten(1).pow(2).sum(1).mean()))
                    p_d.append(float(torch.einsum("tij,tkj->tik", A, B)
                                     .flatten(1).pow(2).sum(1).mean()))
                m_ = U.mean(0)
                pr_traj = [pr_of_np(series[j]) for j in
                           range(0, len(series), max(1, len(series) // 40))]
                pr_dc = [pr_of_np(Wi + Lc * m_) for Lc in
                         range(1, L + 1, max(1, L // 40))]
                rows.append({
                    "frozen": f"{a.ckpt}::mem{mi}::{key}", "block": mi,
                    "mode": f"lucidrains-seq{s}", "alpha_override": None,
                    "clips": a.seqs, "stream_seed": s,
                    "w1_shape": list(W.shape[-2:]), "n_writes": int(Lk),
                    "stopped": None,
                    "write_norm_mean": float(n.mean()),
                    "write_norm_last50_mean": float(n[-50:].mean()),
                    "r_eff_mean": float(r_eff.mean()),
                    "r_eff_median": float(np.median(r_eff)),
                    "q_d": q_d, "p_d": p_d,
                    "init_norm": float(np.linalg.norm(Wi)),
                    "init_pr": pr_of_np(Wi),
                    "m_norm": float(np.linalg.norm(m_)),
                    "m_pr": pr_of_np(m_),
                    "pr_traj_series": pr_traj, "pr_dc": pr_dc,
                    "pr_final": pr_of_np(series[-1]),
                })
                print(f"seq {s} mem{mi} {key}: L={Lk} "
                      f"r_eff={r_eff.mean():.1f} pr_fin={rows[-1]['pr_final']:.1f} "
                      f"pr_dc_fin={pr_dc[-1]:.1f} init_pr={rows[-1]['init_pr']:.1f}",
                      flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("R-PRLAW-PROBE-FINIE", flush=True)


if __name__ == "__main__":
    main()
