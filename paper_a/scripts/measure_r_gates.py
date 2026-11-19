"""Gates appris de la pile lucidrains : eta (to_momentum), alpha
(to_decay_factor), theta/lr (to_adaptive_step) mesures PENDANT un vrai
forward sur enwik8 val.

Question (transfert du mecanisme) : leur baseline entrainee a-t-elle
aussi tue son gate de momentum, et le reentrainement psi_c (R2b2)
l'a-t-il garde vivant ? Symetrique de diag_gates_shadow sur notre pile :
on hooke les trois tetes de chaque NeuralMemory, sigmoide appliquee
comme dans leur forward (adaptive_step passe par leur propre
adaptive_step_transform).

Usage : measure_r_gates.py --ckpt ckpt_prlaw_final.pt [--psic] \
        --seqs 4 --seq-len 16384 --out GATES_r_baseline.jsonl
(cwd = ~/repro/titans-pytorch pour ./data/enwik8.gz)
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
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


def stats(v, eta=False):
    v = torch.cat([t.float().flatten() for t in v])
    q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9]))
    row = {"mean": float(v.mean()), "p10": float(q[0]),
           "med": float(q[1]), "p90": float(q[2]), "n": int(v.numel())}
    if eta:
        row["frac_lt_0.9"] = float((v < 0.9).float().mean())
        row["frac_gt_0.5"] = float((v > 0.5).float().mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--psic", action="store_true",
                    help="active la regle psi_c au rejeu (ckpt psic2)")
    ap.add_argument("--enwik", default="./data/enwik8.gz")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16384, dest="seq_len")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.psic:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import patch_psic
        patch_psic.install()

    model = build_model().cuda()
    blob = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    mems = [m for m in model.modules() if isinstance(m, NeuralMemory)]
    acc = {}  # (mem_idx, gate) -> list of tensors

    def hook(mi, gate, transform):
        def h(mod, inputs, output):
            acc.setdefault((mi, gate), []).append(
                transform(output.detach()).float().cpu())
        return h

    handles = []
    for mi, m in enumerate(mems):
        handles.append(m.to_momentum.register_forward_hook(
            hook(mi, "eta", torch.sigmoid)))
        handles.append(m.to_decay_factor.register_forward_hook(
            hook(mi, "alpha", torch.sigmoid)))
        handles.append(m.to_adaptive_step.register_forward_hook(
            hook(mi, "theta", m.adaptive_step_transform)))

    with gzip.open(a.enwik) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    val = torch.from_numpy(data[int(90e6):])

    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for s in range(a.seqs):
            start = int(torch.randint(0, len(val) - a.seq_len - 1, (1,),
                                      generator=g))
            seq = val[start:start + a.seq_len].long().cuda()[None]
            model(seq, return_loss=False)

    rows = []
    for (mi, gate) in sorted(acc):
        row = {"ckpt": os.path.basename(a.ckpt), "psic": bool(a.psic),
               "mem": mi, "gate": gate,
               **stats(acc[(mi, gate)], eta=(gate == "eta"))}
        rows.append(row)
        extra = (f" frac<0.9 {row['frac_lt_0.9']:.3f}"
                 if gate == "eta" else "")
        print(f"mem{mi} {gate:5s} moy {row['mean']:.3e} "
              f"p10 {row['p10']:.3e} med {row['med']:.3e} "
              f"p90 {row['p90']:.3e}{extra}", flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("R-GATES-FINIES", flush=True)


if __name__ == "__main__":
    main()
