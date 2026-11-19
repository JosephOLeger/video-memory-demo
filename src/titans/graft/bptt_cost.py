"""Chiffrer le cout BPTT : combien d'ecritures differentiables tiennent en VRAM ?

Phase 2 retropropage a travers la passe d'ecriture (la boucle interne en ligne),
dont le cout du graphe croit avec le nombre d'ecritures gardees (= la profondeur de
l'aiguille, bptt-window-severs-echo). Ce micro-benchmark le mesure directement :
l'hote gele est resident (baseline realiste), et on balaie N = nombre de tokens
ecrits dans le graphe, en relevant le pic VRAM de update + read + backward.

On NE fait PAS tourner l'hote dans la boucle (la passe d'ecriture se fait sur des
etats couche-24 pre-calcules en Phase 2, donc le forward 3B n'est pas dans ce
graphe). On isole donc le cout variable : le graphe de la boucle interne. A ajouter
a part : ~2 Go pour la passe de reponse (hote sur la fenetre), et les etats
d'optimiseur si on entraine la tete (full head Adam ~2.5 Go).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m titans.graft.bptt_cost \
        --d-mem 128 --chunk 8 --ns 128,256,512,1024,2048,4096,8192
"""
from __future__ import annotations

import argparse

import torch

from .branch import MemoryBranch
from .qwen import default_layer, find_decoder_layers


def gb(x: int) -> float:
    return x / 2**30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--d-mem", type=int, default=128, dest="d_mem")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--window", type=int, default=16,
                    help="tokens de lecture (la fenetre), negligeable")
    ap.add_argument("--ns", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--no-host", action="store_true", dest="no_host",
                    help="ne pas charger l'hote (mesure la boucle interne seule)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ns = [int(x) for x in a.ns.split(",") if x]
    dev = a.device

    d_model = 2048
    if not a.no_host:
        from .check_qwen import load_host
        model = load_host(a.model, getattr(torch, a.dtype), dev)
        for p in model.parameters():
            p.requires_grad_(False)                # hote gele
        d_model = int(model.config.text_config.hidden_size)
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        print(f"hote resident : {gb(base):.2f} Go (baseline)", flush=True)
    else:
        base = 0
        print("hote non charge : boucle interne seule", flush=True)

    branch = MemoryBranch(d_model, a.d_mem, chunk_size=a.chunk).to(dev)
    branch.train()
    n_tr = sum(p.numel() for p in branch.trainable_parameters())
    print(f"branche : {n_tr} parametres entrainables ; d_mem={a.d_mem} "
          f"chunk={a.chunk}", flush=True)
    free, total = torch.cuda.mem_get_info(dev)
    print(f"VRAM {gb(total):.1f} Go ({gb(free):.1f} libres apres l'hote)\n", flush=True)

    print(f"{'N ecritures':>12}{'pic VRAM':>12}{'dont graphe':>14}{'Go/1k ecr.':>12}")
    qz = torch.randn(1, a.window, a.d_mem, device=dev)
    for N in ns:
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.empty_cache()
            state, S = branch.init_carry(1)
            z = torch.randn(1, N, a.d_mem, device=dev)
            state, S = branch.memory.update(z, state, S, min(a.chunk, N))
            r = branch.memory.retrieve(state, branch.memory.q_proj(qz))
            if branch.norm_r is not None:
                r = branch.norm_r(r)
            loss = branch.up(r).float().pow(2).mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated(dev)
            graph = peak - base
            print(f"{N:>12}{gb(peak):>11.2f}G{gb(graph):>13.2f}G"
                  f"{1000 * gb(graph) / N:>11.3f}G", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{N:>12}   OOM -- plafond entre le precedent et {N}", flush=True)
                torch.cuda.empty_cache()
                break
            raise
        finally:
            for v in ("state", "S", "z", "r", "loss"):
                if v in dir():
                    pass
            branch.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    print("\nBPTT-COST-FINIE", flush=True)


if __name__ == "__main__":
    main()
