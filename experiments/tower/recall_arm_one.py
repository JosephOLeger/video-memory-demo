"""Real-arm recall probe (Q1 functional cost + Q4 psi_c mechanism).

One cell = (load n, arm mode, clip c, eta_init, alpha_init, seed). Uses the REAL
TitansMACBlock write rule via block_kwargs (NOT rank_cap):
  mode='unit'         NS, magnitude discarded            -> collapse-prone (ATLAS-style)
  mode='unit_squash'  NS + psi_c arctan coupling (Josep) -> healthy
  mode='clip'         NS + hard cap at c                 -> same ceiling, no adaptive coupling (Q4)
  mode='raw'          no orthogonalisation (unbounded)   -> divergence witness
NEVER 'rescale'/True -- purged. Compare arms at MATCHED c, eta_init, alpha_init.
Records recall acc (final + best) and train_loss; capacity/mechanism read off the
acc-vs-load and arm contrast. Same geometry/seeding contract as recall_load_one.py."""
import json, time, argparse, torch
from titans.membench.tasks import associative_recall
from titans.membench.run import train_recall_converge

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--mode", default="unit")            # unit | unit_squash | clip | raw
ap.add_argument("--clip", type=float, default=1.0)   # scale c for bounded modes
ap.add_argument("--eta-init", type=float, default=None, dest="eta_init")
ap.add_argument("--alpha-init", type=float, default=None, dest="alpha_init")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--valv", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--maxsteps", type=int, default=50000)
ap.add_argument("--patience", type=int, default=40)
ap.add_argument("--tag", default="arm")
args = ap.parse_args()
n = args.n
kv = max(n, 16)
mode = False if args.mode == "raw" else args.mode

bk = dict(memory_orthogonalize_momentum=mode, memory_orthogonalize_clip=args.clip)
if args.eta_init is not None:
    bk["eta_init"] = args.eta_init
if args.alpha_init is not None:
    bk["alpha_init"] = args.alpha_init

torch.manual_seed(args.seed)
t0 = time.time()
task_fn = (lambda s: associative_recall(
    batch=64, num_pairs=n, num_queries=n, segment_size=8,
    key_vocab=kv, val_vocab=args.valv, seed=s))
r = train_recall_converge(
    task_fn, lr=args.lr, dim=64, depth=2, segment_size=8, num_heads=1,
    eval_every=500, min_steps=2000, max_steps=args.maxsteps,
    ceil=0.95, ceil_patience=3, plateau_patience=args.patience, log_every=2000,
    **bk)
row = dict(tag=args.tag, n=n, q=n, key_vocab=kv, val_vocab=args.valv, lr=args.lr,
           mode=args.mode, clip=args.clip, eta_init=args.eta_init,
           alpha_init=args.alpha_init, seed=args.seed,
           acc=round(r["acc_mean"], 4), acc_std=round(r["acc_std"], 4),
           chance=round(r["chance"], 4), best=round(r.get("best", r["acc_mean"]), 4),
           best_step=r.get("best_step"), stopped=r.get("stopped"),
           train_loss=round(r["train_loss"], 4), diverged=r.get("diverged_at"),
           secs=round(time.time() - t0, 1))
out = f"/home/josephleger/work/iclr/{args.tag}_n{n}_{args.mode}_c{args.clip}_s{args.seed}.json"
json.dump(row, open(out, "w"), indent=2)
print("DONEROW", json.dumps(row), flush=True)
