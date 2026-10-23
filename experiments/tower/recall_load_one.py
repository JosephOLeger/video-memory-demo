"""Capacity / learnability probe cell. One cell = (load n, rank_cap, seed, lr).
rank_cap=1 == hard rank cut at er~1 (collapsed); None == sane (full-rank deep
memory). Trains to convergence (ceiling/plateau/max_steps). Records best_step so
convergence SPEED is comparable: the collapsed arm learns faster; the sane arm
learns slowly or (seed-dependent) not at all. Compare capacity only among cells
that actually LEARNED the task (train_loss low)."""
import json, time, argparse, torch
from titans.membench.tasks import associative_recall
from titans.membench.run import train_recall_converge

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--rc", default="None")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--valv", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--maxsteps", type=int, default=15000)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--tag", default="load")
args = ap.parse_args()
rc = None if args.rc == "None" else int(args.rc)
n = args.n
kv = max(n, 16)

torch.manual_seed(args.seed)
t0 = time.time()
task_fn = (lambda s: associative_recall(
    batch=64, num_pairs=n, num_queries=n, segment_size=8,
    key_vocab=kv, val_vocab=args.valv, seed=s))
r = train_recall_converge(
    task_fn, lr=args.lr, dim=64, depth=2, segment_size=8, num_heads=1,
    rank_cap=rc, eval_every=500, min_steps=2000, max_steps=args.maxsteps,
    ceil=0.95, ceil_patience=3, plateau_patience=args.patience, log_every=2000)
row = dict(tag=args.tag, n=n, q=n, key_vocab=kv, val_vocab=args.valv, lr=args.lr,
           rank_cap=rc, seed=args.seed,
           acc=round(r["acc_mean"], 4), acc_std=round(r["acc_std"], 4),
           chance=round(r["chance"], 4), best=round(r.get("best", r["acc_mean"]), 4),
           best_step=r.get("best_step"), stopped=r.get("stopped"),
           train_loss=round(r["train_loss"], 4), diverged=r.get("diverged_at"),
           secs=round(time.time() - t0, 1))
lrtag = f"{args.lr:.0e}".replace("-0", "-")
out = f"/home/josephleger/work/iclr/{args.tag}_n{n}_rc{args.rc}_s{args.seed}_lr{lrtag}.json"
json.dump(row, open(out, "w"), indent=2)
print("DONEROW", json.dumps(row), flush=True)
