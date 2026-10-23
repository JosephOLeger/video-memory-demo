"""One MQAR diagnostic cell (seed x rank_cap), so the 5 remaining cells can
run in parallel across the tower's 3 cards instead of sequentially on one.
Same config as mqar_diag.py; writes one json per cell, merged afterwards."""
import json, time, argparse, torch
from titans.membench.tasks import associative_recall
from titans.membench.run import train_recall

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--rc", default="None")          # "None" or an int
args = ap.parse_args()
rc = None if args.rc == "None" else int(args.rc)

Q, STEPS = 8, 10000
torch.manual_seed(args.seed)                     # controls init only
t0 = time.time()
task_fn = (lambda s: associative_recall(
    batch=64, num_pairs=8, num_queries=Q, segment_size=8, seed=s))
r = train_recall(task_fn, steps=STEPS, lr=1e-3, dim=64, depth=2,
                 segment_size=8, num_heads=1, rank_cap=rc, log_every=2000)
row = dict(q=Q, rank_cap=rc, seed=args.seed, acc=round(r["acc_mean"], 4),
           acc_std=round(r["acc_std"], 4), chance=round(r["chance"], 4),
           train_loss=round(r["train_loss"], 4), diverged=r["diverged_at"],
           secs=round(time.time() - t0, 1))
out = f"/home/josephleger/work/iclr/mqar_diag_s{args.seed}_rc{args.rc}.json"
json.dump(row, open(out, "w"), indent=2)
print("DONEROW", json.dumps(row), flush=True)
