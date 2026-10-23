"""MQAR diagnostic: is the full-rank arm's q=8 failure an under-training artefact?

Scout (1 seed, 2500 steps) reversed the prediction: at q=8 the full-rank memory
fell to chance (acc 0.069, train_loss 2.77) while rank-1 held (0.257, loss 1.81).
That gap in TRAIN loss says the higher-capacity arm simply did not train on the
short budget -- not a capacity truth. Here we settle it: q=8 (the decisive load),
None vs rank_cap=1, 5 model-init seeds, 10k steps (the campaign budget).

Seeding: train_recall has no seed arg -- the model inits on the ambient RNG, and
data is seeded by the step index inside task_fn (so identical across seeds). So
torch.manual_seed(seed) before each call varies ONLY the model init on fixed data.

Read: if the full-rank train_loss is now LOW and acc still < rank-1 -> genuine
capacity result (diverse worse -> drop the lever). If train_loss is still HIGH ->
still under-trained, go to more steps. Loop is seed-outer/arm-inner so seed 0 of
both arms lands first (~1 h) as an early read.
"""
import json, time, torch
from titans.membench.tasks import associative_recall
from titans.membench.run import train_recall

Q = 8
RCS = [None, 1]          # None = diverse (full rank), 1 = collapsed (rank-1)
SEEDS = [0, 1, 2, 3, 4]
STEPS = 10000
OUT = "/home/josephleger/work/iclr/mqar_diag.json"

rows = []
for seed in SEEDS:
    for rc in RCS:
        torch.manual_seed(seed)          # controls model init only; data is step-seeded
        t0 = time.time()
        task_fn = (lambda s: associative_recall(
            batch=64, num_pairs=8, num_queries=Q, segment_size=8, seed=s))
        r = train_recall(task_fn, steps=STEPS, lr=1e-3, dim=64, depth=2,
                         segment_size=8, num_heads=1, rank_cap=rc, log_every=1000)
        row = dict(q=Q, rank_cap=rc, seed=seed, acc=round(r["acc_mean"], 4),
                   acc_std=round(r["acc_std"], 4), chance=round(r["chance"], 4),
                   train_loss=round(r["train_loss"], 4), diverged=r["diverged_at"],
                   secs=round(time.time() - t0, 1))
        rows.append(row)
        print(f"ROW seed={seed} rank_cap={str(rc):>4} acc={row['acc']:.3f}"
              f"+-{row['acc_std']:.3f} loss={row['train_loss']:.3f} {row['secs']}s",
              flush=True)
        json.dump(rows, open(OUT, "w"), indent=2)
print("DONE", OUT, flush=True)
