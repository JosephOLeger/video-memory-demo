"""A/B job queue: one job per card at a time (per-card lock held for the job's whole life),
a card is used only when it has >= --need MiB free, card 2 only once queue3.sh has ended.
Optional gate: wait until a file exists (e.g. the v3c green light) before the first launch.

    nohup python queue_ab.py jobs_ab.txt --gate GO_AB > logs_ab/queue.log 2>&1 < /dev/null &
jobs file: one job per line, "name script args..." (logs in logs_ab/<name>.log). A job already
done (its log contains its end marker) or running is skipped.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/miniforge3/envs/titans/bin/python")
ENV = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=os.path.expanduser("~/work/titans-mac/src"),
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
MARKERS = ("TRAIN-AB-FINI", "PREP-AB-TRAIN-FINI", "PREP-AB-EVAL-FINI", "PREP-AB-FILL-FINI", "EARLY-STOP")


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return dict((int(i), int(m)) for i, m in (l.split(", ") for l in out.strip().splitlines()))


def queue3_alive():
    out = subprocess.run(["pgrep", "-f", "queue3.sh"], capture_output=True, text=True).stdout
    return bool(out.strip())


def run_job(card, lockf, name, args):
    logp = f"{HERE}/logs_ab/{name}.log"
    with open(logp, "a") as fo:
        fo.write(f"# queue_ab {time.strftime('%F %T')} card {card}: {' '.join(args)}\n")
        fo.flush()
        p = subprocess.Popen([PY] + args, cwd=HERE, stdout=fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=dict(ENV, CUDA_VISIBLE_DEVICES=str(card)))
        log(f"launched {name} on card {card} pid {p.pid}")
        rc = p.wait()
    log(f"ended {name} (card {card}, rc {rc})")
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--need", type=int, default=20480)
    ap.add_argument("--cards", default="0,1,2")
    ap.add_argument("--gate", default=None)
    a = ap.parse_args()
    os.makedirs(f"{HERE}/locks", exist_ok=True)
    if a.gate:
        while not os.path.exists(os.path.join(HERE, a.gate)):
            time.sleep(60)
        log(f"gate {a.gate} open")
    jobs = []
    for l in open(a.jobs):
        l = l.strip()
        if l and not l.startswith("#"):
            name, *args = l.split()
            jobs.append((name, args))
    threads = []
    for name, args in jobs:
        lp = f"{HERE}/logs_ab/{name}.log"
        if os.path.exists(lp) and any(m in open(lp).read() for m in MARKERS):
            log(f"skip {name} (done)")
            continue
        while True:
            fm = free_mib()
            got = None
            for c in [int(x) for x in a.cards.split(",")]:
                if c == 2 and queue3_alive():
                    continue
                if fm.get(c, 0) < a.need:
                    continue
                lf = open(f"{HERE}/locks/card{c}.lock", "w")
                try:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lf.close()
                    continue
                if free_mib().get(c, 0) < a.need:       # re-check under the lock
                    fcntl.flock(lf, fcntl.LOCK_UN)
                    lf.close()
                    continue
                got = (c, lf)
                break
            if got:
                break
            time.sleep(30)
        t = threading.Thread(target=run_job, args=(got[0], got[1], name, args))
        t.start()
        threads.append(t)
        time.sleep(180)               # let the model load before the next free-memory check
    for t in threads:
        t.join()
    log("queue empty")


if __name__ == "__main__":
    main()
