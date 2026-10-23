"""Run a script once its input exists AND a card has >= 12 GiB free (checked right before start).

    python wait_then.py '<glob that must match>' script.py args...
Card choice is serialised between my jobs by a lock held 150 s (time for the model to load).
Other users' queues may grab the same free memory at the same second: a run that dies in its
first 300 s (OOM while loading) is retried on the next free card.
"""
import fcntl
import glob
import os
import subprocess
import sys
import threading
import time

NEED_MIB = 12288
pat, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
while not glob.glob(os.path.expanduser(pat)):
    time.sleep(30)
lock = open(os.path.expanduser("~/work/iclr/graft_v3/logs/.card.lock"), "w")
for attempt in range(50):
    while True:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = dict((int(i), int(m)) for i, m in (l.split(", ") for l in out.strip().splitlines()))
        ok = [i for i, m in free.items() if m >= NEED_MIB]
        if ok:
            break
        fcntl.flock(lock, fcntl.LOCK_UN)
        time.sleep(30)
    print(f"{time.strftime('%T')} {script} attempt {attempt}: card {ok[0]} ({free[ok[0]]} MiB free)", flush=True)
    threading.Timer(150, lambda: fcntl.flock(lock, fcntl.LOCK_UN)).start()
    t0 = time.time()
    rc = subprocess.run([sys.executable, script] + args,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(ok[0]))).returncode
    if rc == 0 or time.time() - t0 > 300:
        sys.exit(rc)
    print(f"{time.strftime('%T')} died after {time.time() - t0:.0f} s (rc {rc}): retry", flush=True)
    time.sleep(60)
