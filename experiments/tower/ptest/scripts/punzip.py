import os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
Z = os.path.expanduser("~/work/iclr/ptest/dl/valid_videos.zip")
D = os.path.expanduser("~/work/iclr/ptest/data")
def work(k, n=16):
    z = zipfile.ZipFile(Z)
    c = 0
    for i, m in enumerate(z.infolist()):
        if i % n != k or m.is_dir():
            continue
        dst = os.path.join(D, m.filename)
        if os.path.exists(dst) and os.path.getsize(dst) == m.file_size:
            continue
        z.extract(m, D); c += 1
    return c
with ProcessPoolExecutor(16) as ex:
    print(sum(ex.map(work, range(16))))
