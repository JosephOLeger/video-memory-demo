# Video track

## Data

`fetch_finevideo.py` downloads 150 hours of FineVideo into `data/finevideo/` and removes every video that overlaps
LongVideoBench; `audit_finevideo.py` proves the purge. `cgb/` builds the CG-Bench items, targets and filters
(`download.sh` fetches the 326 GB of videos).

## Amputated past (November 2026)

1,219 videos, 256 frames each, cut into 16 clips. At a cut point, three passes of the frozen host: the teacher sees
all clips up to the cut; the window-only model sees the last clip; the student sees the last clip plus the memory
written from the cut clips. The memory is trained by KL towards the teacher on the teacher's own answer, on five
kinds of queries (anchored scene, generic past, FineVideo QA, free description, current scene).

```bash
paper_b/video/run_ctx.sh targets 0 4 0      # shard 0 of 4 on card 0 (one shard per card, ~7 h each)
paper_b/video/run_ctx.sh smoke 0
paper_b/video/run_ctx.sh arm ctx_u_s0 0 --weight none
```

Readings per evaluation: *recovered*, the share of the gap between window and teacher that the own memory closes;
*content*, the own memory against the other-video memory. Result so far: 16 to 18 % recovered, 2 to 7 % content,
the content largest on the current scene where the past cannot help; closed questions on the past stay at zero.
