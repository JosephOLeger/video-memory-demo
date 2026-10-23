import sys; sys.path.insert(0, "snap1")
import gen_scenes as G, ceiling as C
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
for fam, g, v in [("init_count", 0, 0), ("init_count", 0, 1), ("sign_number", 5, 3)]:
    it = G.make_group(fam, g)[v]
    fr = G.render(it["episode"])
    for name, x in [("full", fr), ("last8", fr[-8:])]:
        inp = C.build_inputs(proc, x, it["prompt"], G.FPS)
        print(fam, v, name, "T=", x.shape[0], "grid_thw=", inp["video_grid_thw"].tolist(),
              "sec_per_grid=", inp.get("second_per_grid_ts"), "ntok=", inp["input_ids"].shape[1])
