"""Smoke checks: (1) injected path == processor path; (2) vision groups are independent."""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
h = C.Host()
fr = C.read_filler(C.fv_ids(1)[0], 8)            # 16 frames
e_all = h.encode(fr)
e_parts = torch.cat([h.encode(fr[:6]), h.encode(fr[6:])])
print("group independence: max|diff| =", (e_all.float() - e_parts.float()).abs().max().item(),
      " scale", e_all.float().abs().mean().item())
prompt = "What is happening in this video?\nA. cooking\nB. talking\nC. sport\nD. driving\n" + \
         "Answer with the option's letter from the given choices directly."
lg_inj, nt = h.score(None, e_all, prompt, 4)
msgs = [{"role": "user", "content": [{"type": "text", "text": "The most recent seconds of the video stream:"},
                                      {"type": "video"}, {"type": "text", "text": prompt}]}]
text = h.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = h.proc(text=[text], videos=[fr], video_metadata=[dict(fps=2.0, total_num_frames=16)],
             return_tensors="pt", do_sample_frames=False)
d = h.build(None, 0, 448, 8, prompt)
print("ids equal:", torch.equal(inp["input_ids"], d["input_ids"]),
      "mm equal:", torch.equal(inp["mm_token_type_ids"].int(), d["mm_token_type_ids"]),
      "spg:", inp["second_per_grid_ts"])
inp = {k: v.to("cuda") for k, v in inp.items()}
with torch.no_grad():
    lg = h.model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()[h.lid[:4]].cpu()
print("processor path", lg.tolist(), "\ninjected path ", lg_inj.tolist(), nt)
