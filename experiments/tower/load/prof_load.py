import time, random, torch, torch.nn.functional as F
from transformers import AutoProcessor
from titans.graft.check_qwen import load_host
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args
from titans.graft.qwen import graft
from titans.graft.train_load import assemble, write_stream, win_prompt_logits
from titans.graft.train_pairs import pairs_prompt
from titans.graft.needle import LETTERS
import types
dev="cuda"
model=load_host("Qwen/Qwen2.5-VL-3B-Instruct", torch.bfloat16, dev)
proc=AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
pca=torch.load("cache/qwen_l24_pca128.pt",weights_only=False)["pca"]
a=types.SimpleNamespace(memory_hidden=64,alpha_init=1.27e-4,muon="unit",muon_scale=None,cascade=False)
br=MemoryBranch(2048,128,chunk_size=8,write_stride=8,down_pca=pca,**memory_kwargs_from_args(a)).to(dev)
wr=graft(model,br,layer=24); wr.writes_enabled=False
head=torch.nn.Linear(128,128).to(dev)
bank=torch.load("/home/josephleger/work/iclr/load/bank.pt",weights_only=False)["train"]
bank["pairs"]={k:v.to(dev) for k,v in bank["pairs"].items()}; bank["fillers"]=[f.to(dev) for f in bank["fillers"]]
lid=torch.tensor([proc.tokenizer.encode(LETTERS[i],add_special_tokens=False)[0] for i in range(8)],device=dev)
rng=random.Random(0)
def T(): torch.cuda.synchronize(); return time.time()
for K in (1,64):
  for dense in (False,True):
    ts=[]
    for _ in range(4):
      z,_,key,col=assemble(rng,bank,bank["pairs"],K,8,[0,1,2]); z=z.float().unsqueeze(0)
      t0=T(); st,S,ld,_=write_stream(br,z,8,head if dense else None); t1=T()
      wr.carry,wr.token_offset=(st,S),0
      win=bank["windows"][0]
      cls=win_prompt_logits(model,proc,win,pairs_prompt(key),lid,dev); t2=T()
      loss=F.cross_entropy(cls[None],torch.tensor([col],device=dev))+(ld if ld is not None else 0)
      loss.backward(); t3=T()
      ts.append((t1-t0,t2-t1,t3-t2))
    m=[sum(x[i] for x in ts[1:])/3 for i in range(3)]
    print(f"K={K} writes={z.shape[1]} dense={dense}: write {m[0]:.2f}s  answer-fwd {m[1]:.2f}s  backward {m[2]:.2f}s",flush=True)
