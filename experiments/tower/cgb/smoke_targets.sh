#!/bin/bash
# 21/10 : smoke de cgb_targets.py sur une video FineVideo et un item factice (carte 0, apres le smoke des tailles)
O=$HOME/work/iclr/distill/orch.log; C=$HOME/work/iclr/cgb
until grep -q "bigmem : smoke tailles fini" $O; do sleep 60; done
cd $HOME/work/titans-mac
HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 $HOME/miniforge3/envs/titans/bin/python $C/cgb_targets.py --items $C/items_smoke.jsonl --videos data/finevideo/videos --out $C/smoke_tg > $C/smoke_targets.log 2>&1
$HOME/miniforge3/envs/titans/bin/python - <<PY >> $C/smoke_targets.log 2>&1
import torch; d=torch.load("$C/smoke_tg/bswbQtOPk6E.pt",map_location="cpu",weights_only=False)
print("clips",len(d["clips"]),tuple(d["clips"][0].shape),"q",len(d["qs"]))
for q in d["qs"]: print(q["c"],q["p"],q["meta"]["dist_s"],"aveugle",q["lt_blind"].argmax().item(),"fenetre",q["lt_window"].argmax().item(),"plafond",q["lt_teacher"].argmax().item(),"rep",q["answer"])
PY
echo "$(date +%F_%T) cgb : smoke cibles fini" >> $O
