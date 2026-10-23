#!/bin/bash
# Omega (ATLAS, vraie fenetre glissante c=8 s=2) sur la distillation, go de Joseph 14/10.
# Tour, carte 1, apres le bras LVB "mismatched". Reference : dist_unit_s0 (memes cibles, seed 0).
O=$HOME/work/iclr/distill/orch.log
until grep -q "lvb : mis_s0_4000 fini" $O; do sleep 120; done
cd $HOME/work/titans-mac
echo "$(date +%F_%T) omega c8s2 s0 carte 1" >> $O
HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 $HOME/miniforge3/envs/titans/bin/python -m titans.graft.train_distill \
  --targets "$HOME/work/iclr/distill/targets_[0-3].pt" --basis cache/qwen_l24_pca128.pt --muon unit \
  --omega-window 8 --omega-stride 2 --steps 8000 --seed 0 --out $HOME/work/iclr/omega/om_c8s2_s0 \
  > $HOME/work/iclr/omega/om_c8s2_s0.log 2>&1
echo "$(date +%F_%T) omega c8s2 s0 fini (code $?)" >> $O
