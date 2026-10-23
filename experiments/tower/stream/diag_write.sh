#!/bin/bash
# 21/10 : norme des ecritures au fil du flux, aplat colore contre autres clips (CPU, sans Qwen)
S=$HOME/work/iclr/stream; cd $HOME/work/titans-mac
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/miniforge3/envs/titans/bin/python $S/diag_write.py --n 150 --targets "$S/tgc/*.pt" --basis cache/qwen_l24_pca128.pt --out $S/diag_write.jsonl --ckpts RANDOM:$S/C1_color_s0_at4000.pt $S/C1_color_s0_at500.pt $S/C1_color_s0_at4000.pt $S/C1_color_s1_at4000.pt $S/C1v_color_s1_at4000.pt $S/C1v_color_s0_at4000.pt $S/C1v_color_s3_at4000.pt > $S/diag_write.log 2>&1
