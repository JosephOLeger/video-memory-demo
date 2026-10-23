cd ~/work/titans-mac
export HF_HUB_OFFLINE=1 PYTHONPATH=src
P=~/miniforge3/envs/titans/bin/python
CUDA_VISIBLE_DEVICES=0 nohup $P -m titans.graft.train_load --basis cache/qwen_l24_pca128.pt --bank ~/work/iclr/load/bank.pt --muon unit --dense-lambda 1.0 --steps 30 --eval-every 30 --eval-n 16 --eval-k 1,64 --train-k 1 --out ~/work/iclr/load/smoke > ~/work/iclr/load/smoke.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $P -m titans.graft.train_load --basis cache/qwen_l24_pca128.pt --bank ~/work/iclr/load/bank.pt --muon unit --dense-lambda 1.0 --steps 20 --eval-every 1000 --eval-n 8 --eval-k 1 --train-k 64 --out ~/work/iclr/load/smoke64 > ~/work/iclr/load/smoke64.log 2>&1 &
