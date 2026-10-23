cd ~/work/titans-mac
export HF_HUB_OFFLINE=1 PYTHONPATH=src
P=~/miniforge3/envs/titans/bin/python
C="--basis cache/qwen_l24_pca128.pt --bank $HOME/work/iclr/load/bank.pt --steps 1500 --eval-every 750 --eval-n 48 --eval-k 1,4,16,64"
for spec in "0 unit 1" "1 unit 10" "2 unit 100"; do set -- $spec
  CUDA_VISIBLE_DEVICES=$1 nohup $P -m titans.graft.train_load $C --muon $2 --dense-lambda $3 --out $HOME/work/iclr/load/scout_$2_l$3 > $HOME/work/iclr/load/scout_$2_l$3.log 2>&1 &
done
