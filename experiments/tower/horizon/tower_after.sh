# usage : tower_after.sh <carte> <nom_run_charge_a_attendre> <job>
cd ~/work/titans-mac
P=~/miniforge3/envs/titans/bin/python
while pgrep -u josephleger -f "iclr/$2\$" > /dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
COL="--dataset finevideo --data data/finevideo --dur-min 400 --dur-max 800 --basis cache/qwen_l24_pca128.pt --eval-videos 12 --max-train-videos 60 --write-stride 8 --accum 8 --lr 3e-3 --muon unit --steps 4000 --eval-every 800 --save-every 800 --seed 2 --guard-gb 11"
case $3 in
  recent) $P -m titans.graft.train_color $COL --depth-min 0.9 --depth-max 1.0 --out /home/josephleger/work/iclr/horizon/recent_s2 > /home/josephleger/work/iclr/horizon/recent_s2.log 2>&1 ;;
  far)    $P -m titans.graft.train_color $COL --depth-min 0.0 --depth-max 0.3 --out /home/josephleger/work/iclr/horizon/far_s2 > /home/josephleger/work/iclr/horizon/far_s2.log 2>&1 ;;
  pairs)  $P -m titans.graft.train_pairs --dataset finevideo --data data/finevideo --dur-min 400 --dur-max 800 --basis cache/qwen_l24_pca128.pt --lr 3e-3 --write-stride 8 --accum 8 --steps 8000 --eval-every 800 --train-k 1,2,4,8,16,32 --eval-k 1,4,16,32 --curriculum --muon unit_squash --seed 2 --guard-gb 9 --out /home/josephleger/work/iclr/cur_unit_squash_s2 > /home/josephleger/work/iclr/cur_unit_squash_s2.log 2>&1 ;;
esac
if [ "$3" = "far" ]; then
  while pgrep -u josephleger -f "horizon/recent_s2" > /dev/null; do sleep 120; done
  CK=$(ls ~/work/iclr/horizon/{recent,far}_s2_at*.pt | sort -V | paste -sd,)
  $P scripts/diag_graft_pr.py --data data/finevideo --basis cache/qwen_l24_pca128.pt --ckpts untrained,$CK --num-videos 6 --out /home/josephleger/work/iclr/horizon/gates_tower.json > /home/josephleger/work/iclr/horizon/gates_tower.log 2>&1
fi
