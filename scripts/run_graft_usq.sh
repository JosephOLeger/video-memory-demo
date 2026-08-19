#!/usr/bin/env bash
# La meme course que run_graft_train.sh, mais sous unit_squash.
#
# Pourquoi elle vaut d'etre lancee cette nuit : sur le world model, unit_squash a
# rendu memory_gain 0.190 +/- 0.018 contre 0.086 +/- 0.024 pour unit (5 graines
# chacun, t = 7.8), soit indistinguable de rescale tout en restant borne. Si ca
# transfere a la greffe, le cout de la stabilite est nul et c'est la config du
# papier. Meme graine, meme stride, meme alpha : seule la magnitude change.
#
# c = 350 est choisi sur la trace de la greffe, pas repris du world model :
# ||S|| y vaut ~240, donc psi = 0.52 -- en pleine zone de transition. Sur les
# latents du world model ||S|| ~ 6000 mettrait psi a 0.98, quasi sature.
set -eu
cd "$(dirname "$0")/.."
PY=$HOME/venvs/titans/bin/python
CARD=${CARD:-1}
mkdir -p runs logs
CUDA_VISIBLE_DEVICES=$CARD timeout 43200 $PY -m titans.graft.distill \
  --z cache/jake_day1_z.pt --targets cache/jake_day1_tgt.pt \
  --basis cache/qwen_l24_pca128.pt \
  --muon unit_squash --muon-scale 350 --write-stride 8 --alpha-init 5.38e-4 \
  --steps 2500 --lr 1e-3 --warmup 100 \
  --eval-every 250 --eval-n 12 --guard-gb 9 \
  --out runs/graft_usq_s8
echo "=== fini $(date +%H:%M:%S) ==="
