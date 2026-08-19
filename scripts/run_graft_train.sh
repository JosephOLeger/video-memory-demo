#!/usr/bin/env bash
# Premiere vraie course de la greffe. unit, stride 8.
#
# Le stride vient du profil, pas du confort : le cout par pas est LINEAIRE en
# nombre d'ecritures, pas de chunks -- Newton-Schulz tourne par token dans la
# boucle _step, pas une fois par chunk. Mesure : 1.5 s + 0.0058 s/ecriture, sur
# trois points (930 / 1859 / 3718 ecritures -> 8.8 / 15.9 / 28.5 s par pas).
# 930 ecritures couvrent les 11 clips de contexte pour une portee mesuree a 4300,
# donc on est trois fois dans le budget memoire et quatre fois moins cher.
# alpha = 1/(2*930) = 5.38e-4, la loi de portee appliquee a ce nombre d'ecritures.
set -eu
cd "$(dirname "$0")/.."
PY=$HOME/venvs/titans/bin/python
CARD=${CARD:-3}
mkdir -p runs logs
CUDA_VISIBLE_DEVICES=$CARD timeout 43200 $PY -m titans.graft.distill \
  --z cache/jake_day1_z.pt --targets cache/jake_day1_tgt.pt \
  --basis cache/qwen_l24_pca128.pt \
  --muon unit --write-stride 8 --alpha-init 5.38e-4 \
  --steps 2500 --lr 1e-3 --warmup 100 \
  --eval-every 250 --eval-n 12 --guard-gb 9 \
  --out runs/graft_unit_s8
echo "=== fini $(date +%H:%M:%S) ==="
