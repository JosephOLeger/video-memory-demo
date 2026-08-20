#!/usr/bin/env bash
# Cibles supplementaires : meme video, questions reellement differentes.
#
# Le sur-apprentissage vient du nombre d'exemples (184 fenetres vues 13.6 fois),
# pas de la quantite de video. Une seconde sonde sur la meme fenetre est un
# exemple neuf : la cible change, le contexte requis change, seule la video est
# partagee. C'est le seul levier disponible sans telecharger d'autres journees.
#
# Chaque sonde est generee dans SON fichier, et distill.py refuse de melanger un
# fichier de cibles avec une sonde differente -- c'est deja verifie a l'entree.
set -eu
cd "$(dirname "$0")/.."
PY=$HOME/venvs/titans/bin/python
CARD=${CARD:?}
NAME=${NAME:?}
PROBE=${PROBE:?}
CUDA_VISIBLE_DEVICES=$CARD timeout 14400 $PY -m titans.graft.distill --make-targets \
  --z cache/jake_day1_z.pt --span 12 --hop 4 --max-new 48 --guard-gb 8 \
  --probe "$PROBE" --out-targets cache/jake_day1_tgt_$NAME.pt
echo "=== $NAME fini $(date +%H:%M:%S) ==="
