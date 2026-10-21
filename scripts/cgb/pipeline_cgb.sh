#!/bin/bash
# 21/10 : module memoire, premiere version sur CG-Bench, de bout en bout (go de Josep : 64 Ko, 256 Ko, 1 Mo ; 4 Mo annule, trop de jours GPU).
#  1. attend la fin du telechargement + extraction ;
#  2. jeu v2 : 1 image / 4 s (clips de 64 s), fenetre 64 s, instants fin de preuve + 2 / 5 / 15 / 30 min + fin ;
#  3. precalcul des latents + 3 references de Qwen, 3 eclats sur les cartes libres (0 tout de suite, 1-2 apres la couleur) ;
#  4. bilan des filtres ; 5. smoke d'entrainement (20 pas) ;
#  6. runs : carte 0 = 256 Ko ; carte 1 = 1 Mo ; carte 2 = 64 Ko. 32 clips de passe, recalcul par segments.
C=$HOME/work/iclr/cgb; O=$HOME/work/iclr/distill/orch.log; V=$C/v2
PY=$HOME/miniforge3/envs/titans/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date +%F_%T) cgb : $*" >> $O; }
card_busy() { local p; for p in $(pgrep -u "$USER" -f python); do
  tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
wait_card() { while card_busy $1 || { sleep 60; card_busy $1; }; do sleep 120; done; }

# la premiere ligne « extraction finie » vient de l'essai rate de 14:57 : on attend la deuxieme
until [ "$(grep -c 'cgb : extraction finie' $O)" -ge 2 ]; do sleep 300; done
[ -f $V/items.jsonl ] || python3 -I $C/build_cgb.py --src $C/cgbench.json --out $V --window-s 64 --gaps 120,300,900,1800 > $C/build_v2.log 2>&1
log "jeu v2 pret : $(grep -o '"garde": [0-9]*' $V/stats.json)"

cd $HOME/work/titans-mac
for s in 0 1 2; do
  ( wait_card $s
    log "precalcul eclat $s carte $s"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=$s $PY $C/cgb_targets.py --items $V/items.jsonl --videos $C/videos --out $V/tg \
       --fps 0.25 --shard $s --nshards 3 > $C/targets_s$s.log 2>&1
    log "precalcul eclat $s fini (code $?)" ) &
done
wait
$PY $C/cgb_filter.py --tg $V/tg --out $V/filtre > $C/filtre.log 2>&1
log "bilan des filtres ecrit ($V/filtre/bilan.json)"

CO=(--targets "$V/tg/*.pt" --kinds cgb --max-past 32 --lam-lat 0 --lam-con 0 --lr 3e-3 --batch 4
    --eval-every 500 --max-eval 300 --drop-blind --teacher-ok-only 1 --seed 0)
run() { # carte d h segments nom pas
  local c=$1 d=$2 h=$3 sg=$4 n=$5 st=$6
  wait_card $c
  log "run $n carte $c"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$c $PY -m titans.graft.train_stream "${CO[@]}" --basis cache/qwen_l24_stream_pca$d.pt \
     --d-mem $d --memory-hidden $h --ckpt-seg $sg --steps $st --out $C/runs/$n > $C/runs/$n.log 2>&1
  log "run $n fini (code $?)"
}
mkdir -p $C/runs
run 0 128 64 90 smoke_cgb 20
grep -q "TRAIN-STREAM-FINIE" $C/runs/smoke_cgb.log && ! grep -q "Traceback" $C/runs/smoke_cgb.log || { log "smoke ECHEC, runs non lances"; exit 1; }
run 0 256 128 90 cgb_256ko 4000 &
run 1 512 256 90 cgb_1mo 4000 &
run 2 128 64 90 cgb_64ko 4000 &
wait
log "module memoire v1 : 3 runs finis"
