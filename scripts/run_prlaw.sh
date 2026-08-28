#!/usr/bin/env bash
# Campagne collapse 28/08 — carte 1 : sonde de la loi pr sur artefacts geles.
# Volet L (loi, cycle) + T2 (echelle de checkpoints) + S (contraste de flux)
# + D (dose alpha-override). Predictions : predictions_collapse_2026-08-28.md
set -u
cd "$HOME/work/titans-mac" || exit 1
PY="$HOME/venvs/titans/bin/python"
LOG=logs/prlaw.log
mkdir -p results/prlaw
: > "$LOG"

run() {
  echo "--- $* $(date +%H:%M:%S) ---" >> "$LOG"
  CUDA_VISIBLE_DEVICES=1 nice -n 10 timeout 7200 "$PY" \
    scripts/diag_pr_collapse_probe.py --val cache/egolife_pca128_val.pt \
    "$@" >> "$LOG" 2>&1 || echo "ECHEC: $*" >> "$LOG"
}

# Volet L : la loi, mode cycle, 4 bras x 5 graines (tous les blocs)
for arm in mu_unit mu_unit10k mu_usq10k mu_scl; do
  run --frozen "frozen_muon/${arm}_s[0-9].pt" --mode cycle --clips 100 \
      --diag-out "results/prlaw/L_${arm}.jsonl"
done

# depot brut des ecritures pour l'analyse locale profonde (1 graine)
run --frozen "frozen_muon/mu_unit10k_s0.pt" --mode cycle --clips 100 \
    --dump-writes results/prlaw/wr --diag-out results/prlaw/L_dump_u10k_s0.jsonl

# Volet T2 : les statistiques d'ecriture le long de l'ENTRAINEMENT
# (checkpoints at2000..at10000 — g(t), Phi(t) contre pr(t))
for s in 0 1; do
  run --frozen "frozen_muon/mu_unit10k_s${s}_at*.pt" --mode cycle --clips 60 \
      --diag-out "results/prlaw/T2_unit10k_s${s}.jsonl"
  run --frozen "frozen_muon/mu_usq10k_s${s}_at*.pt" --mode cycle --clips 60 \
      --diag-out "results/prlaw/T2_usq10k_s${s}.jsonl"
done

# Volet S : contraste de flux (repetition maximale / aucune repetition)
for m in loop shuffle; do
  run --frozen "frozen_muon/mu_unit10k_s[01].pt" --mode "$m" --clips 100 \
      --diag-out "results/prlaw/S_${m}_unit10k.jsonl"
  run --frozen "frozen_muon/mu_usq10k_s0.pt" --mode "$m" --clips 100 \
      --diag-out "results/prlaw/S_${m}_usq10k.jsonl"
done

# Volet D : dose-reponse alpha-override (le terme de compte doit mordre)
for al in 0.003 0.01 0.03; do
  run --frozen "frozen_muon/mu_unit10k_s[01].pt" --mode cycle --clips 100 \
      --alpha-override "$al" --tokens-per-call 8 \
      --diag-out "results/prlaw/D_a${al}_unit10k.jsonl"
done

echo "PRLAW TOUT FINI $(date)" >> "$LOG"
