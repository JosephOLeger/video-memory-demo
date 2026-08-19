#!/usr/bin/env bash
# Where does the surprise signal actually live -- in ||grad l_t||, in ||S_t||, or
# in the learned gate theta_t?
#
# Josep's arctan squashes *a* magnitude; which magnitude it squashes decides
# whether it transmits anything at all. With eta = 0.995 the momentum averages
# ~200 tokens, so ||S_t|| may already be too smooth to carry a per-token surprise
# -- in which case the nonlinearity belongs on ||grad l_t|| instead. Same stream,
# same 3 clips, both magnitude conventions, and the probe now logs all three.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/venvs/titans/bin/python
VID='/home/josephleger/.cache/huggingface/hub/datasets--lmms-lab--EgoLife/snapshots/*/A2_ALICE/**/*.mp4'
CARD=${CARD:-2}
for M in unit rescale; do
  echo "=== $M ==="
  CUDA_VISIBLE_DEVICES=$CARD timeout 3600 $PY -m titans.graft.trace_stability \
    --videos "$VID" --num-videos 3 --basis cache/qwen_l24_pca128.pt \
    --frames 64 --chunk 8 --guard-gb 10 --print-every 100 \
    --override orthogonalize_momentum=$M \
    --out results/surprise_where_$M.json
done
echo "=== done ==="
