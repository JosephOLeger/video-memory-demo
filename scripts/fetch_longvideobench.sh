#!/usr/bin/env bash
# LongVideoBench (depot verrouille, ~150 Go) -> data/longvideobench/
#
# Le juge de la greffe. EgoLifeQA est devenu le signal d'entrainement, donc
# l'evaluation doit etre ailleurs; LongVideoBench stratifie ses questions par
# duree de video (15 / 60 / 600 / 3600 s), ce qui est exactement l'axe de portee
# de la memoire, et son classement publie la precision par tranche -- nos
# chiffres sont donc directement comparables a GPT-4o, Gemini-1.5-Pro et
# LLaVA-Video sans que nous construisions quoi que ce soit.
#
# L'archive video est monolithique : 31 parts de 4.88 Go a recoller. Pic disque
# ~300 Go (cache HF + extraction).
#
# Le depot est verrouille : il faut avoir accepte les conditions sur
# huggingface.co/datasets/longvideobench/LongVideoBench ET un jeton en place
# (`huggingface-cli login`, ou la variable HF_TOKEN). Les deux sont verifies
# AVANT le telechargement -- sinon l'echec arrive au bout d'une heure.
set -eu
cd "$(dirname "$0")/.."
DEST=${DEST:-data/longvideobench}

# L'interpreteur differe d'une machine a l'autre (miniforge sur la tour, venv
# ailleurs) : on le cherche au lieu de le coder en dur.
PY=${PY:-}
if [ -z "$PY" ]; then
  for c in "$HOME/miniforge3/envs/titans/bin/python" \
           "$HOME/miniforge3/envs/ai/bin/python" \
           "$HOME/venvs/titans/bin/python" "$(command -v python3 || true)"; do
    [ -x "$c" ] && { PY=$c; break; }
  done
fi
[ -n "$PY" ] || { echo "aucun interpreteur python trouve (poser PY=...)"; exit 1; }
echo "python: $PY"

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 320 ] || { echo "abandon : ${FREE_GB} Go libres, il en faut ~320"; exit 1; }
echo "disque: ${FREE_GB} Go libres"

echo "[1/5] acces au depot verrouille"
"$PY" - <<'EOF'
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("longvideobench/LongVideoBench", repo_type="dataset",
                          allow_patterns=["*.json"])
except Exception as e:
    sys.exit("acces refuse : %s: %s\n"
             "  1. accepter les conditions sur "
             "huggingface.co/datasets/longvideobench/LongVideoBench\n"
             "  2. poser un jeton : huggingface-cli login  (ou export HF_TOKEN=...)"
             % (type(e).__name__, e))
print("  ok, annotations dans", p)
EOF

echo "[2/5] telechargement des videos ($(date +%H:%M:%S))"
CACHE=$("$PY" - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("longvideobench/LongVideoBench", repo_type="dataset",
                        max_workers=4, resume_download=True))
EOF
)
echo "  cache: $CACHE"

echo "[3/5] annotations + sous-titres ($(date +%H:%M:%S))"
mkdir -p "$DEST"
cp -f "$CACHE"/lvb_*.json "$DEST"/
tar -xf "$CACHE/subtitles.tar" -C "$DEST"

echo "[4/5] reassemblage des videos ($(date +%H:%M:%S))"
# Les parts sont un seul tar coupe : on les recolle a la volee, sans jamais
# materialiser les 150 Go du tar complet en plus des parts.
cat "$CACHE"/videos.tar.part.* | tar -xf - -C "$DEST"
echo "  videos: $(find "$DEST" -name '*.mp4' | wc -l)"

echo "[5/5] verification ($(date +%H:%M:%S))"
"$PY" - "$DEST" <<'EOF'
import json, os, sys
dest = sys.argv[1]
data = json.load(open(os.path.join(dest, "lvb_val.json")))
missing = [d for d in data if not os.path.exists(os.path.join(dest, "videos",
                                                             d["video_path"]))]
print("  %d questions, %d sans video" % (len(data), len(missing)))
from collections import Counter
for g, n in sorted(Counter(d.get("duration_group") for d in data).items(),
                   key=lambda kv: (kv[0] is None, kv[0])):
    print("    tranche %s s : %d" % (g, n))
if missing:
    sys.exit("  %d videos manquantes -- ne pas evaluer sur ce jeu" % len(missing))
EOF
df -h . | tail -1
echo "=== fini $(date +%H:%M:%S) ==="
echo "les parts sont conservees dans le cache HF ; les supprimer apres verification :"
echo "  rm -rf $CACHE/videos.tar.part.*"
