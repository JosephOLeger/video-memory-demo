#!/usr/bin/env bash
# LongVideoBench (gated, ~150 Go) -> data/longvideobench/
#
# Le juge de la greffe. EgoLifeQA devient l'entrainement, donc l'evaluation doit
# etre ailleurs; LongVideoBench stratifie ses questions par duree de video
# (15 / 60 / 600 / 3600 s), ce qui est exactement l'axe de portee memoire, et son
# leaderboard publie la precision par tranche -- donc nos chiffres sont
# directement comparables a GPT-4o, Gemini-1.5-Pro et LLaVA-Video.
#
# L'archive video est monolithique: 31 parts de 4.88 Go a concatener. Pic disque
# ~300 Go (cache HF + extraction), les parts sont supprimees a la fin.
set -eu
cd "$(dirname "$0")/.."
DEST=${DEST:-data/longvideobench}
PY=$HOME/venvs/titans/bin/python
mkdir -p "$DEST"

echo "[1/4] telechargement ($(date +%H:%M:%S))"
CACHE=$($PY - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("longvideobench/LongVideoBench", repo_type="dataset",
                        max_workers=8, resume_download=True))
EOF
)
echo "  cache: $CACHE"
df -h . | tail -1

echo "[2/4] annotations + sous-titres ($(date +%H:%M:%S))"
cp -f "$CACHE"/lvb_*.json "$DEST"/
tar -xf "$CACHE/subtitles.tar" -C "$DEST"

echo "[3/4] reassemblage des videos ($(date +%H:%M:%S))"
# Les parts sont un seul tar coupe: on les recolle a la volee, sans jamais
# materialiser les 150 Go du tar complet en plus des parts.
cat "$CACHE"/videos.tar.part.* | tar -xf - -C "$DEST"
echo "  videos: $(find "$DEST" -name '*.mp4' | wc -l)"

echo "[4/4] menage ($(date +%H:%M:%S))"
$PY - <<'EOF'
from huggingface_hub import scan_cache_dir
sc = scan_cache_dir()
for repo in sc.repos:
    if repo.repo_id == "longvideobench/LongVideoBench":
        revs = [r.commit_hash for r in repo.revisions]
        print("  cache a supprimer:", repo.size_on_disk_str)
EOF
echo "  (parts conservees pour l'instant -- supprimer a la main apres verification)"
df -h . | tail -1
echo "=== fini $(date +%H:%M:%S) ==="
