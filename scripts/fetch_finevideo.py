"""FineVideo -> data/finevideo/ : des videos variees, et purgees du juge.

Pourquoi ce jeu plutot qu'un autre
----------------------------------
EgoLife nous donne 52 h d'une seule personne, un seul appartement, une seule
semaine. FineVideo donne 3425 h sur 122 categories, et surtout les videos y sont
**hebergees directement** (dans des parquets) : pas de yt-dlp a l'echelle, pas de
videos mortes, licence CC-BY.

Mais la raison decisive est ailleurs : chaque video porte des annotations
**horodatees** -- scenes, accessoires, activites, personnages, transcription.
C'est exactement ce qui manque pour reparer la sonde unique : au lieu de poser
1550 fois la meme question, on peut demander ce qui se passait a un instant
precis, ce qui force une lecture specifique au contenu.

La contamination, qui n'est pas optionnelle
-------------------------------------------
LongVideoBench est notre juge et il est bati sur YouTube ; FineVideo aussi. Une
video presente des deux cotes rendrait toute mesure de decembre ininterpretable,
et le recouvrement serait invisible dans les scores. On croise donc les
identifiants YouTube et on **jette** tout ce qui apparait dans LVB, y compris son
split de test -- 3761 identifiants au total.

    python scripts/fetch_finevideo.py --hours 150 --lvb-ids ~/work/lvb_ids.txt \
        --out data/finevideo
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time


def youtube_id(row: dict) -> str:
    """L'identifiant YouTube, depuis ``json.original_json_filename``.

    Verifie sur le shard 0 : le parquet n'a que deux colonnes, ``mp4`` (octets)
    et ``json`` (struct), et l'identifiant est le nom du fichier source --
    ``d6b4OmUFt7I.json``. Aucun champ de premier niveau ne le porte, donc une
    heuristique qui cherche en surface renverrait "" pour toutes les lignes et
    **la purge du juge ne rejetterait rien, en silence**. D'ou l'exception plus
    bas plutot qu'un retour vide.
    """
    j = row.get("json") or {}
    for k in ("original_json_filename", "original_video_filename"):
        v = j.get(k)
        if isinstance(v, str) and v:
            return re.sub(r"\.(json|mp4)$", "", os.path.basename(v))
    raise KeyError("identifiant YouTube introuvable : le schema de FineVideo a "
                   "change, la purge LongVideoBench serait inoperante")


def video_bytes(row: dict) -> bytes:
    for k in ("mp4", "video", "bytes", "content"):
        v = row.get(k)
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, dict) and isinstance(v.get("bytes"), (bytes, bytearray)):
            return bytes(v["bytes"])
    for v in row.values():                       # dernier recours : le plus gros blob
        if isinstance(v, (bytes, bytearray)) and len(v) > 100_000:
            return bytes(v)
    return b""


def duration_s(row: dict) -> float:
    v = (row.get("json") or {}).get("duration_seconds")
    return float(v) if isinstance(v, (int, float)) and v > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=150.0,
                    help="heures de video a conserver (apres purge)")
    ap.add_argument("--min-seconds", type=float, default=240.0, dest="min_seconds",
                    help="ecarter les videos trop courtes pour porter une fenetre")
    ap.add_argument("--lvb-ids", required=True, dest="lvb_ids",
                    help="identifiants YouTube de LongVideoBench, un par ligne")
    ap.add_argument("--out", default="data/finevideo")
    ap.add_argument("--shards", type=int, default=1357)
    ap.add_argument("--cache", default=os.path.expanduser("~/hf_cache"))
    ap.add_argument("--keep-parquet", action="store_true", dest="keep_parquet",
                    help="ne pas supprimer chaque shard apres extraction")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    banned = {l.strip() for l in open(a.lvb_ids) if l.strip()}
    print(f"{len(banned)} identifiants LongVideoBench a exclure", flush=True)

    vid_dir = os.path.join(a.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.jsonl")
    seen = set()
    if os.path.exists(manifest_path):                       # reprise
        for line in open(manifest_path):
            seen.add(json.loads(line)["id"])
        print(f"reprise : {len(seen)} videos deja extraites", flush=True)

    kept_s = sum(json.loads(l)["duration"] for l in open(manifest_path)) \
        if os.path.exists(manifest_path) else 0.0
    dropped_lvb = short = 0
    t0 = time.time()
    man = open(manifest_path, "a")

    for i in range(a.shards):
        if kept_s >= a.hours * 3600:
            break
        name = f"data/train-{i:05d}-of-{a.shards:05d}.parquet"
        try:
            p = hf_hub_download("HuggingFaceFV/finevideo", name,
                                repo_type="dataset", cache_dir=a.cache)
        except GatedRepoError:
            sys.exit("depot verrouille : accepter les conditions sur "
                     "huggingface.co/datasets/HuggingFaceFV/finevideo puis "
                     "`huggingface-cli login` sur cette machine")
        except Exception as e:
            print(f"  shard {i} saute : {type(e).__name__}: {e}", flush=True)
            continue

        tab = pq.read_table(p)
        for r in tab.to_pylist():
            vid = youtube_id(r)
            if not vid or vid in seen:
                continue
            if vid in banned:                 # <- la purge du juge
                dropped_lvb += 1
                if dropped_lvb <= 10:
                    print(f"    exclu (present dans LongVideoBench) : {vid}",
                          flush=True)
                continue
            dur = duration_s(r)
            if dur and dur < a.min_seconds:
                short += 1
                continue
            blob = video_bytes(r)
            if not blob:
                continue
            with open(os.path.join(vid_dir, vid + ".mp4"), "wb") as f:
                f.write(blob)
            meta = r.get("json") or {}
            man.write(json.dumps({"id": vid, "duration": dur,
                                  "meta": meta}, default=str) + "\n")
            man.flush()
            seen.add(vid)
            kept_s += dur
            if kept_s >= a.hours * 3600:
                break
        if not a.keep_parquet:
            try:
                os.remove(p)
            except OSError:
                pass
        print(f"  shard {i + 1}/{a.shards} : {len(seen)} videos, "
              f"{kept_s / 3600:.1f} h gardees | {dropped_lvb} exclues (LVB), "
              f"{short} trop courtes | {(time.time() - t0) / 60:.0f} min",
              flush=True)

    man.close()
    print(f"\n{len(seen)} videos, {kept_s / 3600:.1f} h -> {vid_dir}")
    print(f"exclues pour recouvrement avec LongVideoBench : {dropped_lvb}")
    print(f"manifeste : {manifest_path}")


if __name__ == "__main__":
    main()
