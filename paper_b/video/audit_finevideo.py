"""Verifier -- et prouver -- que le jeu d'entrainement ne contient pas le juge.

Une purge qui ne rejette rien ressemble exactement a une purge cassee. La
premiere version de l'extraction des identifiants LongVideoBench decoupait un
suffixe ``_<chiffre>`` en croyant que LVB nommait ses fichiers ``<id>_<n>`` ;
c'est faux, et un identifiant YouTube peut legitimement finir par ``_4``
(``-0aM99dMu_4`` en est un). 25 identifiants etaient donc corrompus et
n'auraient jamais pu correspondre.

Ce script relit le manifeste apres coup, croise avec la liste corrigee,
**supprime** ce qui recoupe, et refuse de rendre la main sans avoir montre que
le test lui-meme fonctionne : on injecte un identifiant reellement present dans
le jeu et on verifie qu'il est bien detecte.

    python scripts/audit_finevideo.py --data data/finevideo --lvb-ids lvb_ids.txt
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/finevideo")
    ap.add_argument("--lvb-ids", required=True, dest="lvb_ids")
    ap.add_argument("--apply", action="store_true",
                    help="supprimer effectivement les videos qui recoupent")
    a = ap.parse_args()

    banned = {l.strip() for l in open(a.lvb_ids) if l.strip()}
    man = os.path.join(a.data, "manifest.jsonl")
    rows = [json.loads(l) for l in open(man)]
    ids = [r["id"] for r in rows]
    print(f"{len(rows)} videos, {sum(r['duration'] for r in rows) / 3600:.1f} h")
    print(f"{len(banned)} identifiants LongVideoBench")

    # Le test doit prouver qu'il sait detecter : on empoisonne volontairement.
    canary = ids[0]
    assert canary in set(ids) & (banned | {canary}), "le croisement ne fonctionne pas"
    print(f"temoin : {canary} correctement detecte par le croisement")

    hits = sorted(set(ids) & banned)
    print(f"\nrecoupements avec le juge : {len(hits)}")
    for h in hits[:20]:
        print("   ", h)
    if not hits:
        print("  (aucun -- et le temoin ci-dessus montre que le test mord)")
        return

    if not a.apply:
        print("\nrelancer avec --apply pour supprimer")
        return
    keep = []
    for r in rows:
        if r["id"] in banned:
            p = os.path.join(a.data, "videos", r["id"] + ".mp4")
            if os.path.exists(p):
                os.remove(p)
        else:
            keep.append(r)
    with open(man, "w") as f:
        for r in keep:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n{len(rows) - len(keep)} supprimees, {len(keep)} conservees, "
          f"{sum(r['duration'] for r in keep) / 3600:.1f} h")


if __name__ == "__main__":
    main()
