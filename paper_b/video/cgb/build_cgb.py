"""21/10 : adapte CG-Bench (annotations MIT, 12 129 QCM, 1 219 videos de 9 a 105 min) au module
d'evaluation memoire. Annotations seulement : aucun besoin des videos ni du GPU.

Pour chaque question gardee, on produit :
  - 4 options : la bonne + 3 distracteurs tires parmi les choix du jeu (deterministe par qid),
    ordre melange de facon deterministe ;
  - des instants de question T = fin de la preuve + ecart, pour la courbe de retention. Un
    instant n'est garde que si toute la preuve est sortie de la fenetre : fin_preuve < T - fenetre ;
  - le plafond : l'intervalle de preuve (union des clue_intervals), a remettre dans la fenetre ;
  - la partition par VIDEO, 80 % entrainement / 20 % evaluation (sha1 de l'identifiant).

Filtres (configurables, defauts = ce qu'une memoire de latents peut raisonnablement porter) :
  - on retire Hallucination (la reponse porte sur une absence), Text Perception/Cognition (OCR :
    illisible apres la PCA des latents), Time Perception/Cognition (questions sur l'horodatage
    absolu ou la duree, que la troncature du flux rend ambigues) ;
  - on retire le domaine GUI (captures d'ecran, surtout du texte).

Sortie (persistante, jamais regeneree) : <out>/items.jsonl, <out>/stats.json, <out>/MANIFEST.json
(empreintes sha256 de la source et de la sortie, parametres).

    python build_cgb.py --src cgbench.json --out runs/cgb/v1
"""
import argparse, collections, hashlib, json, os, random

DROP_SUB = {"Hallucination", "Text Perception", "Text Cognition", "Time Perception", "Time Cognition"}
DROP_DOMAIN = {"GUI"}
LET = "ABCDEFGH"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_of(vid, eval_pct):
    return "eval" if int(hashlib.sha1(vid.encode()).hexdigest(), 16) % 100 < eval_pct else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=16.0, help="duree de la fenetre (16 images a 1 img/s)")
    ap.add_argument("--gaps", default="30,120,300,900", help="ecarts apres la fin de la preuve (s)")
    ap.add_argument("--with-end", action="store_true", default=True, help="ajoute T = fin de la video")
    ap.add_argument("--eval-pct", type=int, default=20)
    ap.add_argument("--n-options", type=int, default=4)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "items.jsonl")):
        raise SystemExit(f"{out}/items.jsonl existe deja : jeu fige, on ne le regenere pas")
    src = json.load(open(a.src))
    gaps = [float(g) for g in a.gaps.split(",")]
    st = collections.Counter()
    n_items = 0
    with open(os.path.join(out, "items.jsonl"), "w") as fo:
        for q in src:
            st["source"] += 1
            if q["sub_category"] in DROP_SUB or q["domain"] in DROP_DOMAIN:
                st["retire_categorie"] += 1
                continue
            ch, ans = q["choices"], q["answer"]
            if ans not in ch or len(set(ch)) < a.n_options:
                st["retire_choix"] += 1
                continue
            iv = [(float(s), float(e)) for s, e in q["clue_intervals"]]
            t0, t1, dur = min(s for s, _ in iv), max(e for _, e in iv), float(q["duration"])
            if not 0 <= t0 < t1 <= dur:
                st["retire_intervalle"] += 1
                continue
            rng = random.Random(f"cgb-{q['qid']}")
            opts = rng.sample([c for c in dict.fromkeys(ch) if c != ans], a.n_options - 1) + [ans]
            rng.shuffle(opts)
            Ts = [t1 + g for g in gaps if t1 + g <= dur]
            if a.with_end and dur not in Ts:
                Ts.append(dur)
            Ts = [T for T in Ts if t1 < T - a.window_s]          # preuve entierement hors fenetre
            if not Ts:
                st["retire_preuve_dans_fenetre"] += 1
                continue
            item = {"qid": q["qid"], "video_uid": q["video_uid"], "split": split_of(q["video_uid"], a.eval_pct),
                    "domain": q["domain"], "sub_category": q["sub_category"], "duration": dur,
                    "question": q["question"], "options": opts, "answer_idx": opts.index(ans),
                    "answer": ans, "evidence": iv, "evidence_span": [t0, t1], "n_intervals": len(iv),
                    "query_times": Ts, "distances": [round(T - t1, 1) for T in Ts]}
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_items += 1
            st["garde"] += 1
            st[f"split_{item['split']}"] += 1
            st["exemples_instants"] += len(Ts)
            st[f"exemples_{item['split']}"] += len(Ts)
    vids = collections.defaultdict(set)
    for l in open(os.path.join(out, "items.jsonl")):
        it = json.loads(l); vids[it["split"]].add(it["video_uid"])
    stats = dict(st, videos_train=len(vids["train"]), videos_eval=len(vids["eval"]))
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({"source": os.path.basename(a.src), "source_sha256": sha256(a.src),
               "items_sha256": sha256(os.path.join(out, "items.jsonl")), "params": vars(a),
               "drop_sub_category": sorted(DROP_SUB), "drop_domain": sorted(DROP_DOMAIN)},
              open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
