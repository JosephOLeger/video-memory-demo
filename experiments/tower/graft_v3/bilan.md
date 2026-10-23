# Bilan : greffon mémoire sur le banc de scènes v3 (22/10/2026)

Dossier : `~/work/iclr/graft_v3/` sur la tour. Aucun fichier de `scenes/`, `ladder/`, `data/`, `cache/`, `runs/` n'a été modifié.

## 1. Bug fps (M-RoPE)

### Portée
* transformers 5.14 ignore `fps=` dans `processor(...)`. Il retombe sur 24 fps par défaut, donc `second_per_grid_ts = 2/24 = 1/12 s`.
* Valeur juste : `2 / fps`. Le flux du greffon est à 1 image/s, donc 2,0 s par groupe : le temps était compressé **24 fois** (12 fois à 2 fps). Pire : `get_rope_index` prend la partie entière de `t × spg × 2`, donc les 6 premiers groupes d'un clip avaient tous la position temporelle 0.
* Un seul point d'entrée dans le code du greffon : `fit_basis.video_inputs`. Tout passe par lui : `distill.forced_inputs`, `teacher_continuation`, `train_stream`, `train_load.video_z`, `stream_targets`, `stream_color_targets`, `stream_color_variants`, `distill_targets`, `eval_lvb`, `eval_stream`, `needle`, `reach`, `train_color`, `train_pairs`, `train_mcq`, `cache_z`, `check_qwen`, `trace_stability`, et `~/work/iclr/cgb/cgb_targets.py` (qui importe `titans.graft.fit_basis`). Aucun autre appel direct au processeur avec des vidéos dans `titans-mac/src`.
* Versions (vérifié en appelant le processeur, CPU seulement) :
  | machine | transformers | bug |
  |---|---|---|
  | tour | 5.14.1 | oui (spg 0,0833 au lieu de 2,0) |
  | local (`ai`) | 5.13.0 | oui |
  | .135 (`~/venvs/titans`) | 5.14.1 | oui ; son `fit_basis.py` a toujours l'appel `fps=` (lecture seule, non corrigé) |
* Copies non corrigées encore présentes sur la tour (non touchées, à savoir) : `~/work/iclr/titans-mac/src/titans/graft/fit_basis.py` (copie du 17/09), `iclr/bench_dense/tm/...`, `iclr/cascade/tm/...`, `iclr/audit/ceiling_probe.py`, `iclr/audit/snap1/ceiling.py`.

### Correctif
`fit_basis.video_inputs` passe `video_metadata=[{"fps": fps, "total_num_frames": F}]` et `do_sample_frames=False` (comme `scenes/ceiling_v3.py`). Nouvelle fonction `check_temporal(inputs, fps, processor, n_videos)` : assertion `second_per_grid_ts == temporal_patch_size / fps`, appelée à chaque `video_inputs`. Vérifié : spg = 2,0 à 1 fps, 1,0 à 2 fps, sur la tour (5.14.1) et en local (5.13.0).

Attention : `eval_lvb` (échantillonnage uniforme, fps = images/durée) aura maintenant des positions temporelles réelles, donc grandes (par exemple 112 s par groupe pour 64 images sur 1 h). C'est le comportement natif de Qwen, mais il diffère des mesures LVB passées.

**Effet immédiat** : depuis 15 h 20 environ, tout NOUVEAU processus de la tour qui importe `titans.graft` utilise le bon temps. Les cibles déjà en cache (`stream/tg`, `tgc`, `tgv`, `cgb/*`, la base PCA `qwen_l24_pca128.pt`) ont été calculées avec le bug. Un run lancé maintenant sur ces cibles mélange donc fenêtre corrigée et mémoire / enseignant bugués.

### Impact mesuré (`fps_impact.py`, 25 vidéos d'éval de `stream/tgc`, 100 questions couleur)
| | ancien (bug) | corrigé |
|---|---|---|
| Qwen fenêtre seule | 30 % | 29 % |
| enseignant [preuve, fenêtre] | 97 % | 95 % |
| accord ancien / corrigé, fenêtre | 86 % des questions | |
| accord ancien / corrigé, enseignant | 98 % | |

* Le chemin ancien recalculé reproduit le cache **exactement** (lettres 100 %, écart de logits 0, états enseignant 0) : les cibles en cache sont bien bugguées.
* Les cibles en cache changeraient : états de mémoire (couche 24, un jeton sur 8) écart relatif 27 % en norme (cosinus moyen 0,965, 639 clips) ; aiguille (patch) 25 % ; états de réponse de l'enseignant 7,5 % (médiane 7,3 %, max 14 %).
* Greffon entraîné C1 (`stream/C1_color_s0_at4000.pt`), mêmes 100 questions, `train_stream --steps 0` :
  | | mémoire | mémoire d'une autre vidéo |
  |---|---|---|
  | fenêtre corrigée, mémoire en cache (bug) | 100 % | 17 % |
  | tout corrigé (cibles recalculées) | 99 % | 19 % |
  Le résultat couleur survit au correctif. La tâche est saturée, donc cela ne dit rien des tâches plus fines.
* Lecture : sur la couleur, l'effet sur la précision est faible (1 à 2 points) mais les représentations bougent nettement. Il faut recalculer les cibles avant toute nouvelle comparaison, et refaire la base PCA si on veut une base cohérente.

## 2. Fichiers modifiés
| fichier | sauvegarde | changement |
|---|---|---|
| tour `~/work/titans-mac/src/titans/graft/fit_basis.py` | `fit_basis.py.bak_2210` | correctif + `check_temporal` |
| local `/home/josep/projects/titans-mac/src/titans/graft/fit_basis.py` | git (non commité) | identique à la tour |

Tests locaux : `pytest -q` : 275 passés, 0 échec.

Nouveaux fichiers (tous dans `~/work/iclr/graft_v3/`) : `fps_impact.py`, `prep_v3.py`, `train_v3.py`, `wait_then.py`, `queue.sh`, `fps_impact/` (cibles corrigées de 25 vidéos + liens vers les anciennes), `data/`, `runs/`, `logs/`.

## 3. Pipeline de données (`prep_v3.py`)

### Cohérence avec l'échelle
Le harnais v3 de l'échelle est `ladder/ladder3.py` (écrit aujourd'hui à 15 h 21, file `jobs3.txt`). Je l'ai suivi :
* 2 fps, 448 px, 1 groupe = 2 images = 1 s ; passé P = 160 s ; fenêtre W = 4 s ; tableau de 6 s qui finit D s avant la fenêtre.
* style v3b pour toutes les familles (`TStyle(count_gap=True, door_bright=True)`), générateur = copie gelée `ladder/v3/gen_tableau.py` (md5 identique à `scenes/`).
* familles : count, colour, big_object (gist), sign_number, hat_color (détail).
* prompt = celui du générateur, inchangé (comme ladder3).
* format de l'élève = condition `empty` de l'échelle : « The most recent seconds of the video stream: » + fenêtre + question, logits des lettres à la dernière position. L'oracle = format `oracle` de l'échelle (tableau en vidéo mémoire + fenêtre).

**Écart à trancher.** ladder3 n'utilise PAS FineVideo : son remplissage est le décor extérieur procédural de v3, tiré par groupe. La consigne demandait du FineVideo (split TRAIN). J'ai donc fait :
* **éval** = exactement les flux de ladder3 (remplissage procédural, groupes 10000+, D ∈ {0, 120}, les D de ladder3) ;
* **entraînement** = FineVideo split TRAIN (`stream_targets.is_eval` faux), sans les 120 vidéos de remplissage de l'échelle v1. L'éval mesure donc aussi un transfert de décor FineVideo → procédural. L'option `--filler proc` produit un entraînement à décor procédural (seeds du split train) si le lead préfère ; coût plus élevé (un décor neuf par groupe).

### Ce que la mémoire reçoit
* Le passé est coupé en clips de 8 s (16 images, 8 groupes, 2 048 jetons visuels). Chaque clip est encodé SEUL par Qwen gelé, jusqu'à la couche 24, un jeton sur 8 (256 jetons par clip), projeté sur la base `cache/qwen_l24_pca128.pt` (exactement `branch.down(h)`). 20 clips = 5 120 écritures.
* Temps M-RoPE vérifié (spg = 1,0 s). Accélération : la tour visuelle n'attend qu'au sein d'un groupe (comme `core.Host`), donc les jetons visuels de chaque groupe sont calculés une fois et injectés ; vérifié identique au bit près au chemin processeur (écart relatif 0,0).
* Groupes contrefactuels préservés : les 4 variantes partagent remplissage, fenêtre, question et options ; seuls le ou les clips qui contiennent le tableau diffèrent (`patches[D][variante] = {indice de clip: z}`). « Mémoire d'une autre variante » = même passé à ces clips près.

### Fichiers produits (`data/`)
* `fill_<split>/<clé>.pt` : `z` [clips, 256, 128] fp16 du remplissage.
* `<split>/<famille>_<shard>.pt` : liste de groupes (famille, groupe, split, clé de remplissage, j0, D, prompt, options, 4 réponses, faits, patches).
* fenêtre : relue à la volée (décord ou rendu procédural), pas stockée.
* cibles enseignant : non nécessaires (perte CE sur les lettres + contrastif), l'oracle est calculé à l'éval.

### Volumes
* éval : 30 groupes × 5 familles = 150 groupes (10000 à 10029), 2 D chacun ; 264 Mo ; 65 min de GPU partagé. Extensible à 150 groupes / famille (`--groups 150`) comme `jobs3.txt`.
* entraînement : 400 groupes × 5 familles = 2 000 groupes, D tiré uniformément dans [0, 154], 60 vidéos FineVideo × 30 clips de remplissage, 2 éclats. Fait : 2 000 groupes (873 dans l'éclat 0, 1 127 dans l'éclat 1 ; 349 à 457 par famille), 835 Mo, 48 + 78 min de GPU partagé.

## 4. Smoke

### Entraînement (`train_v3.py`, init C1 couleur, 20 pas, lot 4, `--lam-con 1 --margin 1`)
Perte : `CE(lettres | mémoire propre) + λ·relu(marge + CE(propre) − CE(bonne réponse | mémoire de l'autre variante du même groupe))`.

Écritures batchées, `--mem-stride 1` (5 120 écritures par flux), carte 1 partagée :

| pas | CE | contrastif | CE(autre) − CE(propre) | |g| (avant clip) |
|---|---|---|---|---|
| 0 | 3,40 | 0,74 | +0,69 | 279 |
| 10 | 3,52 | 0,98 | +0,35 | 2 204 |
| 19 | 3,07 | 1,06 | +0,23 | 936 |

* **50,0 s/pas** (lot 4). Par pas : écriture propre 7,6 s, écriture autre variante 7,6 s, élève avant 5,2 s, élève arrière 0,7 s, rétropropagation de l'écriture 28,4 s. Pic 9,9 Gio.
* Contrôle batché = seul : écart relatif max 6,9e-4.
* `--mem-stride 4` (1 280 écritures, 64 par clip de 8 s), 5 pas : **14,4 s/pas** (écriture 1,9 + 1,9 s, rétro 7,1 s), pic 9,6 Gio.
* Le pipeline tourne de bout en bout, terme contrastif actif. Les normes de gradient sont très grandes (jusqu'à 2 800, coupées à 1) : la tête C1 n'est pas faite pour ce format.
* Le premier jet, sans batch, prenait 183 s/pas.

Premier jet, écritures une par une : 183 s/pas (lot 4), dont 61 s d'écriture avant et 116 s de rétropropagation. 640 pas de récurrence séquentiels (5 120 écritures, chunk 8) par flux : la boucle mémoire domine, comme noté au 12/10. Correctif : les mémoires du lot (propres, puis autres variantes) sont écrites en UNE passe batchée, et la rétropropagation traverse l'écriture une seule fois (contrôle au démarrage : écriture batchée = écriture seule).

### Éval smoke (20 groupes, 4 par famille, D ∈ {0, 120})
Point de contrôle `runs/smoke_at20.pt`, 160 questions, 762 s (stride 1). Hasard = 25 %.

| | propre | autre variante (donneur) | fenêtre | oracle |
|---|---|---|---|---|
| tout, item | 25,0 | 26,2 (donneur 25,0) | 25,0 | 97,5 |
| tout, groupe | 0 | 0 | 0 | 90 |
| gist (count, colour, big_object), item | 24,0 | 27,1 | 25,0 | 95,8 (grp 83) |
| détail (sign, hat), item | 26,6 | 25,0 | 25,0 | 100 (grp 100) |
| D = 0 / D = 120 | 25,0 / 25,0 | 26,2 / 26,2 | 25,0 | 97,5 |

* Lecture : après 20 pas, le greffon (init C1 couleur) ne lit pas la mémoire sur ce banc. Propre = fenêtre = hasard, groupe 0. C'est attendu à 20 pas ; le smoke valide la mécanique, pas l'apprentissage.
* Les contrôles sont corrects : fenêtre exactement 25 % (entrée identique sur les 4 variantes), oracle 97,5 % (90 % par groupe) au format de l'échelle, avec la fenêtre procédurale de ladder3 et l'éval split. C'est le plafond à battre.

## 5. Entraînement complet proposé
Commande (une carte avec ≥ 12 Go libres ; à lancer par le lead) :

```
cd ~/work/iclr/graft_v3
HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/work/titans-mac/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
~/miniforge3/envs/titans/bin/python wait_then.py / train_v3.py \
  --init $HOME/work/iclr/stream/C1_color_s0_at4000.pt \
  --mem-stride 4 --lam-con 1 --margin 1 --lr 1e-3 --batch 4 --steps 3000 --warmup 100 \
  --eval-every 500 --eval-groups 6 --save-every 500 --seed 0 --out runs/v3_s0 \
  > logs/v3_s0.log 2>&1 &
```

* Temps estimé : 3 000 pas × 14,4 s ≈ **12 h**, plus 6 évals de 30 groupes (≈ 5 min chacune à stride 4) ≈ 12,5 h sur une carte partagée. À `--mem-stride 1` (toute la résolution) : 50 s/pas, soit ≈ 42 h.
* Éval finale conseillée : les 150 groupes (`--eval-only --init runs/v3_s0_at3000.pt`) ≈ 25 min à stride 4 ; 95 min à stride 1.
* Choix à valider : `--mem-stride 4` (coût / densité de la mémoire), départ de C1 ou d'une tête neuve (sans `--init`), fps fixé à 1 s/groupe. Les cibles d'entraînement sont déjà corrigées (spg 1,0) ; C1 a été entraîné avec le bug.
* Pour une comparaison entre bras, compter 3 graines (l'écart de graine du 7 jours valait 7,6 points).

## Points ouverts
* Remplissage FineVideo (consigne) contre procédural (ladder3) : à trancher.
* Base PCA calculée avec le bug fps : à refaire si l'on veut une base cohérente (même chose pour les cibles `stream/*`, `cgb/*`).
* La GPU est très disputée (3 cartes pleines la plupart de l'après-midi) ; deux lancements ont pris un OOM au chargement parce qu'une autre file a pris la même mémoire à la même seconde. `wait_then.py` relance dans ce cas.
