# Bilan : validation du HARNAIS de l'échelle de mémoires (22/10/2026)

Pré-enregistrement : `predictions.md` (version initiale + addendum A, écrits avant les mesures).
Code : `core.py`, `ladder.py`, `analyse.py`. Données : `runs/val.jsonl`. Figures : `figs/`.

## Portée (après l'audit et la consigne du lead)

* Générateur v1 figé (`gen_scenes_snapshot.py`, md5 `fbd5a18b`). Il a des défauts connus : étiquettes `door_state` fausses et fuite par l'état final. Seules deux familles lues pour la bonne raison sont utilisées : `init_count` (gist, lu sur les premières secondes) et `sign_number` (détail).
* Le run principal (5 familles × 150 groupes) a été **arrêté** sur consigne. Ce bilan valide **l'instrument** (le harnais), pas le banc v1.
* 30 groupes frais par famille (5000 à 5029, jamais vus par le générateur, le sweep ni l'audit), soit 120 questions par cellule. `fifo448`, `solo` et la charge L = 3 n'ont que 5 groupes (20 questions) : ce sont des indications.
* Hasard = 25 %. « groupe » = les 4 variantes contrefactuelles justes. SE par bootstrap sur les groupes.

## Verdict

**Le harnais classe l'échelle comme prédit, sur les 4 points demandés.** Les contrôles négatifs tombent exactement où la construction l'impose. Le mécanisme des images clés est vérifié item par item. Une seule déviation : `mean` garde plus de gist que prévu. Elle va dans le sens prédit (détail avant gist) mais son origine n'est pas établie.

### Tableau principal (exactitude par question % ; groupe % entre parenthèses)

| mémoire | budget | init_count (gist) | sign_number (détail) |
|---|---|---|---|
| oracle | 12 à 24 Mio | 89 (57) | 100 (100) |
| fifo 224 px, D = 0 | 20 Mio | 73 (37) | 100 (100) |
| fifo 224 px, D = 30 | 20 Mio | 81 (50) | 100 (100) |
| fifo 224 px, D = 90 | 20 Mio | **25 (0)** | **25 (0)** |
| key, D = 0 / 30 / 90 | 22 Mio | 51 / 55 / 63 (3 / 13 / 7) | 45 / 42 / 44 (3 / 3 / 7) |
| mean, D = 0 / 30 / 90 | 22 Mio | 44 / 56 / 55 (0 / 7 / 7) | **26 / 26 / 26 (0)** |
| tail (5 s de fin) | 5 Mio | 24 (0) | 22 (0) |
| empty | 0 | 25 (0) | 25 (0) |
| blind | 0 | 25 (0) | 25 (0) |
| other (autre variante, même groupe) | 12 à 24 Mio | **6 (0)**, donneur choisi 89 % | **0 (0)**, donneur choisi 100 % |

SE par question : 2 à 5 points (voir `analyse.py`). Indications à n = 20 : `fifo448@0` 90 / 100, `fifo448@30` 25 / 26.

![exactitude contre D](figs/acc_vs_D_family.png)

### Les 4 questions du lead

1. **FIFO en marche d'escalier contre D : OUI.** Dans le tampon (D + E ≤ 80 s), on est au plafond pour le détail (100 %, groupe 100 %) et près de l'oracle pour le gist (73 à 81 % contre 89 %). Hors du tampon (D = 90), on tombe à 25,0 % exactement et le groupe vaut 0. `fifo448` (20 s) fait la même marche, plus tôt : 90 / 100 à D = 0, hasard dès D = 30. Le léger déficit de `fifo` 224 px à D = 0 sur init_count (73 contre 90 pour `fifo448`, n = 20) est compatible avec un coût de résolution sur le comptage.
2. **Images clés, phase aléatoire : OUI, et le mécanisme est vérifié.** Les courbes sont plates en D pour le détail (45 / 42 / 44). Le niveau correspond à la prédiction P5 : couverture q ≈ 0,25 pour un panneau de 2 s, donc 25 + 0,25 × 75 ≈ 44 %. Pour init_count, q ≈ 0,5 donne ≈ 57 % prédit, et on mesure 51 à 63. En conditionnant sur la question « une seconde clé est-elle tombée sur la preuve ? » :

   | famille | clé sur la preuve | n | exactitude |
   |---|---|---|---|
   | sign_number | oui | 88 | **100 %** |
   | sign_number | non | 272 | **25 %** |
   | init_count | oui (état initial vu) | 175 | 69 % |
   | init_count | non | 185 | 44 % |

   Le détail est tout ou rien selon la couverture. Pour le gist sans l'instant initial, il reste un signal partiel (44 %) : des images clés plus tardives laissent déduire une partie du compte.
3. **La moyenne temporelle dégrade le détail avant le gist : OUI.** Détail au hasard exact (25,8 %, groupe 0) à tout D. Gist à 44 à 56 % (groupe 0 à 7 %).
4. **Le contrôle « other » contrefactuel est bien anti-corrélé : OUI.** Avec la mémoire d'une autre variante du même groupe, l'exactitude sur la cible tombe à 6 % (gist) et 0 % (détail), sous le hasard. La réponse du donneur est choisie à 89 % et 100 %, exactement le niveau de l'oracle. La mémoire est donc lue pour son contenu, et le harnais le détecte. C'est l'inverse de l'échec CG-Bench.

### Contrôles nuls : exacts par construction
* `empty` et `blind` donnent 25,0 % et un groupe de 0 exactement (entrée identique pour les 4 variantes). Une autre valeur aurait signalé un bug.
* `tail` (état final seul) : 24 % et 22 %, groupe 0. Qwen non entraîné ne tire pas init_count de l'état final. Ce contrôle reste **obligatoire** pour toute mémoire entraînée : l'audit montre qu'une règle apprise y atteindrait 63 à 87 %.

## Déviations par rapport aux prédictions

| écart | prédit | mesuré | lecture |
|---|---|---|---|
| mean, gist | ≤ 44 % (≤ 0,3 × (c − h)) | 44 à 56 % | Plus haut que prévu. Hypothèse de l'audit n°18 : la moyenne garde la « masse de couleur » des chemises, donc un comptage statistique et non épisodique. **Non tranché.** À tester avec une borne Bayes sur l'histogramme moyen de couleurs, ou avec des distracteurs de même palette. Tant que ce n'est pas fait, un gain gist d'une mémoire compressive ne prouve pas une mémoire de l'instant. |
| tail, init_count | sous le hasard (5 à 25 %) | 24 % | Au hasard, pas en dessous. Sans conséquence. |
| oracle > solo sur init_count | oracle ≥ solo − 10 | solo 70, oracle 89 (n = 20 pour solo) | Le format flux (« clip », mémoire puis fenêtre) aide Qwen. Effectif solo trop petit pour conclure. |
| key init_count, pente en D | plat | 51 → 63 (≈ 2 SE) | Faible et à la limite. Le détail est plat. À revérifier avec plus de groupes. |
| charge L = 3, init_count | baisse de 0 à 15 points | fifo 81 → 20, key 55 → 25 (n = 20) | **Forte baisse sur le gist, aucune sur le détail** (fifo sign 100). Les distracteurs sont des épisodes du même type : l'hôte compte les gens du mauvais clip, alors que le panneau est unique. C'est une propriété réelle de l'hôte (pas de sélection du « plus récent »), pas un défaut du banc. Mais elle rend la condition de charge inutilisable pour le gist tant que la question ne désigne pas le clip sans ambiguïté. n petit. |

Aucune déviation n'indique de raccourci dans le harnais : tous les nuls sont exacts, et le contrôle contrefactuel suit le contenu.

## Ce qui est certifié
* **Le harnais** : assemblage du flux, injection des jetons (identique bit à bit au chemin processeur, vérifié par `check.py`), budgets, contrôles nuls, contrôle contrefactuel, échelle FIFO / key / mean / tail. Il est prêt pour v3.
* **Sur v1** : `sign_number` passe tous les critères révisés (oracle 100 % en groupe, marche FIFO 100 → 0 en groupe, tail / empty / other à 0). `init_count` passe les critères 1 à 4 (oracle 57 % en groupe, marche FIFO 37 à 50 → 0, tail 0) avec la réserve « mean ». Aucune famille v1 n'est à utiliser pour une mémoire ENTRAÎNÉE, à cause des fuites de l'audit (n°1, durée T et état final).

## Bug trouvé en route (concerne d'autres dossiers)
transformers 5.14 ignore `fps=` dans `processor(...)` et met `second_per_grid_ts = 1/12 s`, ce qui compresse le temps M-RoPE 12 fois. Correctif : `video_metadata=[{"fps": 2, "total_num_frames": T}]` et `do_sample_frames=False`, ou construire les entrées comme `core.Host.build`. `ceiling.py` et `sweep.py` (scenes) sont touchés (confirmé par l'audit n°4), ainsi que tout code du greffon qui passe `fps=` (`fit_basis.video_inputs`, `eval_lvb`, `needle`, `stream_*`), **à vérifier**.

## Coût mesuré et plan de vitesse

**Profil** (`prof_cost.py`, carte 1 partagée) : le coût est **linéaire en jetons de contexte, ≈ 0,21 ms par jeton** (key 7 751 jetons : 1,63 s ; fenêtre seule 2 108 jetons : 0,42 s ; tout en 224 px : 1 991 jetons, 0,44 s). L'encodeur visuel coûte 0,1 s pour 20 s de vidéo. Il est **déjà** calculé une seule fois par remplissage et par épisode, mis en cache sur disque, puis réutilisé par toutes les mémoires et tous les D. Pic mémoire GPU : 7,8 Go.

Le « 51 min par groupe » vient des 22 couples groupe × famille faits par un seul processus. Cela fait 2,3 min et 88 passes avant par couple. Le coût vient du préremplissage du LLM, pas de la vision. Raccourcir le remplissage ne sert qu'à réduire les jetons de `key` et `mean`.

| leviers | passes / question | jetons moyens | coût d'une cellule (150 groupes = 600 passes) | échelle complète / famille / 150 groupes |
|---|---|---|---|---|
| actuel (run val) | 13 + 2 par groupe | ≈ 7 000 | **≈ 15 min** | ≈ 3,3 h |
| 1. couper les cellules redondantes : key et mean à un seul D (planéité prouvée), FIFO à deux D (dans / hors), pas de solo, charge à part | 8 | ≈ 7 000 | 15 min | ≈ 2 h |
| 2. + passé de 120 s, fenêtre de 4 s, FIFO de 40 s | 8 | ≈ 4 500 | ≈ 9,5 min | ≈ 1,3 h |
| 3. + tout en 224 px (`--res lo`, déjà codé), **si v3 reste lisible à 224 px** (plafond à vérifier) | 8 | ≈ 2 000 | **≈ 4,5 min** | **≈ 35 à 40 min** |
| 4. 2 processus par carte quand ≥ 20 Go sont libres | | | ÷ 2 en temps réel | |

Le traitement par lots n'apporte presque rien ici : avec 2 000 à 7 000 jetons, une passe sature déjà la carte (≤ 15 % attendu, non mesuré, la carte n'était pas libre). Le cache de l'encodeur est déjà en place.
**Coût attendu par cellule au régime v3 visé : ≈ 4,5 min GPU pour 150 groupes.** Une échelle complète de 8 cellules sur 4 familles prend environ 2,5 h sur une carte.

## Brancher v3
`LADDER_GEN=/chemin/gen_scenes_v3.py python ladder.py ...`. API attendue : `make_group(fam, g)` qui renvoie les items (`episode`, `prompt` commençant par `INTRO`, `options`, `answer`, `variant`), plus `render(ep)`, `INTRO` et `GIST`. Le snapshot doit être copié dans `ladder/` avant le run (le hash du fichier entre dans la clé du cache des épisodes). Ensuite : plafond à 224 px, puis le plan 1 à 3 ci-dessus. Au moment de ce bilan, aucun `bilan.md` v3 n'existait dans `scenes/`.

## Problèmes ouverts
1. **`mean` et le gist** : comptage statistique ou mémoire ? Il faut une borne Bayes sur l'histogramme de couleurs avant d'interpréter un gain gist d'une mémoire compressive.
2. **Charge** : la question doit désigner le clip cible sans ambiguïté (par exemple un décor unique), sinon le gist s'effondre pour une raison qui ne tient pas à la mémoire.
3. **Remplissage** : 120 vidéos FineVideo, le corpus d'entraînement du greffon (audit n°16). Il faut un ensemble tenu à part.
4. **Puissance** : n = 30 groupes par famille suffit pour les marches de 50 à 75 points mesurées ici, pas pour des écarts de 5 points entre mémoires entraînées. Il en faut 150 à 250 groupes par cellule (audit n°13), ce que le plan de vitesse rend faisable.
5. Perception Test n'a pas été branché : l'audit n°5 montre que l'a priori textuel égale le plafond. Il n'apporterait rien à la validation de l'échelle.
