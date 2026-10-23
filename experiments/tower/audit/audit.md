# Audit adversarial du banc mémoire vidéo (22/10/2026)

Périmètre : `scenes/` (générateur md5 `fbd5a18b` = snapshot de l'échelle, puis `b970a249` avec `clock` et `qset`), `ptest/` (620 questions notées au moment de l'audit), `ladder/` (core.py, ladder.py, predictions.md, runs smoke + solo).
Rien n'a été modifié dans ces dossiers. Scripts et sorties : `~/work/iclr/audit/`.
Snapshots audités : `audit/snap1/` (générateur d'origine, celui figé dans l'échelle) et `audit/snap2/` (version qset/clock).

Notation des tests : « Bayes » = règle optimale apprise sur les groupes 0 à 2999 du générateur et testée sur les groupes 3000 à 3999. Elle ne voit QUE les traits indiqués, jamais l'épisode. C'est la borne haute de ce qu'une branche entraînée peut obtenir par ce raccourci. (item %, groupe % = les 4 variantes justes).

---

## Classement

| # | Gravité | Problème | Statut |
|---|---|---|---|
| 1 | CRITIQUE | L'état final + la durée de l'épisode donnent la réponse « gist » sans mémoire du début | mesuré |
| 2 | CRITIQUE | door_state : 11,8 % des étiquettes fausses (qset 1, figé dans l'échelle) | mesuré + vu à l'image |
| 3 | CRITIQUE | 4 familles sur 8 : le « plafond » Qwen n'est qu'un biais de lettre | mesuré |
| 4 | CRITIQUE | ceiling.py / sweep.py : temps M-RoPE compressé 12x ; le sweep de design est réglé sur un encodage faux | mesuré |
| 5 | CRITIQUE | Perception Test : l'a priori textuel égale Qwen avec la vidéo entière ; full − window = 3 points | mesuré |
| 6 | ÉLEVÉ | Contrôle « other » : autre groupe, donc autres options et autre décor. Pas un contrefactuel | lecture du code |
| 7 | ÉLEVÉ | Graines déterministes partagées, 36 décors seulement : train = eval si le générateur sert à entraîner | lecture du code |
| 8 | ÉLEVÉ | Le nouveau `clock` affiche la durée de l'épisode sur chaque image de fin | lecture du code + Bayes |
| 9 | ÉLEVÉ | Sélection et mesure sur les mêmes groupes (malédiction du gagnant) | lecture du code |
| 10 | MOYEN | first_left mesure surtout « qui est parti », pas « qui est parti en premier » | mesuré |
| 11 | MOYEN | Mémoire `key` : phase fixe par groupe, donc D=30 et D=150 échantillonnent exactement les mêmes secondes | lecture du code + smoke |
| 12 | MOYEN | FIFO à 224 px contre 448 px ailleurs : résolution et rétention confondues sur le détail | lecture du code |
| 13 | MOYEN | Puissance : 25 à 50 groupes ne suffisent pas pour 5 points ; graines | calcul |
| 14 | MOYEN | PT : scènes quasi dupliquées (même pièce, même script, réponses différentes) | mesuré |
| 15 | MOYEN | PT : contamination probable de Qwen ; options « je ne sais pas » jamais justes | mesuré en partie |
| 16 | FAIBLE | Remplissage : 48 vidéos FineVideo seulement, issues du corpus d'entraînement du greffon | lecture du code |
| 17 | FAIBLE | réponse = index de variante : un ordre d'évaluation fixe fuit vers une mémoire à état | lecture du code |
| 18 | FAIBLE | Statistiques de couleur moyennées (`mean`) peuvent porter un comptage sans mémoire épisodique | hypothèse |

Ce qui tient (vérifié, pas de problème) : en fin d'épisode, aucun agent sorti ne reste au bord du cadre (retiré à p = 1, vérifié dans le code et à l'image). Le décor est commun aux 4 variantes d'un groupe, donc il ne porte pas la réponse. L'a priori texte vaut exactement 25 % par item et 0 % par groupe dans les 8 familles. Les marginales sont plates : couleur du premier sortant χ² = 5,9 (7 ddl), case du premier sortant χ² = 3,3, couleur du chapeau χ² = 6,7. Les lettres de réponse sont uniformes par construction. init_count et sign_number sont justes pour la bonne raison (voir #3).

---

## 1. CRITIQUE : la fin de l'épisode donne la réponse « gist »

**Mécanisme.** Plusieurs familles « mémoire du début » sont en grande partie déterminées par l'état final, qui est montré 5 s pendant le maintien de fin, et par la durée T de l'épisode. T = 18 + 6·(sorties + entrées) + 3·(porte) images, donc T décode exactement le nombre d'événements.
* init_count : la contrainte `nl != ne` garantit que le nombre final diffère du nombre initial. Exclure le compte final est donc un indice gratuit.
* first_left : les 4 couleurs proposées sont des personnes présentes au départ. La bonne couleur est toujours absente à la fin, alors que les autres le sont seulement si elles sont aussi sorties. Avec nl = 1 (1/3 des items), la réponse est la seule couleur proposée absente.
* door_state : la moitié de la réponse est l'état final de la porte.

**Preuve** (`leak_bayes.py`, `leak_bayes.json`, `leak_bayes_qset2.json`) :

| famille | Qwen full (v1) | Bayes état final | Bayes T seul | Bayes état final + T |
|---|---|---|---|---|
| init_count | 79 | **63** (grp 12) | 34 | **87** (grp 57) |
| final_count | 37 | 100 (question non mémorielle si la fin est visible) | 36 | 98 |
| n_left | 30 | 37 | **51** | **68** |
| first_left | 58 | **67** (masque des couleurs absentes) | 25 | 67 |
| door_state qset 1 | 30 | 50 | 49 | **87** |
| door_state qset 2 | – | 49 | 51 | **100 (grp 100)** |
| order | 50 | 51 | 56 | 56 |
| hat_color / sign_number | 99 / 100 | 25 / 25 | 25 / 25 | 25 / 25 |

Une branche entraînée qui ne garde que la FIN de l'épisode (exactement ce que fait une mémoire biaisée vers le récent) bat le plafond Qwen sur first_left et init_count. Elle obtient aussi 100 % sur door_state qset 2. Elle n'a pourtant aucune mémoire de l'instant interrogé.
La condition `last` de ceiling.py ne détecte pas ce risque, car Qwen non entraîné y est ANTI-corrélé (first_left last = **1 %**, init_count 27 %). Il lit l'état final et répond « la couleur visible » ou « le compte visible ». Un modèle entraîné inverse le signe en quelques pas.
Dans l'échelle actuelle, la fenêtre est du remplissage, donc pas de fuite directe. Les D actuels ne retiennent jamais un épisode en partie (FIFO 80 s, E ≤ 24 s). La fuite apparaîtra dès qu'une mémoire entraînée garde partiellement l'épisode.

**Correctifs.**
1. Durée constante : T fixe pour tout l'épisode. On pad avec des images immobiles insérées à des positions aléatoires, ou on ajoute des événements nuls (personne qui marche derrière puis revient). Ainsi T ne porte plus le nombre d'événements.
2. init_count : tirer le compte final indépendamment du compte initial. Supprimer `nl != ne` et équilibrer P(final | initial) pour qu'il soit uniforme sur les options.
3. first_left : toutes les couleurs proposées sortent (nl ≥ 4, ou options = les sortants seulement). L'absence à la fin ne discrimine alors plus rien.
4. door_state : demander seulement l'état de DÉPART (« la porte était-elle ouverte au début ? »), avec l'état final équilibré et indépendant, ou retirer la famille.
5. Ajouter à l'échelle une condition `tail` : mémoire oracle limitée au maintien de fin (hold1). Elle doit rester au hasard après entraînement. Faire de `leak_bayes.py` un test obligatoire : chaque révision du générateur doit donner Bayes(final + T) ≈ hasard pour les familles « début ».

## 2. CRITIQUE : étiquettes door_state fausses dans 11,8 % des items

**Mécanisme.** Pour « porte inchangée », le plan `[not d0, d0]` est inséré élément par élément à des positions aléatoires (`evs.insert(rng.randint(...))`). Les deux bascules peuvent donc s'inverser : l'épisode finit dans l'état opposé, mais l'étiquette reste « inchangée ». Une des bascules devient aussi un no-op invisible.
**Preuve.** `gt_check.py` recalcule la réponse depuis les faits de l'épisode rendu : 237/2000 étiquettes fausses (11,8 %), 0 dans les 7 autres familles. Planche `vis_door_state.png` (groupe 2, variante 2) : la porte est ouverte à 0 s et fermée de 9 à 17 s, alors que l'étiquette dit « open at the start and open at the end ».
Le bug est corrigé seulement en qset 2, où la double bascule est supprimée. Il reste présent en qset 1 et dans `ladder/gen_scenes_snapshot.py` (md5 identique au générateur d'origine).
**Correctif.** Insérer la seconde bascule après la première, et ajouter dans `make_group` une assertion `answer == f(facts)` pour toutes les familles. Regénérer le snapshot de l'échelle et invalider le cache `ep_door_state_*`.

## 3. CRITIQUE : plafonds faits de biais de lettre

**Preuve** (`letter_bias.py` sur `scenes/results/v1.json`) :
* order full : prédit A dans **100 %** des cas, donc 50 % = réponse constante (groupe 0).
* door_state full : A dans 95 %, d'où 30 %. full et blind ont la même distribution.
* n_left full : A 49 %, B 43 %, d'où 30 %. final_count full : D 47 %, alors que last fait 84 %.
* Groupe (4 justes) : n_left, final_count, order et door_state = **0**.

Une exactitude par item de 30 à 50 % ressemble à un signal faible, alors que c'est une sortie constante. Le plan contrefactuel est bon (il ramène tout biais de lettre à 25 %), mais seulement si on lit le score par groupe.

Bonne nouvelle, vérifiée par perturbation (`ceiling_probe.py`, 8 groupes, partiel) :
* init_count : full 78, 4 premières secondes seules 88, début + fin sans les événements 81, 1 image sur 2 75, images mélangées 56. Qwen lit bien l'état initial.
* sign_number : 100 / 100 / 100 avec images mélangées, 31 avec le début seul, 25 sans le milieu. Il lit bien le panneau.
* hat_color : 99 (v1).

**Correctif.** Certifier uniquement sur l'exactitude par groupe (ou par paire) et sur un score débiaisé (logits de lettre moins la moyenne du groupe). Retirer order, door_state, n_left et final_count du banc tant que leur plafond par groupe reste à 0.

## 4. CRITIQUE : encodage temporel faux dans ceiling.py et sweep.py

**Preuve** (`proc_check.py`) : `proc(..., fps=2)` donne `second_per_grid_ts = 0,0833`. transformers ignore `fps=` et suppose 24 fps. Un épisode de 20 s est donc présenté comme 1,7 s.
Avec le bon temps (`video_metadata` + `do_sample_frames=False`, spg = 1,0), first_left passe de 59 à **69** et init_count reste à 78. L'échelle (`core.py`) et `eval_ptest.py` corrigent ce point. `ceiling.py` ne le corrige pas, et `sweep.py` importe son `letter_logits`.
Les choix de design en cours (qset 2, clock, ev_len 8) sont donc évalués sous un temps faux. L'horloge peut sembler aider uniquement parce que le temps M-RoPE est cassé.
**Correctif.** Construire les entrées comme dans `core.Host.build` (spg = 1,0), ajouter `assert spg == 1/FPS*2` dans ceiling.py, puis refaire v1 et le sweep.

## 5. CRITIQUE : Perception Test, l'a priori égale le plafond

**Preuve** (`ptest_priors.py` sur les 19 140 questions valid ; `ptest_blind.py` sur les 620 premières notées) :
* Majorité de réponse par gabarit (question + options) apprise sur **train** et appliquée à valid : **55 %** (couverture 75 %, hasard 33 %). En leave-one-out sur valid : 57 %. Avec le texte de la réponse seul : 58 %. Détail : material 82 %, collision 81 %, place recognition 65 %, feature matching 62-66 %. Classe gist : 52-55 %. Classe detail : 55-58 %.
* Les 948 « groupes contrefactuels » à réponse variable restent très déséquilibrés : la majorité du groupe gagne 57 % du temps.
* Options de type « je ne sais pas / on ne peut pas juger » : présentes dans 4 545 questions, justes 6 fois. Les éliminer donne 50 % sur 24 % des questions.
* Qwen sur 620 questions : full 55, blind 44, **window (4 s) 52**. Gist : full 44, blind 40, window 41. Sur les questions couvertes par un gabarit, l'a priori train (55) ≥ Qwen full (55).

C'est exactement l'échec n°1 déjà subi : une branche entraînée sur les gabarits PT (ou sur des QA du même style) atteint le « plafond » sans aucune mémoire. Et le plafond ne dépasse la fenêtre que de 3 points, donc il n'y a presque rien à mesurer.
**Correctifs.**
1. Sous-ensemble équilibré : pour chaque gabarit, sous-échantillonner pour que chaque option soit juste le même nombre de fois, puis noter par gabarit équilibré.
2. Toujours rapporter le gain au-dessus de max(blind, a priori gabarit appris sur les données d'entraînement du greffon, contrôle other-video).
3. Retirer les options IDK, ou les compter comme hasard à 50 %.
4. Choisir les items par la sémantique (preuve hors de la fenêtre), pas par le succès de Qwen.
5. Ne jamais entraîner sur PT train si l'on évalue sur PT valid avec les mêmes gabarits, sauf à soustraire l'a priori appris.

## 6. ÉLEVÉ : le contrôle « other » n'est pas contrefactuel

`ladder.py` prend la mémoire oracle d'un épisode du groupe `g + 10000`. Cet épisode a un autre décor et, pour init_count, final_count, first_left, hat_color et sign_number, une **autre liste d'options**. Le champ `other_answer` indexe donc les options du donneur, pas celles de la cible : le « taux de réponse du donneur » n'a pas de sens.
Pour les familles à options fixes (n_left, order, door_state), c'est par chance un contrefactuel correct.
Un modèle qui détecte « décor différent, donc mémoire hors sujet » se rabat sur l'a priori, et ce contrôle mesure alors l'a priori et non la lecture du contenu.
Taille : la longueur varie (E de 15 à 23 groupes, 3 840 à 5 888 jetons) comme pour l'oracle. Le format est le même (448 px).
**Correctif.** other = **même groupe, autre variante** : même décor, même remplissage, mêmes options, réponse différente garantie. Mesurer (a) l'exactitude sur la réponse cible, qui doit tomber SOUS le hasard, et (b) le taux de réponse du donneur, qui doit monter au niveau de l'oracle. C'est le test le plus tranchant contre le scénario « la mémoire d'une autre vidéo marche aussi bien ». Garder la version inter-groupes en secondaire.

## 7. ÉLEVÉ : graines partagées, peu de décors

`make_group` est entièrement déterminé par `f"{family}-{gseed}"`. ceiling v1 utilise les groupes 0-24, l'échelle 0-49 (+10 000 pour other, 20 000 + 97g + k pour les distracteurs), le sweep 100-109. Si les données d'entraînement viennent du même générateur avec des indices à partir de 0, l'évaluation EST l'entraînement.
De plus il n'y a que 4 bases de mur × 3 sols × 3 positions de porte = 36 pièces, 8 couleurs, un seul gabarit par famille : le décor et le texte d'évaluation sont tous vus à l'entraînement.
**Split proposé.**
* Sel de split dans la graine : `f"{split}-{family}-{gseed}"`, avec train et eval disjoints par construction.
* Décors tenus à part : 4e base de mur et 3e sol jamais en train, porte en x = 320 réservée à l'eval.
* Détail : 2 couleurs sur 8 jamais utilisées comme chapeau en train. Panneau : police ou position différente en eval.
* Gabarits : une paraphrase par famille réservée à l'eval (par exemple qset 1 en train, qset 2 en eval).
* Famille tenue à part pour toute affirmation de généralisation (par exemple entraîner sans n_left).

## 8. ÉLEVÉ : `Style.clock` divulgue la durée

La nouvelle horloge affiche `f // FPS` secondes depuis le DÉBUT de l'épisode. Chaque image du maintien final montre donc T, ce qui active le Bayes « état final + T » du #1 (87 % init_count, 100 % door_state qset 2, 68 % n_left) à partir de quelques images de fin. Elle date aussi chaque image clé, ce qui transforme order en lecture de chiffres.
**Correctif.** Ne pas l'utiliser. Si une horloge est nécessaire pour le temps M-RoPE, corriger plutôt le #4. À défaut, afficher l'heure absolue du flux avec un décalage aléatoire par épisode.

## 9. ÉLEVÉ : sélectionner et mesurer sur les mêmes groupes

`predictions.md` certifie une famille sur `solo` / `oracle` puis compare les mémoires sur les mêmes 25-50 groupes. Choisir les familles et réglages (sweep) où Qwen réussit sur ces items gonfle l'oracle par régression vers la moyenne.
**Correctif.** Certifier et régler le design sur les groupes A, mesurer sur des groupes B frais, fixés à l'avance.

## 10. MOYEN : first_left ne mesure guère l'ordre

Images mélangées dans le temps : 53 % contre 59 % (full). Avec nl = 1 (1/3 des items), la question se réduit à « qui est parti ». Et le #1 montre que l'absence finale suffit à 67 %.
**Correctif.** Toutes les couleurs proposées sortent, nl ≥ 2 imposé, réponse = la première. Valider que le score avec images mélangées tombe au hasard.

## 11. MOYEN : phase des images clés

La phase est tirée par groupe, et les positions clés sont ancrées sur la fin du passé. Comme 30 ≡ 150 (mod 8), key@30 et key@150 voient exactement les mêmes secondes de l'épisode : la « platitude en D » prédite (P5) est garantie entre ces deux D, pas mesurée. Smoke hat_color : key@0 = key@30 = key@150 = 100, key@90 = 25 (n = 4).
**Correctif.** Tirer la phase par (groupe, D), ou mieux, moyenner sur les 8 phases (espérance exacte, 8x le coût de key seulement).

## 12. MOYEN : résolution confondue avec la rétention

FIFO est à 224 px (64 jetons/s), key, mean et oracle à 448 px. Pour sign_number et hat_color, fifo < oracle mesure la résolution. predictions.md l'anticipe, mais la comparaison fifo/key sur le détail reste confondue.
**Correctif.** Ajouter `fifo448` à budget égal (20 s à 448 px) ou `key224`.

## 13. MOYEN : puissance et graines

`regularities.txt` : pour une différence appariée de 5 points (McNemar, α = 0,05, puissance 0,8), il faut 312 / 469 / 626 / 940 items pour un taux de discordance de 0,10 / 0,15 / 0,20 / 0,30. Sans appariement, 1 500 par bras.
Les 4 variantes d'un groupe sont corrélées : multiplier par l'effet de plan 1 + 3·ICC, soit ~150-250 groupes par cellule (mémoire × D × catégorie), pas 25-50.
Graines des mémoires entraînées : avec l'écart de 7,6 points déjà observé sur le greffon (SD ≈ 4), il faut **~11 graines par bras** pour 5 points. Avec SD 3, 6 graines ; avec SD 2, 3.
**Correctif.** Bootstrap par groupe, score continu apparié (marge de logit de la bonne lettre) pour réduire la variance, ≥ 5 graines et modèle mixte (graine, groupe) avant toute conclusion à 5 points.

## 14. MOYEN : PT, quasi-doublons de scène

`ptest_dupes.py` (vignette 32×18 à t = 1 s, 1 500 vidéos valid) : 2,9 % ont un voisin à cos > 0,9 et 1,3 % à cos > 0,95. Par exemple video_7620 / video_7696 : même pièce, même personne, même script (« combien d'objets distincts ? »), réponses 4 contre 3. Sur les 5 260 vidéos, la fraction est plus grande (≈ 3,5x plus de voisins candidats).
Si un remplissage PT est tiré du même cluster que l'épisode, il peut confirmer ou contredire la réponse.
**Correctif.** Regrouper par similarité de vignette (> 0,9) et par ensemble de gabarits, puis exclure tout remplissage du même cluster que l'épisode. Le remplissage FineVideo de l'échelle évite ce problème.

## 15. MOYEN : PT, contamination et options mortes

Le split PT train figure dans des mélanges SFT vidéo courants. Rien ne garantit que Qwen2.5-VL ne l'a pas vu. Indice compatible : les réponses blind de Qwen coïncident avec la majorité train du gabarit dans 53 % des cas (≈ 33 % attendu). Ce n'est pas une preuve.
Conséquence : le « full » de Qwen sur PT peut être en partie de l'a priori mémorisé.
**Correctif.** Ne mesurer que des écarts relatifs, sur le sous-ensemble équilibré du #5.

## 16. FAIBLE : remplissage

`fv_ids(48)` : 48 vidéos seulement (durée ≥ 200 s), tirées de FineVideo, qui est le corpus d'entraînement du greffon. La mémoire entraînée aura vu le remplissage.
**Correctif.** Remplissage pris dans des identifiants FineVideo exclus de l'entraînement, et ≥ 200 vidéos distinctes.

## 17. FAIBLE : réponse = index de variante

Dans les 8 familles, `answer == variant` (ou `variant % 2`). Si une mémoire à état traverse les items d'un groupe dans l'ordre sans remise à zéro, la réponse se prédit à partir de la position.
**Correctif.** Remettre l'état à zéro par item, mélanger l'ordre d'évaluation, ne jamais exposer `variant`.

## 18. FAIBLE (hypothèse) : `mean` et la masse de couleur

Smoke : init_count mean@150 = 4/4, mean@90 = 3/4 (n minuscule). Des jetons moyennés gardent la masse de couleur des chemises, ce qui donne un signal de comptage global (le début et la fin sont mélangés), pas une mémoire de l'instant. À vérifier sur le run principal.
**Correctif** si le signal persiste : Bayes sur l'histogramme de couleur moyen de l'épisode entier comme borne « gist par statistique de pixels ».

---

## Note de conduite

Ma sonde GPU (`ceiling_probe.py`, carte 1) a coexisté avec `sweep.py` de l'autre agent. Ce sweep avait fait un OOM à 11:03 (`scenes/results/sweep1_oom.log` cite mon PID 2404851), puis il a été relancé à 11:14. J'ai tué ma sonde à ~11:20 après 3 familles. Ses résultats partiels sont dans `ceiling_probe_partial.txt`. Aucun autre processus touché.

## Fichiers

`leak_bayes.py` / `.json` / `_qset2.json`, `gt_check.py`, `letter_bias.py` + `letter_bias_v1.txt`, `ceiling_probe.py` + `ceiling_probe_partial.txt`, `proc_check.py`, `regularities.py` + `.txt`, `ptest_priors.py` + `.txt`, `ptest_blind.py` + `.txt`, `ptest_dupes.py` + `.json` + `.log`, `vis.py` + `vis_*.png`.
