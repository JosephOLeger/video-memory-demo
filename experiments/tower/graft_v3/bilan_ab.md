# Bilan A/B « essentiel seulement » contre « tout » (23/10/2026)

**État : en pause sur décision de Josep (priorité au texte).** Aucun long run n'a été lancé. Toutes mes files et préparations sont arrêtées (PID exacts). Je n'occupe plus aucune carte. Dossier : `~/work/iclr/graft_v3/` sur la tour. Rien n'a été modifié dans `titans-mac/src`, `data/`, `cache/`, `runs/`, `scenes/`, `ladder/`. `train_v3.py` est intact.

## 1. Ce qui existe et marche

| fichier | rôle | état |
|---|---|---|
| `gen_ab.py` | scènes riches d'entraînement (split train) + questions libres denses | testé, planches `contact_ab.png` (ancien v3c) et `contact_ab_v3cb.png` (v3c actuel) |
| `prep_ab.py` | base PCA, banque de remplissage, groupes d'entraînement + enseignant, données d'éval + contrôles | testé (auto-contrôles au démarrage) |
| `train_ab.py` | entraînement A / B, mémoire `deep` ou `pool`, éval banc v3 | smokes A, B, pool-A OK |
| `queue_ab.py` | file : une tâche par carte (verrou par carte), ≥ N Mo libres, carte 2 interdite tant que `queue3.sh` tourne, porte `--gate` | testée sur la préparation |
| `diag_teacher.py` | diagnostic de l'enseignant en réponse libre | fini, §4 |
| `slice_basis.py` | tronque la base 1024 à k dims (balayage de taille sans refaire l'ajustement) | |
| `jobs_ab.txt` | les 8 runs prévus (A s0 à s2, B s0 à s2, pool-A s0, pool-B s0) | **non lancé** |

### Conception
* **Scènes riches** (une par variante, 4 variantes par groupe, même pièce, même remplissage, même fenêtre, même D) : personnes (1 à 8, couleurs de chemise distinctes), leurres v3c, porte, lumière, gros objet (60 %), panneau numéroté (60 %), chapeau (60 %, couleurs train seulement), chaussures colorées (60 %), petite affiche avec une forme (60 %).
* **Questions** générées depuis les faits du générateur, réponses courtes libres, 2 ou 3 gabarits par famille, gabarits d'éval tenus à part. Gist (A et B) : count, colour, big_object, room_state (door, light, les deux). Détail (B seulement) : sign_number, hat_color, shoe_color, poster_shape. `side` n'est jamais entraînée.
* **Équité A/B.** A tire une question par famille gist (4). B tire **les mêmes** questions gist, plus jusqu'à 4 questions de détail. Perte = moyenne gist + moyenne détail, à poids 1:1. Le gradient gist de B égale donc celui de A en espérance. B diffère seulement parce qu'il doit **aussi** garder les détails. C'est un choix à valider : la consigne disait seulement « toutes les questions ».
* **Distracteurs** (parade n° 1, info v4) : 1 à 4 « calques » par flux d'entraînement. Ce sont les mêmes personnages, leurres, panneau et objet, mais dehors, sur le décor, sans pièce. Ils sont partagés par les 4 variantes et placés partout dans le passé, y compris entre la cible et la fenêtre. La mémoire ne peut donc pas garder simplement « les images dessinées ». Limite : la pièce reste reconnaissable à ses murs. Une version plus dure mettrait des pièces distractrices avant la cible, mais la question deviendrait ambiguë pour l'hôte (bilan_v3 §2).
* **Perte** : KL(enseignant ‖ élève) sur le top-20 des jetons de réponse (plus un seau « reste »), CE sur la vraie réponse, et contrastif `relu(1 + CE(propre) − CE(autre variante))`. Le terme contrastif ne compte que si la réponse d'or est fausse pour l'autre variante. Enseignant = Qwen au format oracle de l'échelle (tableau en vidéo mémoire + fenêtre). La question est cachée à l'écriture : la mémoire n'écrit que les clips du passé.
* **Options ajoutées après le diagnostic** (§4) : `--kl-all` (par défaut, la KL ne compte que si l'argmax de l'enseignant est juste ; la CE sur l'or s'applique partout) et `--lam-mcq` (KL sur la distribution de l'enseignant sur les lettres de la version QCM de la même question, calculée dans le même lot de suffixes). La prep stocke `mcq_lp` et `mcq_ok`. **Ces deux options n'ont pas de smoke** : les données smoke sont antérieures à ces champs.
* **Mémoire** : d_mem 512, memory_hidden 256, soit w0 [256×512] + w1 [512×256] = **1,00 Mio fp32** (affiché au démarrage). Balayage de taille prévu par `--memory-hidden` à d fixe : h = 16 donne 64 Kio, h = 1024 donne 4 Mio.
* **Témoin pool** (`--mem-type pool`) : `SlotMemory` de `baselines/bl.py`, inchangée. 512 cases de 512, soit 1,00 Mio. Par défaut P = ⌈écritures / cases⌉ = 3 à stride 4 (1 280 écritures, les 1 536 dernières sont gardées) : **tout le passé tient**. Un P plus petit ne garde que les `slots × P` dernières écritures.
* **Base PCA** : refaite avec le chemin fps corrigé (spg 1,0 s vérifié). Couche 24, 115 200 états de la split train (25 600 de remplissage, 51 200 de clips avec tableau, 38 400 de l'élève : fenêtre + question). k = 1024 explique 93,4 % de la variance ; tranche 512 dans `cache_ab/qwen_l24_pca512_fpsfix.pt`. Rapport des écarts de canaux : 44.
* **Sous-échantillonnage** : 1 jeton sur 8 par seconde, avec un **décalage tournant** (la seconde g garde les positions g%8 + 8j). 8 secondes couvrent ainsi les 256 positions ; prep_v3 ne gardait que les colonnes 0 et 8. `--mem-stride k` garde dans le groupe g les jetons j ≡ g mod k : la phase tourne aussi.
* **Vitesse** : préfixe (fenêtre ou tableau + fenêtre) calculé une fois avec cache KV, puis lot des suffixes de questions. Contrôles au démarrage : préfixe + suffixe contre passage direct, écart relatif max 1,7e-2 et argmax identique à 100 % ; jetons visuels injectés contre processeur, p(or) à ± 0,03 près.

### Remplissage : procédural plutôt que FineVideo (raisonnement chiffré)
* L'éval est le banc v3 (décor procédural de ladder3), et les seuils à battre (key 45 à 50 %, mean) y ont été mesurés. FineVideo ajouterait un transfert de domaine non mesuré.
* Coût : banque procédurale de 120 flux × 30 clips en 61 min (≈ 30 s par flux, carte partagée). Le premier pool FineVideo de prep_v3 n'avait que 60 vidéos.
* Le contraste A/B porte sur le contenu de la pièce. Le remplissage est une nuisance, et la pression vient des calques distracteurs.
* v4 (calques sur FineVideo, 92 à 100 % sans charge) serait le bon banc **si** on passe à FineVideo. Je le garde pour une éval de transfert, pas pour la première comparaison.

### Stride mémoire : 4 (écrit dans jobs_ab.txt)
Mesures en lot de 4, sur une carte partagée à 100 % d'utilisation par d'autres :

| stride | écritures par flux | jetons du tableau écrits | s/pas | pic |
|---|---|---|---|---|
| 1 | 5 120 | 192 | OOM (carte partagée) | |
| 2 | 2 560 | 96 | 37 à 39 | 11,7 Gio |
| 4 | 1 280 | 48 | **21 à 22** (A), 20 (B) | 11,4 Gio |
| pool, stride 4 | 1 280 | 48 | 8 | 10,6 Gio |

À stride 4 : écriture propre 3,2 s, écriture de l'autre variante 3,3 s, élève 4,4 s, rétropropagation de l'écriture 10 s. `--chunk 16` ne fait rien gagner. `--ckpt-seg 64` est nécessaire, sinon la rétropropagation de l'écriture déborde. 3 000 pas ≈ 17 à 18 h par run sur carte partagée. 8 runs ≈ 140 h-GPU, soit 2 à 3 jours sur 2 ou 3 cartes. Éval finale (100 groupes × 7 familles × 2 D) ≈ 75 min par run ; éval périodique (20 groupes, D = 120) ≈ 8 min.

## 2. Smokes (30 pas, 40 groupes smoke, ancien v3c « stack »)
* A : CE gist 1,32 → 1,23, KL 0,96 → 0,81, écart contrastif +0,02. Éval (3 groupes par famille) : hasard, 0 % contrastif par groupe. Attendu à 30 pas.
* B : tourne, CE détail 2,0, KL détail 1,38.
* pool-A : tourne (un bug de rétropropagation est corrigé : les cases sont des moyennes de z, il n'y a pas de paramètre d'écriture). CE gist 1,32 → 1,08.
* Contrôles de l'éval corrects : fenêtre 0 % par groupe, oracle 100 % contrastif sur le détail. Sur count et colour, l'oracle vaut 0 % contrastif dans le smoke (3 groupes, ancien v3c), ce qui est cohérent avec l'échec de la version « stack ».

## 3. Données produites (`data_ab/`, 3,3 Go)
* `fill/` : 120 flux procéduraux train, z [30, 256, 512] fp16. **Complet.**
* `eval/` (partiel, préparation arrêtée) : big_object 100, room_state 100, side 80, sign_number 50, hat_color 10 groupes ; count et colour 0 (attendent v3c). Chaque groupe contient D ∈ {0, 120} et les contrôles hors greffe : lettres fenêtre / oracle / tableau d'une autre variante, réponses libres oracle / fenêtre. La reprise est automatique (on relance la même commande).
* `train/` : **vide.** Je l'ai arrêtée exprès : mes scènes utilisaient les leurres « stack » de v3c, que le 3B compte comme des personnes.
* `data_ab_smoke/` : 40 groupes d'entraînement, 3 groupes d'éval par famille (ancien v3c).

## 4. Diagnostic de l'enseignant en réponse libre (`logs_ab/diag_teacher.{log,jsonl}`)
40 groupes neufs × 2 variantes, scènes riches avec le v3c actuel (flat + posters muraux, instantané `ab_gen_v3cb/`, md5 e3e13bea). Enseignant au format oracle. « strict » = réponse gloutonne identique à l'or. « lenient » = après normalisation (casse, ponctuation, articles, nombres en lettres, quelques synonymes, mot unique dans une phrase). « mcq » = même question en lettres, au style du banc.

| famille | n | strict | lenient | QCM | p(or, 1er jeton) |
|---|---|---|---|---|---|
| count | 80 | 73,8 | 73,8 | 83,8 | 0,61 |
| colour | 60 | 66,7 | 66,7 | 80,0 | 0,46 |
| big_object | 56 | 48,2 | 53,6 | 92,9 | 0,34 |
| door | 80 | 90,0 | 90,0 | 100 | 0,63 |
| light | 80 | 75,0 | 75,0 | 71,2 | 0,57 |
| room_state | 80 | 27,5 | 27,5 | 46,2 | 0,83 |
| sign_number | 48 | 100 | 100 | 100 | 0,99 |
| hat_color | 52 | 59,6 | **100** | 100 | 0,51 |
| shoe_color | 48 | 56,2 | 64,6 | 85,4 | 0,34 |
| poster_shape | 47 | 44,7 | **97,9** | 74,5 | 0,41 |

Lecture :
1. **Le format n'explique le bas score que pour le détail.** hat_color passe de 60 à 100 et poster_shape de 45 à 98 après normalisation : l'enseignant répond juste mais le formule autrement. Sur le gist, la normalisation ne change presque rien (gist 64,0 → 64,9 %).
2. **Les erreurs gist sont réelles :**
   * count se trompe d'une unité (1 → 2, 6 → 7) ; 53 % à n = 1, 42 % à n = 6, 100 % à n = 3 et 5 ;
   * big_object nomme mal (box → « table », cone → « triangle », ball → « circle ») mais vaut 93 % en QCM ;
   * room_state se trompe surtout sur la lumière (light 75 % seul, 71 % en QCM).
3. **Même en QCM, l'enseignant reste sous le plafond du banc** (gist 78 % contre 92 à 100 % sur le banc). Les scènes riches sont plus dures que les tableaux du banc : jusqu'à 8 personnes, leurres, chapeaux, affiches, pièces sombres. Le bruit vient en partie de la richesse des scènes, pas seulement du format libre.
4. **Part gardée par le filtre « KL seulement si l'enseignant a juste »** (argmax forcé = or) : **64,0 % des questions gist** (436) et 65,1 % des questions de détail (195). Par famille : count 72, colour 67, big_object 50, door 89, light 75, room_state 27.

Propositions avant tout long run (non appliquées) :
* filtre KL activé (déjà le défaut de `train_ab.py`) ;
* jugement « enseignant juste » par la normalisation lenient plutôt que l'argmax strict. Sinon hat_color et poster_shape perdent 40 à 50 % de leur KL pour une simple question de format ;
* `--lam-mcq 1` pour big_object (93 % en QCM) ;
* simplifier les scènes riches vers le régime du banc (n ≤ 4 personnes sur les questions count, lumière plus contrastée), ou garder la richesse et accepter un enseignant à ~65 % sur le gist. À trancher par Josep.

## 5. Problèmes ouverts
1. **v3c non certifié et son design a changé** (stack → flat pour count, posters muraux pour colour). `gen_ab.sample_rich_v3c` suit déjà le nouveau design dès que le module v3c a `FEET_FLAT`. Il reste à copier la version **certifiée** dans `ab_gen/` (md5 dans `ab_gen/MD5`), à revoir une planche, puis à relancer la préparation train et l'éval count/colour.
2. Enseignant faible sur le gist (§4).
3. pool : seulement un smoke ; P par défaut = tout le passé tient. À fixer avec le lead si l'on veut un pool qui ne tient pas.
4. Distracteurs : les calques extérieurs n'empêchent pas une règle « garder les pièces ». Des pièces distractrices exigeraient une désignation de la cible que l'hôte 3B ne sait pas faire.
5. Les options `--lam-mcq` et `--kl-all` n'ont pas de smoke (il faut des données avec `mcq_lp`).

## 6. Commandes de relance
Environnement commun :
```
cd ~/work/iclr/graft_v3
export HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/work/titans-mac/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/titans/bin/python
```
1. Après certification v3c : copier `../scenes/gen_tableau_v3c.py` dans `ab_gen/`, puis `md5sum ab_gen/*.py > ab_gen/MD5`, puis `$PY gen_ab.py --contact contact_ab_v3c.png`.
2. Préparation (reprise automatique, une tâche par carte, ≥ 12 Go libres) :
```
cat > jobs_prep3.txt <<EOF
prep_eval_v3_s0 prep_ab.py eval --families big_object,room_state,side,sign_number,hat_color --groups 100 --shard 0 --nshards 2
prep_eval_v3_s1 prep_ab.py eval --families big_object,room_state,side,sign_number,hat_color --groups 100 --shard 1 --nshards 2
prep_eval_v3c   prep_ab.py eval --families count,colour --groups 100
prep_train_s0 prep_ab.py train --nfill 120 --groups 3000 --shard 0 --nshards 4
prep_train_s1 prep_ab.py train --nfill 120 --groups 3000 --shard 1 --nshards 4
prep_train_s2 prep_ab.py train --nfill 120 --groups 3000 --shard 2 --nshards 4
prep_train_s3 prep_ab.py train --nfill 120 --groups 3000 --shard 3 --nshards 4
EOF
setsid nohup $PY queue_ab.py jobs_prep3.txt --need 12000 > logs_ab/queue_prep3.log 2>&1 < /dev/null &
```
Attention : les logs `prep_eval_v3_s*.log` existants ne contiennent pas de marqueur de fin, donc la file relancera bien ces tâches.

Coûts mesurés sur carte partagée : 13 s par groupe d'entraînement (3 000 groupes ≈ 11 h-GPU) ; ≈ 40 s par groupe d'éval (≈ 8 h-GPU pour 7 × 100).

3. Smoke des nouvelles options sur quelques groupes : `$PY train_ab.py --arm A --lam-mcq 1 --steps 20 --eval-every 0 --final-eval-groups 3 --out runs_ab/smoke_mcq`.
4. Runs (8 runs, ≥ 20 Go libres par carte, porte `GO_AB` à créer à la main) :
```
setsid nohup $PY queue_ab.py jobs_ab.txt --need 20480 --gate GO_AB > logs_ab/queue_ab.log 2>&1 < /dev/null &
touch GO_AB
```
5. Éval seule d'un point de contrôle : `$PY train_ab.py --arm A --eval-only --init runs_ab/A_s0_at3000.pt --out runs_ab/A_s0_ev`.
6. Diagnostic de l'enseignant : `AB_V3C_DIR=ab_gen_v3cb $PY diag_teacher.py --groups 40`.

Copie locale du dépôt : non touchée (aucune modification de `src`).
