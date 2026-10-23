# Témoins à budget égal sur l'aiguille couleur en flux (22/10)

## Question

La mémoire profonde greffée sur Qwen2.5-VL-3B gelé (C1_color_s0) fait 100 % avec la mémoire de sa vidéo contre 27 % avec la mémoire d'une autre vidéo. Une mémoire simple, de même taille en octets, fait-elle aussi bien ? Si oui, la tâche couleur ne distingue pas les modules de mémoire.

## Run de référence (retrouvé)

- `~/work/iclr/stream/C1_color_s0` (orch.log : lancé le 16/10 16:23, fini le 17/10 03:05, carte 2, **10,7 h**).
- Commande : `python -m titans.graft.train_stream --targets "stream/tgc/*.pt" --kinds color --seed 0 --basis cache/qwen_l24_pca128.pt --max-past 8 --lam-lat 0 --lam-con 0 --lr 3e-3 --steps 4000 --batch 4 --eval-every 500 --max-eval 300` (CE seule, AdamW, warmup 100 + cosinus, clip 1).
- Données : 1219 vidéos (`tgc`), 4274 questions d'entraînement, 300 questions d'éval sur des vidéos jamais vues. Passé = au plus 8 clips de 16 s, 45 tokens par clip, soit **au plus 360 écritures** de 128 dimensions (PCA de la couche 24). QCM à 4 lettres, hasard 25 %.
- Protocole d'éval : mémoire de la vidéo / mémoire d'une autre vidéo / fenêtre seule / enseignant (preuve visible), par distance en clips (2-4, 4-10) et sur le sous-ensemble dur (la fenêtre seule se trompe, n = 210).
- Mémoire profonde : d_mem 128, memory_hidden 64, `update_bias=False`. Contenu écrit = w0 [64×128] + w1 [128×64] = **16 384 flottants = 64 Kio en fp32**. L'élan S a la même taille et n'est pas compté : les témoins ont donc **deux fois moins** d'état porté que la profonde.

## Témoins

Code : `~/work/iclr/baselines/bl.py`. Rien n'est modifié dans `src`. Le module remplace `MemoryBranch` et `write_stream` dans l'espace de noms de `train_stream`, puis appelle `train_stream.main()` sans le toucher. Données, ordre des lots, hyperparamètres et éval sont donc ceux de C1 (CE au pas 0 : 0,602 dans tous les runs, comme C1). Restent identiques : la PCA `down` gelée, `up` initialisé à zéro, `norm_r`, et la requête `q_proj(z)` lue sur chaque token de l'hôte. Seule la mémoire change.

| Témoin | État porté | Écriture | Lecture |
|---|---|---|---|
| `linear` | M [128×128] = 16 384 fl. | règle delta (DeltaNet) : M ← (1−a)M + b(v − Mk)kᵀ, ‖k‖ = 1, a et b par portes sigmoïdes | M q/‖q‖ |
| `pool` | 128 cases × 128 = 16 384 fl. | moyenne de 3 écritures consécutives par case, FIFO sur les cases (360 écritures donnent 120 cases : tout le passé tient) | softmax(τ q·k) sur les cases, k = k_proj(case), v = v_proj(case) |
| `fifo` | 128 × 128 = 16 384 fl. | les 128 dernières écritures telles quelles (2,8 clips) | même lecture que `pool` |
| `mean` | 128 fl. (**128 fois moins**) | moyenne de tout le passé | v_proj(moyenne), indépendante de la requête (borne basse) |

## Résultats (éval n = 300, pas 4000, en %)

| Run | Mémoire | Autre vidéo | Fenêtre | Enseignant | Dur (mém./autre) | Dist 2-4 (mém./autre) | Dist 4-10 (mém./autre) | Premier pas ≥ 90 | Durée |
|---|---|---|---|---|---|---|---|---|---|
| profonde s0 (C1) | **100** | 27 | 30 | 99 | 100/26 | 100/29 | 100/26 | 2000 | 10,7 h |
| profonde s1 | 26 | 22 | 30 | 99 | 25/18 | 22/21 | 30/22 | jamais | 10,7 h |
| profonde s2 | 36 | 24 | 30 | 99 | 33/23 | 38/22 | 34/26 | jamais | 10,7 h |
| **pool s0** | **100** | 29 | 30 | 99 | 100/25 | 100/30 | 99/27 | 2500 | **1,7 h** |
| linear s0 | 28 | 24 | 30 | 99 | 29/23 | 30/26 | 27/22 | jamais | 3,1 h |
| linear s1 | 25 | 26 | 30 | 99 | 26/27 | 28/27 | 23/25 | jamais | 3,4 h |
| fifo s0 | 29 | 27 | 30 | 99 | 28/27 | 31/34 | 28/22 | jamais | 1,8 h |
| mean s0 | 25 | 20 | 30 | 99 | 26/19 | 28/22 | 23/18 | jamais | 1,7 h |

Trajectoires (mémoire/autre, tous les 500 pas) :
- profonde s0 : 30/30, 27/27, 26/22, **87**/27, 99/28, 100/28, 100/28, 100/27, 100/27
- pool s0 : 30/30, 27/26, 24/24, 27/21, 26/24, **95**/28, 100/28, 100/29, 100/29

Sonde CPU (`probe_mean.py`) : couleur parmi 6 (hasard 17 %) décodée par une régression logistique sur la moyenne du passé PCA. Moyenne globale : 36 % en éval (40 % à distance 2-4, 33 % à distance 4-10). Moyenne des 128 dernières écritures : 31 % (51 % à distance 2-4, 17 % au-delà). La moyenne globale garde un peu de couleur, mais diluée.

**Graines restantes.** Il manque pool, fifo et mean s1, puis les quatre en s2. Elles sont en file sur la carte 2 (`queue2.sh`, `jobs.txt`), mais la carte est occupée depuis 15 h 30 par `ladder3.py` et `ceiling_v4.py` (chantier scènes). La file attend 10,5 Go libres et relance seule. Pour regénérer le tableau : `python ~/work/iclr/baselines/table.py`.

## Lecture

1. **Oui, une mémoire simple de même taille atteint 100 % contre 29 %.** Le tampon de moyennes par 3 (`pool`), sans aucune règle d'apprentissage interne, reproduit exactement le meilleur run profond : 100 / 29 / 30 / 99, 100 % sur le sous-ensemble dur, 99-100 % aux deux distances. Il apprend en 2500 pas (2000 pour la profonde), avec deux fois moins d'état porté et 6 fois moins de temps de calcul. La tâche couleur ne sépare donc pas une mémoire profonde d'un cache moyenné.
2. **Les échecs ont tous la même forme, y compris pour la profonde.** linear s0 et s1, fifo s0, mean s0 et profonde s1 et s2 restent sur le plateau CE ≈ 0,46. Leur précision d'entraînement monte ensuite vers 35-45 % sans que l'éval bouge : c'est le « par cœur via la mémoire » déjà diagnostiqué pour profonde s1 le 20/10. À cette recette, la réussite est une loterie de graine : 1 sur 3 pour la profonde, 1 sur 1 pour pool, 0 sur 2 pour linear, 0 sur 1 pour fifo et mean. Avec ces effectifs, on ne peut pas classer linear ou fifo contre la profonde. On peut seulement dire que la profonde ne fait pas mieux que pool.
3. **mean échoue, ce qui est le seul signal net.** Une moyenne globale de 128 flottants ne suffit pas : l'aiguille y pèse au plus 1/8 et la lecture ne dépend pas de la question. Il faut garder le passé avec assez de résolution temporelle pour que le clip de l'aiguille reste séparable. L'échec de fifo s0 aux distances 2-4, où l'aiguille est pourtant encore dans le tampon, ressemble plutôt à la loterie qu'à une limite de portée. Une graine de plus le dira.

## Ce que teste la tâche couleur

- Elle teste qu'un **trait saillant et uniforme** du passé (un écran entier d'une seule couleur, qui survit à la moyenne de 3 tokens voisins) reste **récupérable** quelques clips plus tard, et qu'une **lecture apprise** relie la question textuelle à ce trait. C'est un test de bout en bout de la tuyauterie : écriture, survie, adressage depuis le texte, injection dans l'hôte. C'est aussi un test d'optimisation : sortir du plateau sans apprendre par cœur.
- Elle **ne met aucune pression de capacité**. Le passé fait au plus 360 × 128 = 46 080 flottants, soit 2,8 fois le budget. Une simple réduction temporelle par 3 le garde en entier, et l'aiguille ne perd rien à la moyenne. Il n'y a ni interférence (une seule aiguille), ni liaison entre deux contenus, ni sélection à faire.
- Le résultat 100 % contre 27 % reste vrai et utile : le contenu passe bien par la mémoire. Mais il ne prouve rien sur la **forme** de la mémoire. Il ne peut pas porter l'affirmation « une mémoire profonde est nécessaire ».

## Quelle tâche séparerait une mémoire profonde d'un cache moyenné

Il faut que le passé dépasse nettement le budget (10 à 100 fois), et que ce qu'il faut garder ne soit ni saillant ni additif :

- **Charge** : beaucoup d'épisodes (aiguilles, scènes) entre l'écriture et la question, avec un passé de milliers d'écritures pour 16 k flottants. Un cache doit alors jeter du temps (FIFO) ou moyenner fort (pool), et la profonde doit compresser. C'est l'hypothèse « sous charge » du carnet du 22/10.
- **Liaison associative** : plusieurs couleurs et plusieurs objets dans la même vidéo, puis une question du type « de quelle couleur était l'écran juste après l'entrée du chien ? ». La réponse dépend d'une paire (clé, valeur), pas d'un trait global. Une moyenne mélange les paires ; une mémoire associative peut les séparer.
- **Interférence et ordre** : la même aiguille répétée avec des couleurs différentes, puis une question sur la k-ième ou la plus récente. Un cache moyenné les confond.
- **Attention aux tâches « essentiel » additives** : compter des personnes, des entrées, des sorties. Une somme ou une moyenne en est la statistique suffisante, donc `pool` ou même `mean` peuvent y suffire. Ces questions ne sépareront les modules que si le comptage exige de lier (qui est entré, puis qui est sorti) ou de dépasser le budget.
- Dans tous les cas, garder **`pool` et `fifo` à budget égal comme témoins obligatoires** dans le nouveau banc, avec au moins 3 graines par bras : à cette recette, la loterie de graine est du même ordre que l'écart qu'on cherche.

## Fichiers

`~/work/iclr/baselines/` : `bl.py` (témoins), `queue.sh` puis `queue2.sh` (files, carte 2), `{linear,pool,fifo,mean}_s*.{log,json}` (+ `_at*.pt`), `table.py`, `probe_mean.py` / `.log`, `queue.log`.
