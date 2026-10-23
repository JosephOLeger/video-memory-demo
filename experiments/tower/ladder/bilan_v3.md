# Bilan v3 : échelle de mémoires non entraînées sur le générateur « tableaux » (22-23/10/2026)

Générateur : copie figée `ladder/v3/` (gen_tableau.py md5 `ddf4e1d1`, gen_scenes.py `cd0bfad3`), réglages v3b `TStyle(count_gap=True, door_bright=True)`, **split eval uniquement**.
Groupes de mesure : 10000 à 10149 (jamais vus par le générateur, sa certification ni l'audit). Groupes 20000 et suivants pour les contrôles de plafond (224 px, charge), afin de séparer réglage et mesure (audit n°9).
Code : `ladder3.py`, `core.py`, `analyse3.py`, `bayes_mass.py`. Données : `runs3/`. Figures : `figs3/`.

## Protocole

* Flux : P = 160 s de décor procédural v3 (identique au pixel près entre les 4 variantes d'un groupe), le tableau cible (6 s) se termine D s avant une fenêtre de 4 s de décor. 448 px, 256 jetons par seconde.
* **Budget égalisé : 12 Mio (3 072 jetons) par mémoire compressive.** oracle = 6 Mio, tail = 1 Mio.

| mémoire | contenu | Mio |
|---|---|---|
| oracle | le tableau (6 s) | 6 |
| tail | la dernière seconde du tableau | 1 |
| oracle_shuf | oracle, positions des 256 jetons permutées | 6 |
| fifo | 12 dernières s évincées, 448 px | 12 |
| fifo224 | 48 dernières s évincées, 224 px (mêmes jetons) | 12 |
| key | 1 s toutes les 13 s sur tout le passé, phase aléatoire par (groupe, D) | 12 |
| mean | passé en 12 segments de 13 s, jetons moyennés par position | 12 |
| mean_shuf | mean, positions permutées | 12 |
| empty / blind | fenêtre seule / rien | 0 |
| other | tableau d'une AUTRE variante du même groupe | 6 |

* D : fifo à 0 (dans le tampon) et 120 (hors) ; key, mean à 120. oracle, tail, empty, other ne dépendent pas de D.
* Arrêt précoce (consigne) : une cellule oracle, tail, other, empty, blind ou fifo s'arrête après 40 groupes si elle est à ≥ 95 % ou à 0 % par groupe. Les autres vont à 150 groupes.
* Scores : exactitude par question, par **groupe** (les 4 variantes justes) et **contrastive par groupe** (logits de lettre moins leur moyenne sur les 4 variantes, puis argmax ; méthode du bilan scenes). SE par bootstrap sur les groupes.
* Vérification de lisibilité à 224 px (15 groupes, groupes 20000+) : oracle par groupe à 224 px = count 60, colour 87, big_object 100, sign 100, hat 100 ; contrastif 100 partout. **Le détail reste lisible à 224 px.** Comme v3 est déjà court en jetons (≤ 4 200 par passe, 0,55 s), j'ai gardé 448 px comme base et `fifo224` comme bras.

## 1. Grille principale

### Contrastif par groupe, % ± SE (n groupes)

| mémoire | count | colour | big_object | sign_number | hat_color |
|---|---|---|---|---|---|
| oracle | 99 ± 1 (150) | 100 (95) | 100 (150) | 100 (40) | 100 (40) |
| tail (1 Mio) | 99 ± 1 (150) | 100 (95) | 100 (150) | 100 (40) | 100 (40) |
| oracle_shuf | 97 ± 1 | 99 ± 1 | 100 | 100 | 100 |
| fifo@0 | 99 ± 1 | 100 | 100 | 100 | 100 |
| fifo224@0 | 99 ± 1 | 99 ± 1 | 100 | 100 | 100 |
| fifo@120, fifo224@120 | **0** (40) | **0** | **0** | **0** | **0** |
| key@120 | 45 ± 4 | 45 ± 4 | 46 ± 4 | 50 ± 4 | 43 ± 4 |
| mean@120 | **1 ± 1** | **79 ± 3** | 28 ± 4 | 17 ± 3 | **99 ± 1** |
| mean_shuf@120 | 2 ± 1 | 67 ± 4 | 19 ± 3 | 16 ± 3 | 99 ± 1 |
| empty, blind | 0 (40) | 0 | 0 | 0 | 0 |
| other | 0 (40) | 0 | 0 | 0 | 0 |

### Brut par groupe % (par question %)

| mémoire | count | colour | big_object | sign_number | hat_color |
|---|---|---|---|---|---|
| oracle | 53 (88) | 95 (99) | 79 (95) | 100 (100) | 100 (100) |
| tail | 47 (87) | 91 (98) | 45 (85) | 100 (100) | 100 (100) |
| fifo@0 | 55 (89) | 88 (97) | 94 (98) | 100 (100) | 100 (100) |
| fifo224@0 | 44 (86) | 88 (97) | 88 (97) | 100 (100) | 98 (99) |
| fifo@120 | 0 (25) | 0 (25) | 0 (25) | 0 (25) | 0 (25) |
| key@120 | 18 (53) | 41 (58) | 33 (57) | 50 (62) | 43 (57) |
| mean@120 | 0 (26) | 25 (71) | 0 (34) | 0 (29) | 89 (97) |
| empty / blind | 0 (25) | 0 (25) | 0 (25) | 0 (25) | 0 (25) |
| other | 0 (0) | 0 (0) | 0 (1) | 0 (0) | 0 (0) |

Contrôle « autre variante » : la réponse du DONNEUR est choisie à 88 % (count), 99 % (colour), 94 % (big_object), 100 % (sign, hat). La cible tombe à 0 % par question.

![grille](figs3/grid_v3.png)
![gist contre détail](figs3/gist_detail.png)

### Lecture

1. **Le banc classe l'échelle comme prévu.**
   * Les contrôles nuls valent 0 exactement par groupe.
   * `other` suit le contenu du donneur (88 à 100 %).
   * FIFO fait une marche parfaite : 99 à 100 dans le tampon, 0 dehors.
   * Les images clés atteignent la couverture prédite : un tableau de 6 s et une clé toutes les 13 s donnent q = 6/13 = 46 % ; on mesure 43 à 50 % en contrastif sur les 5 familles.
2. **Le score brut pénalise count.** L'oracle brut ne fait que 53 % par groupe sur 150 groupes (contre 76 % sur 25 groupes dans le bilan scenes, effet de petit échantillon et de biais d'option). Le contrastif donne 99 %. **Il faut utiliser le score contrastif par groupe comme mesure principale**, le brut restant le critère de certification. Sur big_object, `tail` brut (45) est loin de l'oracle (79), alors que le contrastif vaut 100 pour les deux : là aussi l'écart est un biais d'option, pas une perte d'information.
3. **Une seconde suffit.** `tail` (1 Mio) égale l'oracle sur toutes les familles en contrastif. C'est le principe de v3 (tableau immobile), et cela fixe l'enjeu : une mémoire n'a pas besoin de compresser, elle doit **garder la bonne seconde**. La difficulté vient de la rétention et de la sélection, pas de la capacité.
4. **La moyenne temporelle ne sépare pas gist et détail. Elle sépare COULEUR et STRUCTURE.** Elle garde `hat_color` (99) et `colour` (79), deux questions de couleur. Elle perd `count` (1), `sign_number` (17) et `big_object` (28), des questions de forme ou de nombre. Le découpage gist/détail de v3 est donc **confondu avec couleur/structure** : les deux familles « couleur » tombent une de chaque côté. Avec ces familles, on ne peut pas conclure qu'une mémoire « garde le gist et perd le détail ». La prédiction « mean dégrade le détail avant le gist » vaut pour sign contre count (17 contre 1, et toutes deux perdues), pas en général.

## 2. Charge : le plafond ÉCHOUE

Dispositif : la pièce cible a des murs turquoise (jamais une couleur de chemise, jamais utilisés ailleurs). La question commence par « Only one room has turquoise walls. The question is about the room with the turquoise walls. » L distracteurs = les autres variantes du même groupe (murs ordinaires), placés ENTRE la cible et la fenêtre. Plafond = `oracle_all`, les L+1 tableaux dans la mémoire.

| groupe % (par question %) | count | colour | big_object | sign_number | hat_color |
|---|---|---|---|---|---|
| oracle_all, L = 0 (repère) | 20 (80) | 100 | 100 | 100 | 100 |
| oracle_all, L = 2 | **0 (22)** | **0 (19)** | **0 (44)** | 50 (84) | **0 (34)** |
| oracle_all, L = 5 | **0 (12)** | **0 (9)** | **0 (9)** | **0 (28)** | **0 (9)** |
| variante A : tableaux séparés par 2 s de décor, L = 5 | 0 (19) | 0 (0) | 0 (3) | 0 (9) | 0 (0) |
| variante B : un clip vidéo étiqueté par tableau (« Room clip k of L+1 »), L = 2 | 0 (34) | 0 (38) | 0 (59) | **100 (100)** | 25 (62) |
| variante B, L = 5 | 0 (34) | 0 (31) | 0 (44) | **100 (100)** | 0 (50) |

n = 8 groupes par cellule (5 pour le repère). Contrastif : 0 partout à L = 5 sauf variante B (sign 100, big_object 38, count 12, hat 12). Figure : `figs3/load.png`.

**Verdict : le plafond sous charge est loin de 80 par groupe, sur les 5 familles et les 3 mises en forme.** À L = 5, l'exactitude par question passe SOUS le hasard (9 à 12 %) : Qwen2.5-VL-3B répond sur un distracteur, il ne lie pas le repère « murs turquoise » à la bonne pièce. La seule exception est sign_number en clips séparés (100), sans doute parce qu'un nombre est une réponse « unique » par clip. Ce n'est pas la mémoire qui échoue. C'est **l'hôte qui ne sait pas sélectionner une scène parmi plusieurs de même type**. Mesurer une mémoire sous cette charge mesurerait l'hôte. Je n'ai donc PAS lancé la grille sous charge (key, mean à L = 2 et 5) : elle n'aurait pas de plafond.

Alternatives proposées, à certifier avant usage (≈ 10 min GPU chacune avec `ladder3.py --load`) :
1. **Sélection hors hôte** : la charge porte sur la MÉMOIRE (elle doit garder le bon clip parmi L+1), mais la lecture finale ne reçoit que ce que la mémoire restitue. Pour les mémoires non entraînées, cela revient à mesurer la probabilité que le tableau cible survive (FIFO de taille fixe, key avec L+1 tableaux concurrents pour le budget). C'est une mesure de rétention sous concurrence, et son plafond est l'oracle à L = 0. **Je recommande cette option** : c'est la seule où la sélectivité est celle de la mémoire.
2. Distracteurs « même type, autre famille de faits » : des pièces SANS personne, ou sans chapeau ni panneau. La question « la pièce avec des gens » se résout alors sans lier un attribut fin. Charge plus faible, mais certifiable.
3. Repère textuel fort : un grand panneau « ROOM A » dans la cible et des lettres différentes dans les distracteurs. Il faut le tester, car sign_number en clips séparés suggère que l'hôte lie mieux un symbole qu'une couleur de mur.
4. Hôte 7B pour la seule charge. À tester ; d'après scenes, le 7B n'aide pas au comptage.

## 3. Moyenne temporelle : mémoire ou statistique ?

**Borne Bayes par masse de couleur** (`bayes_mass.py`). Pour chaque item, l'histogramme de couleur du segment de 13 s qui contient le tableau : fraction moyenne de pixels proches de 14 couleurs (8 chemises, objet, noir, peau, porte, jour). Les formes et positions sont jetées. Régression logistique entraînée sur le split TRAIN (400 groupes), testée sur EVAL (200 groupes).

| famille | Bayes masse de couleur, par question % (groupe %) | mean@120 contrastif grp | mean@120 brut, par question % |
|---|---|---|---|
| count | **99,9 (99,5)** | **1** | 26 |
| colour | 95,5 (84,5) | 79 | 71 |
| big_object | 26,5 (0,5) | 28 | 34 |
| sign_number | 25,6 (0) | 17 | 29 |
| hat_color | 16,9 (0) | 99 | 97 |

Contrôle causal : `mean_shuf` (positions des jetons permutées) donne presque le même résultat que `mean` (colour 67 contre 79, hat 99 contre 99, count 2 contre 1). Attention : `oracle_shuf` égale aussi l'oracle (count 97, sign 100). Les jetons du ViT portent déjà leur position (RoPE 2D interne), donc la permutation ne détruit pas la forme. **Ce contrôle montre que le LLM lit les jetons comme un sac, mais il ne prouve rien sur la masse de couleur.**

**Conclusion pour la question 3 :**
* Sur v3, **mean-pool n'a AUCUN gist de comptage** (1 % contrastif), alors que la statistique de masse de couleur le décode à 99,9 %. L'hôte n'exploite pas cette statistique dans des jetons moyennés. Le « gist partiel » vu sur v1 (init_count 44 à 56 %) ne se reproduit pas sur v3.
* Ce que mean-pool garde, c'est la **couleur** : `colour` 79 (Bayes couleur 95) et `hat_color` 99 (Bayes pixels 17 seulement, car le chapeau est trop petit pour mon seuil de pixels, mais l'hôte le lit). C'est de la statistique au sens propre : présence d'une couleur dans une fenêtre, sans lieu ni nombre. **Un gain d'une mémoire compressive sur `colour` ou `hat_color` ne prouve donc pas une mémoire épisodique.** Ces deux familles ne départagent pas « mémoire » et « statistique de couleur ».

## 4. Ce qu'une mémoire entraînée doit battre (12 Mio, D = 120 s, contrastif par groupe)

Meilleure mémoire non entraînée à octets égaux (≤ 12 Mio), à D = 120 s :

| famille | à battre | par | plafond (oracle) | utilisable pour une mémoire entraînée ? |
|---|---|---|---|---|
| count | **45 ± 4** | key | 99 | **oui** (mean = 1, aucune statistique de couleur exploitée) |
| big_object | **46 ± 4** | key (mean 28) | 100 | **oui** |
| sign_number | **50 ± 4** | key (mean 17) | 100 | **oui** (détail) |
| colour | **79 ± 3** | mean | 100 | réserve : battable par une statistique de couleur |
| hat_color | **99 ± 1** | mean | 100 | **non** : mean-pool le résout déjà |
| room_state (contrastif seul, secondaire) | 42 ± 7 | key (mean 2) | 100 | oui en contrastif |
| side (contrastif seul, secondaire) | 48 ± 7 | key (mean 22) | 92 | oui en contrastif, plafond 92 |

Bruts par groupe pour mémoire : count 18 (key), colour 41 (key) ou 25 (mean), big_object 33, sign 50, hat 89 (mean).
Règle proposée : une mémoire entraînée de ≤ 12 Mio « garde le gist » si, à D = 120 s, elle dépasse key de plus de 2 SE (environ 8 points en contrastif par groupe, n = 150) sur count ET big_object. Elle « garde le détail » si elle fait de même sur sign_number. hat_color est à exclure, colour à rapporter avec la borne mean. Le contrôle `other` (cible 0, donneur ≥ 88) est obligatoire.

**Familles secondaires** (50 groupes, contrastif seulement) : room_state et side se comportent comme les autres. Oracle 100 et 92 ; FIFO 100 → 0 ; key 42 et 48 ; mean 2 et 22 ; empty et other 0. Le donneur est choisi à 58 % pour room_state (biais d'option) et à 96 % pour side.

## Coût réel
Grille principale : 23 380 passes en 3,4 h (carte 1) et 3,9 h (carte 2), soit ≈ 1,0 s par passe sur des cartes partagées (0,55 s mesuré sur carte calme). Familles secondaires : 4 720 passes en 47 min. **Une cellule de 150 groupes (600 passes) coûte 5,5 à 10 min GPU** selon l'occupation de la carte. L'arrêt précoce a économisé ≈ 40 % des passes.

## Problèmes ouverts
1. **Charge** : le plafond échoue avec l'hôte 3B. Il faut choisir une alternative (n°1 recommandée) avant toute mesure de sélectivité.
2. **Gist/détail confondu avec couleur/structure** dans v3. Il faut une famille de détail NON colorée et assez fine (petit panneau, petite forme) et une famille de gist couleur-neutre. count et big_object sont les gist propres ; sign_number est le détail propre.
3. Score brut contre contrastif : count brut à 53 % par groupe sur 150 groupes, contre 76 % annoncé au bilan scenes (25 groupes). Le plafond brut de count est surestimé dans la certification.
4. `oracle_shuf` montre que la permutation de jetons n'est pas un contrôle de forme. Un vrai contrôle de « masse de couleur » demande une manipulation en pixels (par exemple brouiller les tuiles de 28 px avant l'encodeur).
5. Domaine dessiné : le transfert d'un greffon entraîné sur FineVideo reste à vérifier.
