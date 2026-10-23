# Bilan v4 : épisodes composités sur des vidéos réelles (22/10/2026)

Dossier : `~/work/iclr/scenes/`. Code : `gen_composite_v4.py`, `ceiling_v4.py`, `leak_gate_v4.py`, `anomaly_v4.py`, `certify.py`. Résultats : `results/v4*.json(l)`, `results/loadsweep*_v4_*`, `results/anomaly_v4.json`, `results/leak_gate_v4.log`.

## Verdict court

* **Test de fuite : PASS.** Les variantes d'un groupe ne diffèrent qu'à l'intérieur du calque. Les pools de vidéos train et eval sont disjoints entre eux, et disjoints des 120 vidéos de l'échelle.
* **Sans charge, 3 familles sur 5 passent le critère brut au 3B** (plafond grp ≥ 80, marge ≥ 50) :
  * hat_color (100), sign_number (96 en full, 100 en oracle) et colour (80 en full, à la limite) ;
  * count passe en oracle (92 à 96) mais pas en full (76) ;
  * big_object échoue (60 en full, 76 à 80 en oracle).
  * Le plancher vaut 0 partout. La variante donneuse est choisie dans 91 à 100 % des cas.
  * En **score contrastif**, les 5 familles sont à 92-100 par groupe, avec un plancher à 0.
* **Sous charge, le critère échoue**, sauf pour sign_number (L2 100, L5 96).
  * Avec l'indice de l'échelle (bordure turquoise, cible en premier), le 3B **ignore l'indice** : le score contrastif vaut 0 à L5, alors que les 4 variantes sont toutes présentes en mémoire et que seule la bordure les distingue.
  * Il répond à partir de la position. Cible en dernier (récence), on obtient L5 contrastif 72 à 100, mais L2 reste faible (4 à 60).
  * C'est la limite de l'hôte que l'échelle avait déjà vue en v1. Ce n'est pas un défaut du générateur.
* **Anomalie** : avec des stats de couleur, on détecte l'épisode à 66 % en v4 contre 100 % en v3. Avec les traits de vision de Qwen, on reste à 99 % (contre 100 %) : les dessins composités sont le seul indice.
* **Verdict** : v4 corrige le problème de style, mais pas encore celui de la charge. Le banc reste utilisable sans charge pour 3 familles en brut, ou pour les 5 en score contrastif. Sous charge, seul le détail « panneau » est mesurable avec le 3B.

## Construction

* **Épisode** : les silhouettes, l'objet, le panneau et le chapeau de v3 sont **composités** sur les images d'un extrait FineVideo réel. Le flux d'un item vient d'**un seul** extrait : [4 s][épisode 6 s][8 s][fenêtre 8 s] à 2 fps, recadrage carré 448 px. La vidéo continue de bouger sous le calque. L'épisode ne diffère du flux que par le **contenu** ajouté, pas par le style de l'image.
* **Lisibilité** : un halo (dilatation 7 px, flou) entoure chaque forme dessinée, une ombre douce est posée sous les pieds, et les silhouettes se tiennent dans la moitié basse (pieds à y = 410). La taille dépend du groupe (train 105 à 125 px, eval 115 à 135 px). Un contrôle visuel a été fait sur `contact_v4.png`.
* **Groupes contrefactuels** : même extrait, même départ, même taille, même position de panneau, même question, mêmes options. Seul le fait testé change. En dehors du calque, tout est identique au pixel près (vérifié).
* **Splits FineVideo** :
  * fonds train = `is_eval(id) == False` sans les 120 vidéos de remplissage de l'échelle (986 vidéos) ;
  * fonds eval = `is_eval(id) == True` sans ces 120 (113 vidéos) ;
  * le pool d'entraînement de graft_v3 (`is_eval False` moins les 120 de l'échelle) est donc disjoint des fonds eval ;
  * s'y ajoutent le sel de graine par split, les chapeaux rose et blanc réservés à l'eval, la police du panneau (sans en train, serif en eval), la plage de taille, et les gabarits paraphrasés.
* **Familles** : count, colour, big_object (gist) ; hat_color, sign_number (détail). **room_state est retiré** : porte et lampe n'ont pas d'équivalent naturel sur une vidéo quelconque.
* **Indice de charge** : c'est le mécanisme de l'échelle (`ladder3.py`, `CUE_WALL` turquoise (64, 200, 200), distracteurs = les autres variantes du même groupe, en cycle). L'épisode cible porte une bordure turquoise de 10 px et la question la nomme. Les distracteurs n'ont pas de bordure.
* **M‑RoPE** : métadonnées réelles, pas de rééchantillonnage, et assertion `second_per_grid_ts == 1.0` (Scorer de `ceiling_v3`).

## Test de fuite (`leak_gate_v4.py`) : PASS

* Pools train / eval / échelle-120 disjoints. Tous les fonds eval vérifient `is_eval`.
* Sur 2 000 groupes par famille et par split, chaque option est la bonne exactement 2 000 fois. `answer == f(faits)` est vérifié par assertion, et le prompt (simple et avec indice) est identique dans le groupe : le Bayes texte seul vaut donc le hasard par question et 0 par groupe.
* Sur 6 groupes par famille et par split, rendus et comparés pixel à pixel :
  * le début, la suite et la fenêtre sont identiques entre variantes ;
  * dans l'épisode, toutes les différences tombent **à l'intérieur** de l'union des masques de calque ;
  * la longueur est constante (52 images).

## Tableau par famille (Qwen2.5‑VL‑3B, split eval, score par GROUPE ; par question entre parenthèses)

Groupes 0 à 24 (count v4b : 200 à 224, et 300 à 324 en vérification). Plancher = max(blind, window, other) par groupe : **0 dans toutes les cellules**. full = un seul flux [début|épisode|suite|fenêtre]. oracle = format échelle (mémoire = épisode, puis fenêtre).

| famille | type | full grp (item) | oracle grp (item) | oracle, groupes 300-324 | other grp (item), donneur | marge full / oracle | contrastif full / oracle | certifiée (brut) |
|---|---|---|---|---|---|---|---|---|
| count (v4b, options {2,4,6,8}) | gist | 76 (93) | **92** (98) | 96 (98) | 0 (2), 98 % | 76 / 92 | 96 / 96 | oracle OUI, full non |
| count (v4, {1,3,5,7} ou {2,4,6,8}) | gist | 48 (86) | 68 (92) | | 0 (0), 92 % | 48 / 68 | 100 / 100 | non (un seul personnage lu comme « 3 ») |
| colour | gist | **80** (95) | 76 (94) | 68 (92) | 0 (1), 94 % | 80 / 76 | 100 / 100 | full OUI (à la limite), oracle non |
| big_object | gist | 60 (87) | 76 (91) | 80 (94) | 0 (2), 91 % | 60 / 76 | 92 / 92 | non |
| hat_color | détail | **100** (100) | **100** (100) | 100 (100) | 0 (0), 100 % | 100 / 100 | 100 / 100 | OUI |
| sign_number | détail | **96** (99) | **100** (100) | 100 (100) | 0 (0), 100 % | 96 / 100 | 100 / 100 | OUI |

blind et window valent 0 par groupe (25 % par question) dans toutes les familles, par construction : même texte, même fenêtre.

Essai v4c (halo sombre, objet 0,9 × la taille d'un personnage), sur des groupes de design 100 à 114 : colour full 80 / oracle 73, big_object 47 / 60. Aucun gain, donc non retenu.

## Plafond sous charge (oracle_all : épisode cible + L distracteurs dans la mémoire)

Mécanisme de l'échelle : la cible a une bordure turquoise (64, 200, 200) et la question la nomme ; les distracteurs sont les autres variantes du même groupe (en cycle). Score par groupe (par question) [contrastif].

| famille | bordure, cible EN PREMIER (comme l'échelle), groupes 0-24 : L2 / L5 | bordure, cible EN DERNIER (récence), groupes frais 300-324 : L2 / L5 |
|---|---|---|
| count | 0 (36) [0] / 0 (29) [0] ; v4b 0 (29) / 0 (26) | 4 (51) [4] / 48 (80) [76] |
| colour | 0 (32) [4] / 0 (16) [0] | 24 (75) [60] / 40 (84) [92] |
| big_object | 0 (24) [0] / 0 (21) [0] | 0 (50) [12] / 4 (55) [92] |
| hat_color | 0 (40) [4] / 0 (19) [0] | 16 (60) [16] / 32 (70) [72] |
| sign_number | 60 (84) [76] / 24 (51) [40] | **100 (100) [96] / 96 (99) [100]** |

Sweep de conception sur les groupes 100-109 (count, colour), score par question, L2 / L5 :
* bordure + premier : 30 / 25 et 42 / 25 ;
* bordure + dernier : 45 / 88 et 75 / 92 ;
* distracteurs d'autres groupes + premier : 45 / 48 et 72 / 62 ;
* autres groupes + dernier : 68 / 50 et 72 / 68 ;
* bandeau « TARGET » turquoise (24 px + étiquette) + dernier : 48 / 55 et 78 / 82. Le bandeau + premier a été perdu (OOM, carte partagée).

Lecture :
* **L'indice n'est pas utilisé.** Au L5 avec cible en premier, les 4 variantes sont toutes dans la mémoire et seule la bordure distingue la cible. Le score contrastif y vaut 0 : la réponse ne bouge pas quand la bordure change de clip.
* **Ce qui compte, c'est la position.** En dernier, la réponse suit le clip le plus récent. Cela marche à L5 (contrastif 72-100) mais pas à L2, où les distracteurs se répartissent autrement (variantes v+1 et v+2 avant la cible).
* Seul le panneau survit à la charge dans les deux ordres. C'est un détail unique et lisible, que l'hôte peut chercher dans tout le contexte. L'échelle trouvait la même chose en v1 : le gist s'effondre sous charge, le panneau non.
* **Le plafond sous charge n'est donc pas certifiable avec le 3B**, sauf pour sign_number. Il faudra une désignation que l'hôte suit vraiment. Pistes, dans l'ordre :
  1. un vrai texte entre les clips : la mémoire en plusieurs vidéos séparées par « Clip k: », la question nommant k ;
  2. une désignation par récence assumée (« la scène la plus récente ») avec des distracteurs toujours AVANT la cible ;
  3. des distracteurs d'une autre famille (pas de confusion de réponse possible).

## Classifieur d'anomalie (épisode contre reste du flux, unités de 1 s)

Les unités de 1 s de l'épisode sont opposées aux unités du même flux prises juste avant, après, et dans la fenêtre. Régression logistique L2, ajustée sur des groupes train (60 groupes × 5 familles × 2 variantes), testée sur des groupes eval (30 × 5 × 2), soit 7 200 / 3 600 unités.

| générateur | stats couleur (56 dims) : train / **eval** | Qwen vision moyennée (2 048 dims) : train / **eval** |
|---|---|---|
| v3 (pièce dessinée dans un décor procédural) | 100 / **100** | 100 / **100** |
| v4 (calque sur vidéo réelle) | 63,7 / **65,7** | 100 / **99,3** |

Lecture :
* Avec des statistiques de couleur, l'épisode v4 n'est presque plus séparable : 66 % contre 100 % en v3, pour un hasard à 50 %. Le style de l'image ne trahit plus l'épisode.
* Avec les traits de vision de Qwen, il reste séparable à 99 %. Le seul indice est le calque lui-même : les dessins sont dans l'épisode et absents ailleurs. C'est le cas jugé « acceptable » par le lead.
* Pour qu'une mémoire entraînée ne puisse pas apprendre « garder les images avec des dessins », il faut que le reste du flux contienne **aussi** des calques du même type, sans pertinence pour la question. Ce sont exactement les distracteurs de la condition de charge. Je recommande que le flux d'entraînement en contienne systématiquement (L ≥ 2).

## Journal des essais v4

| étape | changement | résultat | décision |
|---|---|---|---|
| v4 | calque sur FineVideo, halo clair, ombre, count {1,3,5,7}/{2,4,6,8}, bordure turquoise, cible en premier | voir le tableau ; count 48 car n = 1 est lu « 3 » sur 11 cas sur 12 | corriger count |
| v4b | count toujours {2,4,6,8} (groupes frais 200-224) | full 76 (93), oracle 92 (98) | retenu |
| sweep de charge | ordre premier/dernier × distracteurs variantes/autres groupes × bandeau (groupes de design 100-109) | seul « dernier » aide, et seulement à L5 | certifié sur des groupes frais 300-324 : échoue sauf pour le panneau |
| v4c | halo sombre (le blanc se perdait dans le halo clair), objet plus grand | colour 80/73, big_object 47/60 (design) | rejeté |
| anomalie | régression logistique couleur et Qwen, train → eval | v3 100/100, v4 66/99 | voir plus haut |
| incident | coupure réseau de la tour (~16 h). Les runs avaient fini avant. Deux OOM par course entre lancements (carte partagée) | un verrou `flock` sérialise maintenant mes lancements | relancé |

## Recommandation

1. **Pour le gist sans charge, garder v4** (calque sur vidéo réelle) **avec le score contrastif comme mesure principale** : 5 familles à 92-100 par groupe, plancher à 0, marge ≥ 92. Au score brut, seules hat_color, sign_number et colour (à la limite) passent ; count passe en oracle. Le biais d'option du 3B coûte 10 à 30 points de groupe en brut, et le contrastif le retire sans toucher au plancher.
2. **Charge** : ne pas utiliser l'indice turquoise tel quel avec le 3B, ni dans l'échelle ni ici. Il faut tester d'abord la désignation par texte entre clips (piste 1). Jusque‑là, seul sign_number est mesurable sous charge.
3. **Anomalie** : le style n'est plus un indice (66 % contre 100 % en stats de couleur). Le calque l'est par nature (99 % en traits Qwen). Pour l'entraînement, il faut mettre des épisodes composités non pertinents (distracteurs du même type) partout dans le flux, sinon la mémoire peut apprendre « garder les images avec des dessins ».
4. big_object reste en dessous du seuil brut sur vidéo réelle (60-80). Les formes simples se confondent avec le décor. On peut le garder en contrastif (92) ou le retirer.
5. Note de puissance : 25 groupes par cellule. L'audit demande 150 à 250 groupes par cellule pour comparer des mémoires.

## Fichiers

* Contact sheet : `~/work/iclr/scenes/contact_v4.png`, copie locale `/tmp/claude-1000/-home-josep-projects/d38af68f-dd51-4392-8d80-c80dbb5d4631/scratchpad/scenes_contact_v4.png`.
* Résultats : `results/v4.json`, `v4b_count.json`, `v4c_design.json`, `v4_loadlast_*.json`, `loadsweep*_v4_*.json`, `anomaly_v4.json`, `leak_gate_v4.log`.
* Outils : `summarize_v4.py` (score par groupe), `deb_v4.py` (contrastif), `launch_any.sh` (attente de VRAM ≥ 12 Go et verrou).
