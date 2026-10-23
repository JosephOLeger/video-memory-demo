# Pré-enregistrement : échelle de mémoires non entraînées

Écrit le 22/10/2026 AVANT toute mesure de l'échelle (aucune ligne de `runs/` n'existait).
Le code qui produit les mesures est `ladder.py` + `core.py` dans ce dossier, générateur de scènes figé dans `gen_scenes_snapshot.py`.

## Le dispositif

* Unité de temps : un groupe temporel Qwen = 2 images à 2 fps = 1 s. Image 448 px carrée = 256 jetons visuels par seconde. En 224 px = 64 jetons.
* Flux : P = 176 s de passé (FineVideo, recadré carré) puis une fenêtre W = 8 s de remplissage. L'épisode cible (12 à 24 s, 4 s pour `flat`) **remplace** le remplissage et se termine D secondes avant la fenêtre. D ∈ {0, 30, 90, 150}. L'épisode n'est jamais dans la fenêtre.
* Charge : L = 3 épisodes distracteurs (d'autres familles et groupes), placés AVANT la cible, à D = 30. La question demande alors le clip le plus récent.
* Paires contrefactuelles : les 4 variantes d'un groupe partagent le remplissage, la question et les options. Seul l'épisode change, et chaque option est juste une fois.
* Lecture : logits des lettres à la première position de réponse. Mémoire = première vidéo du contexte, fenêtre = seconde vidéo.
* `second_per_grid_ts` = 1,0 s (transformers 5.14 ignore `fps=` et met 1/12 s ; corrigé ici).

## L'échelle et ses budgets (jetons × 2048 × 2 octets)

| mémoire | contenu | jetons | Mio |
|---|---|---|---|
| oracle | l'épisode cible, 448 px | 256 × E (3 000 à 6 100) | 12 à 24 |
| fifo | les 80 dernières secondes évincées, 224 px | 5 120 | 20 |
| key | 1 s toutes les 8 s sur tout le passé, phase aléatoire par groupe, 448 px | 22 × 256 = 5 632 | 22 |
| mean | passé coupé en 22 segments de 8 s, jetons moyennés par position, 448 px | 5 632 | 22 |
| empty | fenêtre seule | 0 | 0 |
| blind | ni mémoire ni fenêtre | 0 | 0 |
| other | mémoire oracle d'un épisode d'un AUTRE groupe (variante tirée au hasard) | comme oracle | 12 à 24 |
| solo | épisode seul, prompt du générateur (= plafond du générateur, contrôle de pipeline) | 256 × E | 12 à 24 |

Budgets fifo, key et mean égalisés à ±10 %. L'oracle coûte au plus autant.

## Prédictions

Notations : `h` = hasard (25 %, 50 % pour `order`). `c_f` = plafond `solo` de la famille f. "≈" veut dire dans ±2 SE.

### P1. Mémoires vides (empty, blind)
* Exactitude par question ≈ `h` en moyenne, quelle que soit la famille.
* **Exactitude par paire = 0 exactement, et par groupe = 0** : l'entrée est identique pour les 4 variantes, donc la réponse aussi. Toute autre valeur est un bug.
* Indépendant de D et de la charge (pas de dépendance possible).

### P2. Autre vidéo (other)
* Exactitude par question ≈ `h`. Exactitude par paire ≈ `h²` (≈ 6 %), par groupe ≈ 0.
* Un score au dessus du hasard signale un raccourci (corrélation entre le rang de l'option et le contenu, ou un a priori partagé entre groupes).

### P3. Oracle
* oracle ≥ solo − 10 points, pour toutes les familles. Plat en D par construction.
* L'écart solo − oracle mesure le coût du format flux (intro différente, fenêtre de remplissage ajoutée). Prédit petit.

### P4. FIFO (80 s, 224 px)
* Fonction marche en D. Épisode entièrement dans le tampon pour D ∈ {0, 30} (D + E ≤ 54 < 80), absent pour D ∈ {90, 150}.
* D ∈ {0, 30} : gist ≈ oracle (les silhouettes de 60 px restent lisibles à 224 px). Détail en dessous de l'oracle : le panneau (chiffres de 13 px à 224 px) et le chapeau perdent de la résolution. Prédit : `sign_number` et `hat_color` chutent de 10 à 40 points par rapport à l'oracle ; `flat` reste au plafond.
* D ∈ {90, 150} : ≈ `h`, paires ≈ 0.
* Charge L = 3 à D = 30 : 1 ou 2 distracteurs entrent aussi dans le tampon. Baisse attendue modérée (0 à 15 points) si l'hôte sait prendre le clip le plus récent, forte sinon. C'est une propriété réelle de l'hôte, pas du banc.

### P5. Images clés (1 s toutes les 8 s)
* **Plate en D** (la phase est aléatoire ; la couverture ne dépend pas de D).
* Au-dessus du hasard d'environ `q_f × (c_f − h)`, où `q_f` est la probabilité qu'une seconde clé tombe sur la preuve :
  * `flat` (4 s d'aplat) : q ≈ 0,5.
  * `hat_color` (chapeau visible plusieurs secondes avant la sortie) : q ≈ 0,7 à 1. **Le détail durable survit.**
  * `sign_number` (panneau 2 s) : q ≈ 0,25. **Le détail bref meurt.**
  * `init_count` (4 s d'état initial immobile) : q ≈ 0,5. `final_count` (5 s d'état final) : q ≈ 0,6.
  * `n_left`, `first_left`, `door_state`, `order` (il faut deux instants bien placés, ou un instant entre deux événements) : q ≈ 0,2 à 0,4.
* Donc key < fifo(D ≤ 30) sauf peut-être sur `hat_color`, et key > fifo(D ≥ 90) partout où `c_f` est nettement au-dessus du hasard.

### P6. Moyenne temporelle (segments de 8 s, moyenne des jetons)
* Plate en D.
* Les jetons moyennés sont hors distribution pour l'hôte. Prédit : **ordre, première sortie, état de la porte, chapeau, panneau ≈ hasard**. Comptages (`init_count`, `final_count`) : légèrement au-dessus du hasard au mieux (≤ 0,3 × (c_f − h)), parce que les segments qui couvrent les maintiens immobiles du début et de la fin gardent la scène statique.
* `flat` : au-dessus du hasard (la couleur domine la moyenne d'un segment à moitié uni) ; prédit 40 à 80 %.
* mean ≤ key sur toutes les familles.

### P7. Ordre global attendu
* D ∈ {0, 30} : oracle ≥ fifo > key ≥ mean ≥ {empty, blind, other} ≈ hasard.
* D ∈ {90, 150} : oracle > key ≥ mean ≥ fifo ≈ {empty, blind, other} ≈ hasard.
* Gist contre détail : key garde le détail durable mais pas le bref ; mean garde un peu de gist et pas de détail ; fifo à 224 px coûte plus au détail qu'au gist.

### P8. Paires contrefactuelles
* Pour une mémoire d'exactitude p indépendante par question, exactitude par paire ≈ p² (au mieux p). Mémoires vides : 0 exactement.
* Une mémoire dont l'exactitude par question dépasse le hasard SANS que l'exactitude par paire monte signale un biais de réponse (a priori), pas une lecture du contenu.

## Critères de certification d'une famille (fixés maintenant)

Une famille est **certifiée** si toutes ces conditions tiennent :
1. `c_f` (solo) ≥ h + 30 points, et oracle ≥ h + 25 points (sinon il n'y a rien à mesurer).
2. empty, blind, other ≈ h (±2 SE) ; paire(empty) = 0.
3. fifo marche : fifo(D ≤ 30) − fifo(D ≥ 90) ≥ 20 points, et fifo(D ≥ 90) ≈ h.
4. L'ordre P7 est respecté au moins au sens large (aucune inversion significative à 2 SE).
5. Pour revendiquer "gist contre détail", le contraste prédit doit apparaître : key garde mieux `hat_color` que `sign_number` ; mean ≤ key.

Une famille qui satisfait 1 à 3 mais pas 4 ou 5 est "utilisable avec réserve" et la déviation est discutée dans `bilan.md`.

## Puissance (calcul a priori)

* SE d'une proportion : sqrt(p(1−p)/n). Pire cas p = 0,5 : n = 400 pour SE = 2,5 points, n = 625 pour 2 points, n = 278 pour 3 points.
* Le plan est apparié (mêmes items dans toutes les conditions) : la SE d'une différence entre deux mémoires est sqrt((p01 + p10)/n), plus petite que la SE non appariée quand les réponses sont corrélées.
* Cible : ≥ 400 questions par catégorie (gist, détail) et par cellule (mémoire × D), soit ≥ 100 groupes de 4 par catégorie. Par famille isolée, 25 à 50 groupes (100 à 200 questions, SE 3,5 à 5 points), suffisants pour les écarts attendus de 20 à 50 points de la marche FIFO.
* La SE par paire et par groupe est calculée au niveau du groupe (bootstrap sur les groupes), car les paires d'un même groupe sont corrélées.

---

# Addendum A (22/10, 11 h 40) : après l'audit `~/work/iclr/audit/audit.md`, AVANT le run principal

Seules mesures vues à ce moment : le smoke (1 groupe × 3 familles, n = 4 par cellule) et une passe `solo/empty/blind` en cours sur les groupes 0 à 24. Le run principal utilise des groupes **frais, 5000 et suivants**, jamais vus par le générateur, le sweep, l'audit ni par moi (audit n°9).

## Changements du protocole
1. **Familles** : `flat` (calibration), `init_count`, `first_left` (gist), `hat_color`, `sign_number` (détail). **Exclues** : `door_state` (11,8 % d'étiquettes fausses dans le snapshot), `order`, `n_left`, `final_count` (plafond par groupe = 0, pur biais de lettre). Le snapshot du générateur reste md5 `fbd5a18b` (sans `clock` ni `qset 2`, que l'audit déconseille).
2. **`other` devient contrefactuel** : la mémoire oracle de la variante suivante du MÊME groupe (même décor, même remplissage, mêmes options, autre réponse). L'ancienne version inter-groupes est abandonnée.
3. **Nouvelle mémoire `tail`** : les 5 dernières secondes de l'épisode (le maintien final immobile), 448 px, 1 280 jetons (5 Mio).
4. **Nouvelle mémoire `fifo448`** : les 20 dernières secondes évincées à 448 px, 5 120 jetons (20 Mio), même budget que `fifo` (80 s à 224 px).
5. **Phase des images clés tirée par (groupe, D)** et non plus par groupe.
6. **D ∈ {0, 30, 90}** (D = 150 retiré pour le coût ; D = 90 est déjà hors FIFO).
7. 120 vidéos de remplissage au lieu de 48 (toujours FineVideo, voir limite n°16 de l'audit).
8. **La certification se fait au niveau du GROUPE** (les 4 variantes justes) et de la paire. L'exactitude par question n'est plus qu'indicative.

## Prédictions ajoutées
* **tail** : `init_count` sous le hasard par question (Qwen non entraîné lit le compte final, différent du compte initial par construction ; prédit 5 à 25 %). `first_left` très sous le hasard (prédit < 10 % : il nomme une personne encore visible). `hat_color`, `sign_number` ≈ hasard (chapeau sorti, panneau jamais dans le maintien final). **Groupe ≈ 0 partout.**
* **other contrefactuel** : exactitude sur la réponse cible **sous le hasard**, environ (1 − o)/3 où o = exactitude oracle par question. Taux de réponse du donneur ≈ o. Groupe = 0 (prédit ≤ 2 %).
* **fifo448** : D = 0 : épisode entier dans le tampon si E ≤ 20 s (une partie des items), sinon le début manque. Prédit : ≈ oracle sur les items `in_buf`, `init_count` dégradé sur les autres, détail ≥ `fifo` 224 px au même D. D ≥ 30 : hasard.
* **fifo** (224 px), **key**, **mean**, **empty**, **blind** : prédictions P1 à P8 inchangées.

## Critères de certification révisés (par famille)
1. groupe(solo) ≥ 30 % et groupe(oracle) ≥ 25 %.
2. groupe(empty) = groupe(blind) = 0 exactement ; groupe(other) ≤ 2 % ; other sous le hasard par question.
3. Marche FIFO au niveau du groupe : groupe(fifo, D ∈ {0, 30}) − groupe(fifo, D = 90) ≥ 15 points, et groupe(fifo, D = 90) ≤ 2 %.
4. Pour les familles gist : groupe(tail) ≤ 5 %.
5. Aucune inversion significative (2 SE, bootstrap par groupe) de l'ordre P7.

## Puissance révisée
Cible ≥ 150 groupes par catégorie (gist, détail) et par cellule (mémoire × D) ; 75 à 100 groupes par famille. Le run tourne groupe par groupe en alternant les familles, donc tout arrêt anticipé laisse un échantillon équilibré ; le n atteint sera rapporté.
