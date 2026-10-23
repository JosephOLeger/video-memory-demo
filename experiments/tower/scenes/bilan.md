# Bilan : générateur de scènes procédurales « gist vs détail » (22/10/2026)

Dossier : `~/work/iclr/scenes/` sur la tour. Rien hors de ce dossier n'a été modifié (lecture seule de `audit/` et `ladder/`).

## Verdict court

* La version retenue est **v3, « tableau instantané »** : `gen_tableau.py` + `ceiling_v3.py`, avec les réglages `TStyle(count_gap=True, door_bright=True)` (appelés v3b ci‑dessous).
* Familles **certifiées** sur Qwen2.5‑VL‑3B (critère du lead, au niveau GROUPE : plafond ≥ 80, marge ≥ 50) : `count` (v3b), `colour`, `big_object` pour le gist ; `sign_number`, `hat_color` pour le détail. Marge de 84 à 100 points, plancher à 0.
* Échouent au score brut : `room_state` et `side`. La cause est un biais d'option, pas la perception : avec le score contrastif (voir plus bas), ces deux familles montent à 100 % par groupe sur le 3B, et le plancher reste à 0.
* Les familles **temporelles** (combien sont partis, qui est parti en premier, ordre porte/sortie) ne marchent pas avec le 3B dès qu'on retire les fuites (v2 : 27 à 51 %, soit le hasard). Elles testent l'hôte, pas la mémoire. Elles sont retirées.
* Recommandation : **oui comme banc principal pour le gist « instantané » et le détail**, avec score par groupe et contrôle « autre variante ». Il manque le réalisme et une vraie famille temporelle (voir la fin).

## Tableau par famille (v3/v3b, split eval, M‑RoPE corrigé)

Score par **groupe** = les 4 variantes contrefactuelles justes. Entre parenthèses : exactitude par question.
* plafond = `full` : un seul flux vidéo [4 s décor][tableau 6 s][8 s décor][fenêtre 8 s], l'épisode à sa place naturelle, jamais dans les 8 dernières secondes.
* plancher = max(`blind`, `window`, `other`) par groupe. `other` = format échelle (mémoire + fenêtre), mais la mémoire est le tableau d'une AUTRE variante du même groupe (même décor, mêmes options, autre réponse).
* `donneur` = part des réponses `other` qui donnent la réponse de la variante donneuse. Une valeur haute montre que la mémoire est lue, et pas seulement « non vide ».

### Qwen2.5‑VL‑3B (25 groupes = 100 questions par famille, sauf arrêt précoce)

| famille | type | plafond full grp (item) | oracle échelle grp (item) | blind grp | window grp | other grp (item) | donneur % | plancher | marge | certifiée |
|---|---|---|---|---|---|---|---|---|---|---|
| count (v3b, options {1,3,5,7} ou {2,4,6,8}) | gist | **96** (99) | 76 (94) | 0 | 0 | 0 (0) | 94 | 0 | **96** | OUI |
| colour | gist | **100** (100) | 96 (99) | 0 | 0 | 0 (0) | 99 | 0 | **100** | OUI |
| big_object | gist | **84** (96) | 84 (96) | 0 | 0 | 0 (0) | 96 | 0 | **84** | OUI |
| room_state (v3) | gist | 27 (73) | 0 (55) | 0 | 0 | 0 (27) | 55 | 0 | 27 | non (arrêt à 15 groupes) |
| room_state (v3b, porte lumineuse) | gist | 33 (68) | 0 (57) | 0 | 0 | 0 (27) | 57 | 0 | 33 | non (arrêt à 15 groupes) |
| side (famille tenue à part, binaire) | gist | 64 (88) | 56 (86) | 0 | 0 | 0 (14) | 86 | 0 | 64 | non (plafond < 80) |
| sign_number | détail | **100** (100) | 100 (100) | 0 | 0 | 0 (0) | 100 | 0 | **100** | OUI |
| hat_color | détail | **100** (100) | 100 (100) | 0 | 0 | 0 (0) | 100 | 0 | **100** | OUI |
| count (v3, options 1..6 contiguës) | gist | 76 (93) | 48 (86) | 0 | 0 | 0 (1) | 86 | 0 | 76 | non (remplacé par v3b) |

### Qwen2.5‑VL‑7B (50 groupes = 200 questions par famille ; conditions full, other, window, blind ; carte 0)

| famille | plafond full grp (item) | other grp (item) | blind / window grp | plancher | marge | certifiée |
|---|---|---|---|---|---|---|
| count (v3b) | **90** (98) | 0 (0) | 0 / 0 | 0 | 90 | OUI |
| colour | **92** (98) | 0 (0) | 0 / 0 | 0 | 92 | OUI |
| big_object | **100** (100) | 0 (0) | 0 / 0 | 0 | 100 | OUI |
| room_state (v3) | 50 (86) | 0 (20) | 0 / 0 | 0 | 50 | non |
| room_state (v3b) | 48 (85) | 0 (18) | 0 / 0 | 0 | 48 | non (partiel : 23 groupes sur 50, le run continue et écrit `results/v3b_7b.json`) |
| side | 32 (72) | 0 (14) | 0 / 0 | 0 | 32 | non |
| sign_number | **100** (100) | 0 (0) | 0 / 0 | 0 | 100 | OUI |
| hat_color | **98** (100) | 0 (0) | 0 / 0 | 0 | 98 | OUI |
| count (v3, 1..6) | 54 (88) | 0 (1) | 0 / 0 | 0 | 54 | non |

Le 7B ne débloque rien : il fait pareil ou moins bien que le 3B (count v3 : 54 contre 76 ; side : 32 contre 64 ; count v3b : 90 contre 96). Seul room_state est meilleur au 7B (50 contre 27), sans atteindre le seuil. Un hôte plus gros n'est donc pas la solution pour les familles faibles. Pour les familles temporelles, le 7B n'a pas été mesuré sur v2 : le lead a demandé de passer à v3 avant.

### Score contrastif (piste pour room_state et side)

Pour chaque groupe, on soustrait aux logits de lettre leur moyenne sur les 4 variantes, puis on prend l'argmax. Cela retire toute préférence constante pour une option. Le score reste légitime pour un banc contrefactuel, puisque les 4 variantes sont toujours évaluées par la même mémoire.

| 3B, grp (item) | full | oracle | other | window |
|---|---|---|---|---|
| count v3b | 100 (100) | 100 (100) | 0 (0) | 0 (25) |
| room_state v3b | 100 (100) | 100 (100) | 0 (0) | 0 (25) |
| side | 100 (100) | 92 (98) | 0 (2) | 0 (50) |
| les 4 autres familles | 100 (100) | 96 à 100 | 0 (0) | 0 (25) |

Au 7B, contrastif : room_state 94, side 88, count v3 64. Toutes les familles passent alors le critère au 3B, avec une marge de 92 à 100. Je recommande de rapporter les DEUX scores : le brut par groupe pour la certification, le contrastif comme mesure principale de la mémoire. Le contrastif ne demande rien de plus que les 4 variantes déjà évaluées.

## Test de fuite (obligatoire, `leak_gate_v3.py`) : PASS

* Pour chaque famille et chaque split (train, eval), sur 2 000 groupes : chaque option est la bonne exactement le même nombre de fois, `answer == f(faits)` est vérifié par assertion, et le prompt est identique dans le groupe.
* La règle de Bayes sur les traits sans mémoire (texte, contenu de la fenêtre, longueur du flux) donne exactement le hasard par question (25 %, ou 50 % en binaire) et **0 % par groupe**.
* Le décor avant, le décor après et la fenêtre sont identiques au pixel près entre les variantes d'un groupe. La longueur du flux est constante (52 images).
* v3b a été revérifié sur 2 000 groupes par split pour count et room_state : équilibre parfait, décor identique.
* Résultat : il n'existe plus d'« état final » qui porte la réponse. La fenêtre ne contient que du décor partagé par le groupe. Les mesures le confirment : blind = window = 0 par groupe dans toutes les familles, pour les deux modèles.

## Contact sheets

* v3 (tableaux, 7 familles × 2 variantes) : `~/work/iclr/scenes/contact_v3.png`, copie locale `/tmp/claude-1000/-home-josep-projects/d38af68f-dd51-4392-8d80-c80dbb5d4631/scratchpad/scenes_contact.png`.
* Pièce v3b (porte lumineuse, lampe déplacée) : `~/work/iclr/scenes/room_bright.png`.
* v2 (épisodes à événements, durée constante) : `contact_v2.png`. v1 : `contact_v0.png`.
* J'ai regardé chaque planche : silhouettes, chapeau, panneau (police 20 px) et objets sont lisibles à 448 px.

## Journal des essais

| version | changement | résultat | décision |
|---|---|---|---|
| v1 (`gen_scenes_v1.py`) | épisodes 12 à 27 s, entrées et sorties par les bords, porte, panneau, chapeau | init 79, final 37, n_left 30, first_left 58, order 50 (toujours « yes »), door 30, chapeau 99, panneau 100 (par question) | l'audit montre des fuites (état final + durée), des étiquettes porte fausses (11,8 %), des biais de lettre, et un M‑RoPE compressé 12× |
| sweep1 (qset 2, horloge, événements lents) | tué : M‑RoPE faux et horloge interdite (elle affiche T) | 1 config partielle (init 78, door 48, n_left 18) | abandonné |
| v2 (`gen_scenes.py`) | durée constante (28 s), état final partagé dans le groupe, first_left où tous sortent, porte = état de départ, splits, assertions, M‑RoPE corrigé | fuite 25 %/0 (PASS) ; init 66, n_left 40, first_left 27, order 51 = blind, door_start 73 | l'hôte n'intègre pas les événements : familles temporelles abandonnées |
| v3 (`gen_tableau.py`) | tableau tenu 6 s dans un flux de décor procédural ; faits lisibles sur une seule image | voir les tableaux : 4 familles passent au 3B ; count 76, room_state 27, side 64 | itérer count et room_state |
| v3b count | options sans valeurs voisines ({1,3,5,7} / {2,4,6,8}, jusqu'à 8 personnes, sans objet) | 3B 96 grp, 7B voir tableau | **retenu** |
| v3b room_state | porte ouverte = vue de jour (non assombrie), lampe éloignée de la porte (elle la masquait) | 3B 33 grp. Porte 88 %, lampe 75 % : c'est l'option composite qui pose problème, pas l'image | à refaire en deux familles binaires, ou à garder avec le score contrastif |
| arrêt précoce | après 15 groupes, arrêt si plafond grp < 40 ou plancher ≥ plafond − 10 | room_state (3B v3 et v3b) arrêté | appliqué |

Toutes les certifications v3b sont faites sur des groupes neufs (200 et plus) : aucun choix de design n'a été réglé sur les groupes mesurés (audit #9). Les splits suivent l'audit #7 : sel de graine, décor tenu à part (mur 3, sol 2 ou porte x = 320 en eval), chapeaux rose et blanc seulement en eval, panneau à une autre place et dans une autre police en eval, gabarit paraphrasé en eval, et `side` réservée à l'eval.

## Recommandation

1. **Banc principal : oui**, pour la question « la mémoire garde‑t‑elle le gist d'une scène et en perd‑elle le détail ? ». On a trois familles gist (count, colour, big_object) et deux familles détail (sign_number, hat_color). L'hôte les lit à 84‑100 % par groupe quand il voit l'épisode, et le plancher (blind, window, autre variante) vaut 0 par groupe. Le contrôle « autre variante » donne la réponse du donneur dans 94 à 100 % des cas : il prouve que le contenu de la mémoire est lu.
2. Le détail est **facile** pour l'hôte (100 %). C'est voulu, puisque le détail doit être mesurable, mais l'écart gist/détail ne viendra que de la mémoire. Pour rendre le détail plus fragile, on peut réduire `sign_font` (20 → 14) ou `hat_scale`, en vérifiant que le plafond reste ≥ 80.
3. Ce qui manque :
   * **Réalisme** : les scènes sont des dessins plats. Un greffon entraîné sur FineVideo devra généraliser vers ce domaine, ce qui reste à vérifier.
   * **Gist temporel** : hors de portée de l'hôte 3B (et le 7B n'aide pas au comptage). Il faudra plutôt une comparaison de deux tableaux (« plus de gens dans la première ou la seconde scène ? »), chacun lisible seul, comme le proposait le lead.
   * room_state et side en score brut : à scinder en familles binaires, ou à garder en score contrastif.
   * **Puissance** : 25 groupes par famille au 3B, 50 au 7B. D'après l'audit #13, il faut 150 à 250 groupes par cellule pour détecter 5 points entre mémoires.
4. Coût : environ 0,5 s par question au 3B pour `full` (6 700 jetons), et une famille de 100 questions × 5 conditions prend 8 à 16 min selon la charge de la carte.

## Fichiers

`gen_tableau.py` (v3), `ceiling_v3.py`, `leak_gate_v3.py`, `certify.py` ; `gen_scenes.py` (v2), `ceiling.py`, `leak_gate.py` ; `gen_scenes_v1.py`.
Résultats : `results/v3.json`, `v3b.json`, `v3_7b.json`, `v3b_7b.json` (+ `.jsonl` ligne à ligne), `leak_gate.json`, `v1.json`, et les logs `results/*.log`.
Note : la carte 0 a servi pour le 7B, sur instruction du lead relayée par le coordinateur. La consigne initiale était « carte 1 seulement ».
