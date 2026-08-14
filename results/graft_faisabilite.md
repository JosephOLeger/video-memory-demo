# Greffe dans un VLM — étude de faisabilité chiffrée

Mesuré le 2026-08-14 sur la box (4× A5000 22 Go), `Qwen/Qwen2.5-VL-3B-Instruct`,
`transformers` 5.14.1, torch 2.6.0. Scripts : `~/work/graft/{measure,graft_mem,timing2,autocorr2}.py`.

Le design est **additif** : une branche latérale sur le flux résiduel à une couche
haute, porte de sortie initialisée à zéro. On ne remplace aucun bloc d'attention, on ne
réduit pas le budget de tokens du modèle, on ne réentraîne pas le tronc.

---

## 1. Le modèle et ce qui tient en VRAM

**Correction au plan initial : Qwen2.5-VL-3B a 36 couches, pas 28.** Les deux tiers de
la profondeur sont donc la **couche 24**.

| | |
|---|---|
| poids bf16 chargés sur GPU | **7,63 Go** |
| `d_model` / couches | 2048 / 36 |
| tokens visuels par frame à 224×224 | **32** |
| résolution native des vidéos EgoLife | 1408×1408 (à forcer, sinon la tour visuelle OOM) |

Activations mesurées, greffe couche 24, gradient checkpointing, perte restreinte aux
tokens de réponse (`logits_to_keep`) :

| frames | tokens | activations |
|---|---|---|
| 64 | 2 071 | 1,49 Go |
| 128 | 4 119 | 2,94 Go |
| 256 | 8 215 | 5,83 Go |

Soit **0,71 Go par millier de tokens**, parfaitement linéaire. Sur une carte dédiée
(23 − 7,63 = 15,4 Go disponibles) → **~21 700 tokens ≈ 660 frames par fenêtre**, soit
**11 minutes à 1 fps**. *(Extrapolation : les mesures ont été prises sur une carte
partagée avec la phase 3, plafonnée à 10,1 Go libres.)*

### Le site d'insertion décide du coût, et c'est le plus bas qui compte

À 2 071 tokens, en faisant varier la couche de greffe :

| site | 4 | 12 | 24 | 32 |
|---|---|---|---|---|
| activations | 8,59 Go | 7,09 Go | 4,80 Go | 3,28 Go |

**0,19 Go par couche au-dessus du site.** Les couches sous le site tournent sans
graphe d'autograd et ne conservent rien. *(Ces quatre valeurs ont été prises avant la
correction `logits_to_keep` et sont donc gonflées d'environ 3,3 Go chacune ; l'écart
entre sites, lui, est valide — le coût des logits est constant.)*

**Deux pièges coûteux trouvés en route.** Sans `logits_to_keep`, matérialiser les
logits sur toutes les positions (vocabulaire 152 k) coûtait **plus que tout le reste** :
4,80 Go contre 1,49 Go à 64 frames. Et tant que ce n'était pas corrigé, le gradient
checkpointing ne gagnait que 23 %, ce qui donnait une fausse impression sur son utilité.

---

## 2. Débit, et où est réellement le goulot

Forward + backward, par carte :

| composant | frames/s |
|---|---|
| pile Qwen complète (tour visuelle + 36 couches) | 18 |
| **12 couches sur activations en cache** | **163** |
| mémoire SANE scalaire, `dim=128`, chunk 8 | 405 |
| + Muon (`sane_ns`) | 139 |
| + cascade K=6, leak surprise 0.1 | 114 |

Le tronc étant gelé et la greffe haute, la tour visuelle et les couches 0-23 sont
**déterministes** : on les calcule une fois et on met les activations couche-24 en cache.
Le pas d'entraînement se réduit alors à 12 couches sur des tenseurs préparés.

| | naïf | avec cache |
|---|---|---|
| débit combiné (`sane_ns`) | 16 f/s | **75 f/s** |
| débit combiné (cascade) | — | 67 f/s |

Facteur **4,7** — et non 9, parce qu'une fois le cache en place **c'est notre bloc
mémoire le goulot, pas le VLM**. Muon coûte 3× à lui seul (405 → 139 f/s) ; la cascade
ne rajoute que 18 % par-dessus.

> Réserve : les 163 f/s sont mesurés avec un `nn.Linear` en guise de branche. Le
> chiffre combiné suppose que les deux coûts s'additionnent en série, ce qui est le
> majorant.

### Budget sur le corpus d'entraînement

A2-A6 ≈ 230 h de vidéo (A2 jour 1 = 797 clips de 30 s = 6,6 h/jour), soit **830 k
frames à 1 fps**, sur 4 cartes :

| | |
|---|---|
| passe de cache, une seule fois | ~1 à 1,5 h *(estimé, non mesuré directement)* |
| taille du cache | 109 Go non poolé / **3,4 Go poolé** (1,2 To libres) |
| **une époque d'entraînement** | **~46 min** (`sane_ns`) / ~52 min (cascade) |

Dix époques tiennent en 8 heures. **Le calcul n'est pas le facteur limitant.**

---

## 3. Le banc : EgoLifeQA, et ce qu'il impose

`lmms-lab/EgoLifeQA` n'existe pas comme dépôt — le QA est dans `lmms-lab/EgoLife`, et
il ne couvre **qu'un participant sur six** (`A1_JAKE`) : **500 questions** en QCM,
réparties en EventRecall 126, EntityLog 125, RelationMap 125, TaskMaster 63,
HabitInsight 61.

**Chaque question porte `query_time` et `target_time`** — la distance entre l'indice et
la question est donnée. La courbe score-vs-distance, seule figure qu'un fine-tuning ne
sait pas produire, est donc fournie par le banc.

Sur les 463 questions à distance calculable :

| distance indice → question | n |
|---|---|
| < 5 min | 2 |
| 5-30 min | 73 |
| 30 min - 2 h | 78 |
| **2-24 h** | **201** |
| **> 1 jour** | **109** |

Médiane **5,9 h**, maximum **138,5 h**.

À 32 tokens/frame et 1 fps, 6 heures font **678 000 tokens** : le banc est
structurellement hors de portée de toute fenêtre d'attention. C'est exactement le
régime que la thèse vise.

**Split propre disponible** : les 808 sous-titres `DenseCaption` sont organisés par
participant et par jour. Entraîner la branche sur les captions de **A2-A6**, évaluer sur
le QA de **A1** — in-domain, aucune fuite, aucun QA vu à l'entraînement.

---

## 4. Initialisation des portes, dérivée du banc

### `α` est la porte qui compte ici, et la nôtre est très mal réglée

`τ_ret = 1/α` pas de mémoire ; à 1 fps, un pas vaut une seconde.

| | τ_ret | α |
|---|---|---|
| notre `alpha_init = 0.002` | 500 pas = **8,3 min** | 2·10⁻³ |
| question médiane (5,9 h) | 21 200 pas | **4,7·10⁻⁵** |
| question la plus longue (138 h) | 497 000 pas | **2,0·10⁻⁶** |

**`alpha_init` est 40 à 1000 fois trop agressif pour ce banc.** Il avait été réglé sur
des séquences de 24 à 64 tokens ; l'échelle a changé de quatre ordres de grandeur.

À noter : le bras 40 000 pas avait *appris* α = 2·10⁻⁶, soit τ_ret = 5,8 jours à 1 fps —
exactement le bon ordre pour un banc dont les questions vont jusqu'à 138 h. Ce qui avait
été signalé comme un accumulateur pur pathologique est ce que la tâche exige.

**Valeur retenue : `alpha_init ≈ 1e-5`** (τ_ret ≈ 28 h à 1 fps), dérivée du banc et non
balayée.

### `η` : mesuré sur le vrai flux, et non concluant

Autocorrélation du flux couche-24 **poolé par frame** — celui que la mémoire lira — sur
19,6 min de vraie vidéo EgoLife (40 clips A2 jour 1, 1 152 frames à 1 fps) :

| | latents V-JEPA | **couche 24 de Qwen** |
|---|---|---|
| composante commune à tout le clip | 36 % | **91 %** |
| cos du résidu à 8 frames | −0,015 | **0,42** |
| cos du résidu à 512 frames | — | **0,29** |

Deux conclusions :

1. **Le centrage du flux est obligatoire, pas optionnel.** 91 % de ce que la mémoire
   écrirait est la même direction à chaque frame — de l'interférence pure, et bien pire
   qu'en espace V-JEPA.
2. **Les adresses reviennent.** Là où le résidu V-JEPA tombait à zéro en 8 frames, celui
   de la couche 24 tient à ~0,3 jusqu'à 8,5 minutes. C'est la première évidence directe
   que greffer haut résout le problème de ré-occurrence identifié la veille.

En revanche **`eta_init` ne se déduit pas de cette mesure** : pas de croisement à 1/e
dans la fenêtre observée. Le creux au décalage 1 (0,18 contre 0,58 au décalage 2) est un
artefact de la fusion temporelle par paires de Qwen, pas un signal.

Faute de mieux, `τ_acc` à l'échelle de l'événement — ~8 frames, là où le résidu est
encore au-dessus de 0,4 — donc **`eta_init ≈ 0.875`, et gelé** pour les premiers bras.
La porte apprise est bimodale et pilotée par la seed, donc un balayage mesurerait la
loterie et pas l'architecture.

---

## 5. La cascade : premier cadre où sa prémisse est vraie

Le banc demande de tenir des échelles allant de la seconde (un geste) à 138 heures :
**cinq ordres de grandeur**. Un couple scalaire (α, η) ne peut pas les couvrir. C'est
la prémisse de la cascade, et c'est la première fois qu'elle est satisfaite — sur des
séquences de 64 frames dont le contenu se décorrèle en 8, il n'y avait rien à
représenter.

### La portée réelle est dans le spectre de B, pas dans `g^(K-1)`

`g^(K-1)` est le rapport des couplages, pas celui des constantes de temps. En
diagonalisant le `B` tridiagonal (`τ_k = 1/(1−λ_k)`, pas de mémoire) :

| config | τ des réservoirs (pas) | portée finie |
|---|---|---|
| défaut K=6, g=2 | ∞, 93, 24, 10, 4,4, 1,7 | **56×** |
| **option A** K=6, g=9 | ∞, 107 000, 10 400, 1 076, 105, 8,7 | 12 200× |
| **option B** K=17, g=2 | ∞, 303 000, 68 900, … , 4,4, 1,7 (17 valeurs) | **180 000×** |
| option C K=10, g=3 | ∞, 53 700, 11 600, 3 740, … , 2,7 | 19 800× |

Le défaut couvre donc **56×** et non 32×, et chaque configuration porte en plus un
mode `λ = 1` exactement (masse conservée : le réservoir le plus profond n'a pas de
fuite propre). C'est le gate `α` de la mémoire qui borne ce mode — à `alpha_init = 1e-5`,
à 100 000 pas, soit 28 h. Les deux mécanismes s'emboîtent proprement.

**Le critère n'est pas la portée mais la densité.** Option A atteint bien 10⁵ pas, mais
avec des trous d'un facteur 10 entre réservoirs consécutifs : rien entre 105 et 1 076
pas, c'est-à-dire rien entre 2 min et 18 min. Option B espace ses τ d'un facteur 2 sur
toute l'étendue, de 1,7 pas à 303 000 pas — la couverture est uniforme et englobe le
banc entier de bout en bout.

**Retenu : option B, `K=17, g=2`, `beta1=0.5`** (contrainte `beta1(1+1/g) = 0,75 ≤ 1`
respectée, tous les λ positifs, aucun risque de bord à −1). Coût à mesurer : K=6 ajoutait
1,6 ms/frame sur les 7,2 ms de `sane_ns` ; l'état étant `[B, d_out, d_in, K]`, K=17
devrait coûter ~4,5 ms → ~56 f/s combiné au lieu de 67, soit une époque à ~62 min.

C'est surtout un **critère de conception** pour K et g, dérivé du banc comme `α`, au
lieu du balayage qui n'a jamais rien donné.

### La contrainte qui mord vraiment : les réservoirs lents ne sont jamais excités

Dans une fenêtre de 660 frames, seuls les réservoirs de τ < 660 atteignent leur régime
établi : **8 des 17** en option B, 2 des 6 en option A. Les neuf autres reçoivent bien le
gradient mais se comportent en accumulateurs purs sur la fenêtre — ils intègrent, ils ne
discriminent pas les distances.

Ce n'est pas rédhibitoire, et c'est même la formulation précise du risque du §6 : les
réservoirs lents ne peuvent acquérir leur statistique longue que si **l'état de la
cascade est enfilé d'une fenêtre à la suivante** sur une journée de vidéo entière, le
gradient restant tronqué à la fenêtre. C'est exactement ce que `S` fait déjà dans
`_scan_chunked` (l'état n'est jamais remis à zéro), donc du TBPTT standard et défendable
— mais il faut le vérifier explicitement au niveau du `DataLoader` : ordre temporel
strict, pas de shuffle des fenêtres à l'intérieur d'une journée.

---

## 6. Le risque qui reste entier

**Fenêtre d'entraînement 11 min, distance médiane demandée 5,9 h : facteur 32.**

L'hypothèse que des projections apprises à courte distance fonctionnent à longue
distance n'est pas une commodité d'implémentation, c'est toute l'expérience. Si elle
tombe, il n'y a pas de résultat. Rien de ce qui est mesuré ici ne la réduit ; tout le
reste est confortable.

Trois sorties possibles, à chiffrer avant de coder : baisser le fps (à 0,2 fps la
fenêtre couvre ~55 min), pooler plus agressivement par frame, ou assumer l'entraînement
court et faire de la généralisation le résultat principal. La troisième est la seule qui
produise une figure inexplicable par le fine-tuning, et c'est aussi la plus risquée.
