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

Soit **0,71 Go par millier de tokens**, parfaitement linéaire, c'est-à-dire **23,3 Mo par
frame** (32 tokens visuels). Sur une carte dédiée (23 − 7,63 = 15,4 Go disponibles) →
**676 frames, soit 11,3 minutes à 1 fps**. *(Extrapolation : les mesures ont été prises
sur une carte partagée avec la phase 3.)*

**Mais c'est le VLM seul, et le bloc mémoire coûte plus cher que lui** (§2bis) : la
fenêtre réelle tombe à **8,9 min** avec `sane_ns` et **6,9 min** avec la cascade K=17.

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

### 2bis. Le bloc mémoire coûte plus de VRAM que le VLM

Mesuré séparément, `WorldModel(dim=128, depth=2, seg=8)`, batch 1, 288 → 1152 tokens
(un token = une frame poolée), fwd+bwd, sur A5000 :

| bloc | Mo/frame | + VLM (23,3) | fenêtre max | pic à 1152 tok |
|---|---|---|---|---|
| `sane_ns` seul | 6,34 | 29,6 | **532 frames / 8,9 min** | 7,15 Go |
| cascade K=6 | 9,10 | 32,4 | 486 frames / 8,1 min | 10,26 Go |
| cascade K=10 | 11,11 | 34,4 | 458 frames / 7,6 min | 12,51 Go |
| cascade K=17 | 14,60 | 37,9 | **416 frames / 6,9 min** | 16,45 Go |

Le graphe d'autograd du scan par token, avec Newton-Schulz à chaque pas, est ce qui
domine — pas le VLM. Un premier essai à 4 608 tokens a fait **OOM à 17 Go**. La VRAM
croît linéairement en K ; le **temps**, lui, ne bouge pas (22,0 s à K=6 comme à K=17 :
le scan est vectorisé sur l'axe des réservoirs). La cascade coûte +30 % en temps par
rapport à `sane_ns`, indépendamment de K.

Stabilité vérifiée sur 4 608 tokens : `finite=True` et dérive de norme ×0,99–1,01 pour
les quatre configurations. Aucune ne diverge.

*(Ces chiffres sont pris à `depth=2`, soit deux blocs MAC ; la greffe n'en aura qu'un,
donc compter environ la moitié.)*

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

### Mais la cascade ne donne pas K échelles de temps — elle en donne ~1,7

La portée n'est pas la bonne mesure. Ce que le readout peut réellement construire, c'est
l'espace engendré par les noyaux de réponse `R[k,n] = (Bⁿ)[k,0]` : le gradient est injecté
dans le réservoir 0, et `S_t[k] = Σ_i R[k, t−i]·g_i`, donc un readout linéaire de poids
`w` réalise le noyau effectif `Σ_k w_k R[k,·]`. **Le nombre de filtres temporels
distinguables est le rang effectif de ce banc**, pas le nombre de réservoirs.

Rang effectif = participation ratio du spectre singulier des `R[k,·]` normalisés :

| config | τ lent fini | PR sur 660 pas | PR sur 100 000 pas |
|---|---|---|---|
| défaut K=6, g=2 | 93 | **1,12** | 1,00 |
| option A K=6, g=9 | 106 752 | **1,40** | 1,13 |
| option C K=10, g=3 | 53 704 | **1,54** | 1,11 |
| option B K=17, g=2 | 303 030 | **1,70** | 1,26 |

**Dix-sept réservoirs valent 1,7 filtre indépendant, et un seul à longue distance.** La
cause est le mode `λ = 1` : à grand décalage, la réponse de *tous* les réservoirs converge
vers le vecteur propre uniforme, donc les K noyaux deviennent colinéaires. Le réservoir 0
domine d'ailleurs la réponse à **toutes** les distances — aucun réservoir profond n'a de
régime où il porte le pic.

Et le leak ne corrige rien : `full`, `surprise`, `slow` et `none` multiplient l'état
**entier** par un scalaire, ce qui envoie `B → cB` — enveloppe commune, formes relatives
inchangées. Vérifié : le rang ne bouge qu'en raccourcissant l'horizon (c=0,9 → PR 2,23,
mais τ divisé par 10). Seul `shallow`, qui ne draine que le réservoir 0, est non uniforme
(η=0,5 → PR 3,08) — au prix du même raccourcissement.

| levier | PR à K=6 | effet sur l'horizon |
|---|---|---|
| K : 6 → 17 | 1,12 → 1,70 | ×3 260 |
| leak uniforme c : 1 → 0,9 | 1,12 → 2,23 | ÷10 |
| leak `shallow` η : 1 → 0,5 | 1,12 → 3,08 | ÷10 |
| `topology=independent`, K=6 | 1,59 | — |

Même le banc de filtres « indépendants » — K exponentielles à taux distincts, l'hypothèse
nulle du docstring — plafonne à PR 1,59 (K=6) et 2,03 (K=17). Ce n'est donc pas un défaut
de la cascade : **tout banc d'exponentielles est mal conditionné** (l'inversion de Laplace
est un problème mal posé). Aucune architecture de ce type ne donnera K échelles séparables.

Deux corollaires immédiats :

- **`readout=gated` ne sert à rien.** Le noyau `mean` capte **99,93 %** de la première
  direction du span, et 100 % des deux premières. Une porte apprise par token n'a rien à
  arbitrer sur un span de dimension 1,1. Autant prendre `mean`, moins cher et plus stable.
- **Réserver la fuite aux réservoirs profonds** (« conserver la masse dans le premier »)
  fait bien monter le rang — 1,12 → 1,81 — mais en payant le prix habituel : τ_max chute
  de 1 200 à 15,7 pas. Même arbitrage que tous les autres leaks.

> **Le §5bis supersède cette section.** Le rang effectif n'est pas le bon diagnostic :
> Benna-Fusi ne cherche pas K filtres séparables mais une décroissance en loi de
> puissance du seul `u₁`. Un PR ≈ 1 est attendu. Ce qui reste valide ici, c'est le
> constat que le mode `λ=1` égalise les réservoirs — et le §5bis en donne la cause
> (les `C_k` manquants) et le correctif.

**Réserve :** tout ceci vaut pour un `B` **fixe**. La revendication propre de
la cascade (`selective=True`, `B_t` par arête, non commutatif) est précisément ce qu'un
banc de filtres fixe ne peut pas reproduire, et le PR ne la mesure pas. Ce calcul borne
le **plancher** de la cascade, il ne la réfute pas.

### 5bis. Confrontation à Benna & Fusi 2016 — trois écarts structurels

*(Benna & Fusi, « Computational principles of synaptic memory consolidation »,
Nat. Neurosci. 19(12), 2016 — le papier dont la cascade est tirée.)*

**Le bon diagnostic n'est pas le rang, c'est l'exposant de décroissance.** Benna-Fusi ne
revendique jamais K échelles séparables. `u₁` **est** le poids synaptique et c'est le seul
readout ; les autres variables n'existent que pour *façonner la décroissance de `u₁` en
loi de puissance* `t^(−1/2)` — le décay le plus lent qu'on puisse tenir sans faire diverger
la variance du poids. Un PR ≈ 1 est donc le design, pas un bug ; §5 mesurait la mauvaise
quantité. La bonne mesure est la pente de `u₁(t)` en log-log, cible **−0,5**.

Contrôle du code : chaîne homogène (Fig 2a du papier, C_k=1, g=1/8, m=31) → pente
**−0,493**. La diffusion 1D pure est bien reproduite.

| | pente sur 10³–10⁵ |
|---|---|
| **notre chaîne** (C_k = 1, pas de fuite terminale) | **−0,066** |
| + fuite terminale seule | −0,264 |
| + beakers `C_k = 2^k` → **Benna-Fusi Fig 2c** | **−0,500** |

**Écart n°1, et c'est le principal : il manque les `C_k`.** L'équation du papier est
`C_k du_k/dt = g_{k−1,k}(u_{k−1}−u_k) + g_{k,k+1}(u_{k+1}−u_k)` — les `C_k` sont les
*sections des béchers*, qui croissent géométriquement (`C_k = 2^k`). Notre `cascade_step`
applique le flux `b_e(u_{e+1}−u_e)` **symétriquement** aux deux réservoirs : c'est
`C_k ≡ 1`, des béchers tous de la même taille. Or c'est exactement l'asymétrie qui fait
le mécanisme : un bécher profond est énorme, son niveau bouge à peine quand il reçoit, et
il retient. Nous avons les tubes (`geometric_betas`) sans les béchers — une combinaison
que le papier n'utilise nulle part (Fig 2a = béchers égaux + tubes égaux ; Fig 2c = les
deux géométriques).

La loi de conception est simple, et elle reproduit le `O(n²)` de l'éq. (6) du papier :

| `C_k = c^k` \ `g_e = β₁·gg^(−e)` | gg=1 | gg=2 | gg=4 |
|---|---|---|---|
| **c=1** *(notre modèle)* | −17,2 | **−0,26** | −0,10 |
| **c=2** | −2,60 | **−0,500** | −0,343 |
| **c=4** | −0,998 | −0,676 | **−0,523** |

**`c = gg` donne −0,5.** Un seul paramètre, et c'est le `g` qu'on expose déjà.

**Écart n°2 : pas de fuite terminale.** Benna-Fusi pose `u_{m+1} = 0`, ce qui donne au
bécher le plus profond une fuite propre. Nous n'en avons aucune — d'où le mode `λ = 1`
diagnostiqué au §5. C'est le même bug vu sous un autre angle.

**Écart n°3 : notre fuite est au mauvais endroit et beaucoup trop forte.** Le papier ne
fait fuir que `u_m` ; nos modes `full`, `surprise`, `slow` multiplient l'**état entier**
par `exp(−r·θ_t)`, ce qui impose une enveloppe exponentielle à tout — précisément ce
qu'une mémoire en loi de puissance ne doit pas avoir.

Sur la chaîne corrigée, jusqu'où `r` peut monter :

| r | pente 10³–10⁵ | |
|---|---|---|
| 0 | −0,500 | loi de puissance intacte |
| 10⁻⁶ | −0,529 | intacte |
| 10⁻⁵ | −0,792 | dégradée |
| 10⁻⁴ | −3,43 | détruite |
| **0,1 (notre défaut)** | **NaN** (−10,5 déjà sur 10–10³) | **détruite** |

La fuite étant `exp(−r·θ_t)`, le taux effectif est `r·θ_t` ; avec les `θ_max` mesurés sur
les artefacts (0,004 à 0,043) cela fait 4·10⁻⁴ à 4·10⁻³, soit **encore 400 à 4 000 fois
trop** devant le seuil de 10⁻⁶. L'intuition « r est trop grand » est confirmée, et de
loin.

**Écart n°4, mineur : le readout.** Le papier lit `u₁`, point. Sur la chaîne corrigée
seul `shallow` a la bonne pente :

| readout | pente 10³–10⁵ |
|---|---|
| **`shallow` = u₁** | **−0,500** |
| `mean` / `sum` | −0,397 |
| `deep` | +5,17 (croît) |

`shallow` est déjà le défaut de la topologie `chain`. *(Le §5 concluait à `mean` sur la
base d'un span de dimension 1,1 où `mean ≈ u₁` ; une fois les `C_k` en place les
réservoirs se différencient et l'équivalence tombe.)*

### 5ter. Le mode « fuite terminale », chiffré sur le critère du papier

Le bon critère n'est pas la pente seule mais le **SNR de l'observateur idéal** de
Benna-Fusi : `SNR(t) = √N · r(t) / √(Σ_s r(s)²)`, où `r` est la réponse impulsionnelle de
`u₁`, le dénominateur est la variance **stationnaire** apportée par toutes les
modifications passées, et `N` est le nombre de synapses — pour nous la taille de la
matrice mémoire. La **durée de vie** est le dernier `t` où `SNR > 1`.

C'est ce critère qui arbitre la plasticité contre la rigidité : une fuite forte donne un
`SNR(0)` élevé et une durée de vie nulle, une fuite nulle l'inverse.

Matrice 128×128 (N = 16 384) :

| configuration | pente 10–10³ | SNR(0) | variance | durée de vie |
|---|---|---|---|---|
| accumulateur pur (`r(t)` = cte) | 0,000 | 0,1 | 2,0·10⁶ | **0** |
| **notre chaîne C_k=1, sans fuite** (= `cas_none`) | −0,216 | 1,1 | 1,4·10⁴ | **0** |
| **notre chaîne C_k=1 + fuite unif. r=0,1** (défaut) | −10,6 | 84,3 | 2,3 | **30** |
| **chaîne corrigée C_k=2^k, sans fuite** | −0,497 | 34,4 | 13,9 | **1 060** |
| chaîne corrigée + fuite uniforme r=10⁻⁶ | −0,497 | 36,8 | 12,1 | 1 216 |

Deux choses tranchées par ce tableau :

- **`cas_none` a une durée de vie de 0**, à peine mieux que l'accumulateur pur. Ta
  consigne de ne jamais publier de résultat cascade à r=0 est quantitativement justifiée.
  Mais la cause est le `C_k` manquant, **pas** l'absence de fuite : une fois `C_k = 2^k`
  en place, *sans aucune fuite*, la durée de vie passe à 1 060 — 35× notre défaut actuel.
  « Pas de fuite uniforme » et « accumulateur pur » cessent d'être la même chose.
- **Notre défaut r=0,1 tient 30 pas.** À 1 fps, trente secondes de mémoire.

### La fuite terminale : décisive à petit K, invisible à grand K

Elle n'agit que si la masse **atteint** le dernier bécher dans l'horizon utile, c'est-à-dire
si `τ_profond = C_K/g_K < horizon`. Matrice 128×128 :

| K | τ du plus profond | sans fuite term. | **avec** | gain |
|---|---|---|---|---|
| 4 | 2,6·10² | 1 | **816** | ×816 |
| **6** | 4,1·10³ | 28 | **1 605** | **×57** |
| 8 | 6,6·10⁴ | 363 | 1 258 | ×3,5 |
| 10 | 1,0·10⁶ | 1 026 | 1 065 | ×1,0 |
| 12 | 1,7·10⁷ | 1 060 | 1 060 | ×1,0 |
| 16 | 4,3·10⁹ | 1 060 | 1 060 | ×1,0 |

**Et l'optimum est K=6 avec fuite terminale (1 605), pas K=12 (1 060).** Au-delà de K≈8
les béchers supplémentaires absorbent du signal dans des modes trop lents pour être relus
dans l'horizon : ils ajoutent de la variance sans ajouter de signal récupérable.

C'est le résultat qui décide : **la fuite terminale vaut d'être implémentée précisément
parce qu'elle rend K=6 meilleur que K=12** — donc moins de VRAM (9,10 Mo/frame contre
11,1), et K=6 est déjà notre défaut. Sans elle, il faudrait monter à K=12 pour un
résultat 35 % moins bon.

### Et c'est la taille de la matrice mémoire, pas K, qui fixe l'horizon

Durée de vie (pas) avec fuite terminale, selon `N` = nombre d'éléments de la matrice :

| matrice | N | K=4 | **K=6** | **K=8** | K=10 | K=12 | K opt |
|---|---|---|---|---|---|---|---|
| **128×128** *(actuel)* | 16 384 | 816 | **1 605** | 1 258 | 1 065 | 1 060 | 6 |
| 256×256 | 65 536 | 1 194 | **5 775** | 5 032 | 4 263 | 4 243 | 6 |
| **512×512** | 262 144 | 1 572 | 12 170 | **20 127** | 17 055 | 16 975 | 8 |
| 1024×1024 | 1 048 576 | 1 949 | 18 634 | **76 267** | 68 224 | 67 902 | 8 |
| 2048×2048 | 4 194 304 | 2 327 | 25 099 | 178 486 | **272 884** | 271 607 | 10 |

La durée de vie croît en **N^0,93** — « almost linearly with the number of synapses »,
le résultat central du papier, reproduit ici indépendamment. C'est aussi le contrôle le
plus fort que la simulation est correcte.

Traduit à 1 fps :

| matrice | horizon | K optimal |
|---|---|---|
| **128×128** *(notre `dim=128`)* | **27 min** | 6 |
| 256×256 | 1 h 36 | 6 |
| **512×512** | **5 h 35** | **8** |
| 1024×1024 | 21 h | 8 |
| 2048×2048 | 76 h | 10 |

**Pour atteindre la médiane EgoLifeQA (5,9 h) il faut une matrice 512×512 avec K=8**
(526×526 exactement). Nous sommes à 128×128 : **facteur 13 sur le côté linéaire, soit 16×
en surface.**

C'est le §6 reformulé, et c'est une meilleure nouvelle : le risque n'était pas seulement
la fenêtre d'entraînement, c'est aussi la **capacité**, et la capacité est un choix de
conception, pas une contrainte de calcul. Mais les deux sont couplés — une matrice 16×
plus grande coûte de la VRAM par token, donc raccourcit la fenêtre. **À mesurer avant de
coder quoi que ce soit d'autre** : la VRAM à `dim=512, K=8` (l'extrapolation naïve donne
~146 Mo/frame, soit une fenêtre de ~105 frames, ce qui serait rédhibitoire — mais seule
une fraction du coût mesuré vient de l'état de cascade, donc il faut la mesure, pas
l'extrapolation).

### 5quater. Le domaine admissible, et le facteur de conversion qui manquait

Jusqu'ici je raisonnais sur `leak_rate`, le drapeau. Ce n'est pas la quantité qui agit :
la fuite appliquée est `exp(−leak_rate · θ_t)`, donc le taux effectif par pas est
`r_eff = 1 − exp(−leak_rate · ⟨θ_t⟩)`. **`⟨θ_t⟩` mesuré sur les cascades déjà entraînées**
(hook sur `theta_head`, 8 séquences du cache de validation) :

| artefact | `⟨θ_t⟩` | `θ_max` |
|---|---|---|
| `frozen_10k/cascade_s0.pt` | 0,0122 | 0,069 |
| `frozen_10k/cascade_s1.pt` | 0,0241 | 0,174 |
| `frozen_10k/cascade_s2.pt` | 0,0515 | 0,392 |
| `frozen/cas_tag.pt` | 0,0499 | 0,196 |

`⟨θ_t⟩ ≈ 0,03`, et il **varie d'un facteur 4 entre graines** — donc tout réglage de
`leak_rate` doit tenir sur la graine la plus chaude, pas sur la moyenne.

Ces quatre artefacts sont par ailleurs toutes les cascades entraînées que nous ayons, et
passées au scoreur elles donnent **pente −0,000 et durée de vie 0** : sans `C_k`, ce que
nous avons appelé « la cascade » dans tous les résultats antérieurs est un accumulateur
pur au sens du critère du papier. Aucun de ces runs ne mesurait ce que nous croyions
mesurer.

Carte du critère sur `(K, r_eff)`, béchers `c=2` appariés à `g=2`, fuite terminale active,
readout `shallow`, N = 16 384 (durée de vie en pas / pente sur 10–10³) :

| K | r=0 | 10⁻⁷ | 10⁻⁶ | 10⁻⁵ | 10⁻⁴ | 10⁻³ | 10⁻² |
|---|---|---|---|---|---|---|---|
| 4 | 816 | 816 | 816 | 812 | 779 | 555 | 157 |
| **5** | **1 596** | 1 596 | 1 593 | 1 569 | 1 366 | 660 | 157 |
| **6** | **1 605** | 1 605 | 1 604 | 1 592 | 1 412 | 662 | 157 |
| 7 | 1 411 | 1 412 | 1 420 | **1 472** | 1 404 | 662 | 157 |
| 8 | 1 275 | 1 278 | 1 303 | **1 437** | 1 404 | 662 | 157 |
| 12 | 1 251 | 1 255 | 1 284 | **1 435** | 1 404 | 662 | 157 |
| *pente* | −0,497 | −0,497 | −0,497 | −0,500 | −0,526 | −0,789 | −3,425 |

Trois lectures :

- **Le domaine admissible est `r_eff ≤ 10⁻⁵`.** À 10⁻⁴ la pente sort déjà de la tolérance
  (−0,526) et on perd 12 % de durée de vie ; à 10⁻³ la loi de puissance est morte.
- **K=4 n'atteint jamais −0,5** (−0,695) : trop peu de béchers pour couvrir la gamme.
  L'optimum est **K=5–6**, et il est plat entre les deux.
- Non trivial : pour K ≥ 7 la durée de vie est **maximale à r_eff = 10⁻⁵, pas à r=0**
  (1 472 contre 1 411 à K=7). Une fuite faible rogne la traîne qui gonfle le dénominateur
  du SNR sans porter de signal relisible. À K=6 l'effet s'inverse et r=0 gagne de 1 %.

Converti en drapeau, sur la graine la plus chaude (`θ=0,051`) :

| `--leak-rate` | `r_eff` à θ=0,012 | à θ=0,051 | verdict |
|---|---|---|---|
| 10⁻⁶ | 1,2·10⁻⁸ | 5,1·10⁻⁸ | **accumulateur pur** — exclu par ta consigne |
| 10⁻⁴ | 1,2·10⁻⁶ | 5,1·10⁻⁶ | admissible |
| **3·10⁻⁴** | 3,6·10⁻⁶ | **1,5·10⁻⁵** | **admissible, fuite réelle** |
| 10⁻³ | 1,2·10⁻⁵ | 5,1·10⁻⁵ | marginal |
| 10⁻² | 1,2·10⁻⁴ | 5,1·10⁻⁴ | détruit la loi |
| **10⁻¹** *(défaut actuel)* | 1,2·10⁻³ | 5,1·10⁻³ | **détruit la loi** |

**Ton intuition était juste et le facteur est ~300.** Le défaut `0,1` place la chaîne deux
décades au-delà du domaine admissible : pente −0,79, durée de vie 662 au lieu de 1 605.

`3·10⁻⁴` est le réglage retenu : c'est le plus grand qui reste admissible sur toutes les
graines, donc celui qui garde une **vraie** fuite par la surprise — conforme à « un leak
par la surprise est moralement la meilleure solution, seulement il doit être très lent »,
et sans retomber sur l'accumulateur pur que `10⁻⁶` produisait.

### Le readout : les deux critères se contredisent

| readout | pente 10–10³ | durée de vie |
|---|---|---|
| **`shallow`** (= `u₁` de Benna-Fusi) | **−0,497** | 1 605 |
| `mean` / `sum` | −0,323 | **3 697** |
| `deep` | +2,385 | 7 758 |

`gated` (notre défaut) est une mixture softmax ; son espérance non entraînée est `mean`.

C'est le même désaccord que rétention contre `memory_gain` : **`shallow` est le seul
fidèle à la loi −1/2, mais `mean` décroît plus lentement donc vit 2,3× plus longtemps.**
La cible −1/2 de Benna-Fusi est une borne *asymptotique* — la décroissance la plus lente
gardant la variance finie sur une chaîne infinie. La nôtre a K=6 et une frontière
terminale, donc la variance est finie de toute façon et l'argument ne mord pas.

Je ne trancherai pas cet axe a priori : il sort de l'échelle d'ablation et devient une
question séparée, à mesurer sur le score d'écho plutôt qu'à décider sur la théorie.

**Correction, mesurée sur les artefacts du smoke.** Passés au scoreur, `bf2_areas` et
`bf3_term` donnent une pente conçue de **−0,322**, pas −0,497 : ils héritent de
`readout=gated`, et `gated` plafonne à −0,32 **quoi qu'on corrige par ailleurs**. Le
readout n'est donc pas un axe indépendant qu'on peut laisser de côté — c'est une condition
*nécessaire* de la loi de puissance. Une ablation sans bras `shallow` ne contient aucune
configuration atteignant −0,5 et reste muette sur la question qu'elle prétend trancher.
D'où l'ajout de `bf5_shallow` (= `bf4_slow` + `--readout shallow`).

Le scoreur montre aussi que `r_eff ≈ 1,1·10⁻³` sur `bf1`/`bf2`/`bf3` — tous héritent de
`leak-rate 0.1`. À ce niveau de fuite, **ajouter les `areas` dégrade la durée de vie**
(1 011 contre 1 630 pour la chaîne sans `areas`) : l'enveloppe exponentielle coupe avant
que la géométrie ne serve. Prédiction avant résultats : `bf2` et `bf3` ne montreront aucun
gain sur le score d'écho, et le saut — s'il existe — sera sur `bf4_slow` et `bf5_shallow`.

### 5quinquies. La recette de réglage, en forme close

Les tableaux précédents sont des balayages. En les recoupant, les deux réglages qui
restaient empiriques — `K` et `leak_rate` — se déduisent tous les deux du **seul** horizon
visé `T`. Ce n'est pas un ajustement : les deux formules tombent de la dynamique.

**1. La fuite fixe jusqu'où la loi de puissance tient.** La trace est
`r(t) = t^(−1/2)·e^(−r_eff·t)`, donc la pente locale vaut exactement

> `d log r / d log t = −1/2 − r_eff·t`

L'écart à −1/2 est donc `r_eff·t` : la loi tient jusqu'à `t ≈ tol/r_eff`, et pas au-delà.
Vérifié à K=16 sur trois décades de `r_eff` — pente −0,60 à `t = 0,1/r_eff` et −1,5 à
`t = 1/r_eff`, identique à chaque décade, signature d'une loi d'échelle pure. **C'est ce
qui réconcilie les deux mesures contradictoires** : lue sur 10–10³ ou sur 10³–10⁵, la même
chaîne donne des verdicts décalés d'une décade en `r`, parce qu'une exponentielle mord
d'abord aux grands `t`.

> **`r_eff ≤ 0,1 / T`**, soit **`leak_rate ≤ 0,1 / (T · ⟨θ_t⟩)`**

**2. La profondeur fixe jusqu'où la chaîne *peut* aller.** Le bécher le plus profond a
`τ = C_K/g_K = c^(K−1)/(β₁·g^(−K))`, soit avec `c = g = 2`, `β₁ = 0,5` :

> **`τ_profond = 4^K`** → **`K ≥ log₄(T)`**

Reproduit exactement les `τ` mesurés en §5ter (K=4 → 2,6·10² ; 6 → 4,1·10³ ; 8 → 6,6·10⁴ ;
10 → 1,0·10⁶ ; 12 → 1,7·10⁷, cinq sur cinq). Et surtout, ça **explique** la colonne
« K optimal » que j'avais balayée en aveugle sur la taille de matrice :

| matrice | durée de vie | `log₄(T)` | K optimal balayé |
|---|---|---|---|
| **128×128** *(actuel)* | 1 605 | 5,3 | **6** |
| 256×256 | 5 760 | 6,2 | 6 |
| 512×512 | 20 100 | 7,2 | **8** |
| 1024×1024 | 75 600 | 8,1 | 8 |
| 2048×2048 | 273 600 | 9,0 | 10 |

À ±1 sur quatre décades de `N`. Le « K optimal » n'est donc pas un compromis mystérieux :
c'est **la chaîne la moins profonde dont le dernier bécher atteint l'horizon**. Plus court,
la masse butte sur la frontière terminale et la trace tombe d'une falaise ; plus long, les
béchers en trop absorbent du signal dans des modes trop lents pour être relus — ils
ajoutent de la variance au dénominateur du SNR sans ajouter de signal.

**La recette complète**, pour un horizon `T` visé :

| | règle | à `T` = 1 600 (128², actuel) | à `T` = 21 000 (médiane EgoLifeQA) |
|---|---|---|---|
| `K` | `⌈log₄ T⌉` | **6** | 8 |
| `leak_rate` | `0,1/(T·⟨θ⟩)`, θ = 0,05 | **1,3·10⁻³** | 9,5·10⁻⁵ |
| `cascade_areas` | `c = g` | **2,0** | 2,0 |
| fuite terminale | toujours | **oui** | oui |
| `N` | `T ∝ N^0,93` | 16 384 | 262 144 (512²) |

Le `3·10⁻⁴` retenu en §5quater est un cran sous le plafond `1,3·10⁻³` de la ligne actuelle
— délibérément, parce que `⟨θ_t⟩` varie d'un facteur 4 entre graines et que le plafond doit
tenir sur la plus chaude. Il couvre `T ≈ 6 700`, soit 4× l'horizon actuel.

**Ce que ça coûte de se tromper d'axe** : monter `N` sans monter `K` ne sert à rien (la
chaîne plafonne à `4^K`), et monter `K` sans baisser `leak_rate` ne sert à rien non plus
(l'enveloppe exponentielle coupe avant le dernier bécher). Les trois bougent ensemble.

### Ce que j'abandonne : la chaîne dirigée

**Retirée.** La bidirectionnalité est *la* contribution du papier : « the interactions
between fast and slow variables are bidirectional », et c'est elle qui fait passer la
capacité de `√N` (le modèle cascade de 2005, réf. 9) à ~`N`. Une chaîne dirigée est
exactement le modèle antérieur, moins bon. Le gain de rang ×3,3 que je mesurais achetait
une propriété dont le papier montre qu'elle ne sert pas, au prix de celle qui sert.

### Ce qui est retenu

| | valeur | raison |
|---|---|---|
| topologie | `chain`, **bidirectionnelle** | contribution centrale de Benna-Fusi |
| **`C_k = g^k`** | **à implémenter** | l'ingrédient manquant : −0,066 → −0,500 |
| **fuite terminale** (`u_{K}=0`) | **à implémenter** | tue le mode λ=1 |
| `leak_rate` r | **3·10⁻⁴** (contre 0,1) | §5quater : le plus grand `r_eff ≤ 10⁻⁵` sur toutes les graines |
| readout | **non tranché** | `shallow` seul à −0,5, mais `mean` vit 2,3× plus longtemps |
| K | **6** (et non 12) | §5ter : optimum 1 605 avec fuite terminale ; plat entre 5 et 6 |
| g, β₁ | 2 ; 0,5 | `c = gg = 2` donne l'exposant −1/2 |

L'option B (K=17, g=2) est abandonnée : 2,3× la VRAM de `sane_ns`, fenêtre réduite de 532
à 416 frames, pour une portée que les `C_k` fournissent à K=12.

**Le critère de validation change aussi** : ce n'est plus « la cascade bat-elle le
scalaire sur `memory_gain` », c'est **la pente de `u₁(t)` vaut-elle −0,5 sur plusieurs
décades**. C'est mesurable sur un artefact gelé, sans réentraîner.

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
