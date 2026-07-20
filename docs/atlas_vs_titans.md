# ATLAS vs Titans — synthèse

**ATLAS** — *"Atlas: Learning to Optimally Memorize the Context at Test Time"*,
Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni (Google Research),
arXiv:2505.23735 (mai 2025). Même équipe que Titans (arXiv:2501.00663).

## Idée centrale

ATLAS identifie **trois défauts** des mémoires récurrentes type Titans et corrige
chacun. **Titans est un cas particulier dégénéré d'ATLAS** : fenêtre `c=1`, pas de
feature map (`φ=id`), optimiseur interne du 1er ordre.

| Défaut de Titans | Correction ATLAS |
|---|---|
| ① Capacité mémoire limitée | Feature maps **polynomiales** sur k/q |
| ② Mise à jour *online* (le seul dernier token) | **Règle Omega** (fenêtre glissante) |
| ③ Gestion mémoire peu expressive (GD 1er ordre) | Optimiseur **Muon** (2nd ordre) → *Atlas* |

## ① Capacité — feature maps d'ordre supérieur

Titans apprend `M(k) ≈ v`. La capacité (nombre d'associations clé→valeur
linéairement indépendantes stockables dans une mémoire de taille fixe) est bornée
par l'architecture de `M` et la dimension des clés.

ATLAS applique un **noyau polynomial** `φ` : on apprend `M(φ(k)) ≈ v`. Les
features d'ordre supérieur augmentent le nombre d'associations séparables
(justification théorique dans le papier). Une mémoire plus **profonde** aide
également.

## ② Règle Omega — la fenêtre glissante

Titans (règle *Delta*, online) : à chaque pas, la mémoire minimise la perte
associative **du seul token courant** :

```
min_M  ‖M(k_t) − v_t‖²                                   (c = 1)
```

→ la mémoire mémorise des **tokens individuels**.

ATLAS (règle **Omega**) : minimise une somme pondérée sur une **fenêtre glissante**
des `c` derniers tokens :

```
min_M  Σ_{i=t−c+1}^{t}  γ_i^(t) · ‖M(k_i) − v_i‖²         (Eq 9)
```

- `c` : longueur de la fenêtre de contexte local.
- `γ_i^(t) ∈ [0,1]` : **portes data-dependent** qui élaguent quels tokens passés
  comptent (pruning en contexte).
- Cas limites : `c=1` → **Titans** ; `c=∞` → optimisation globale sur tout le passé.

→ la mémoire mémorise un **contexte local** au lieu de tokens isolés. Coût
constant par token (`c` portes), donc parallélisable comme Titans.

## ③ Muon — mémoire « localement optimale » (2nd ordre)

Titans met à jour par **descente de gradient + momentum** (1er ordre) → peut se
coincer dans un mauvais optimum local.

Atlas utilise l'optimiseur **Muon** pour la boucle interne :

```
M_t = α_t M_{t−1} − η_t · NewtonSchulz_k(S_t)            (Eq 32)
S_t = θ_t S_{t−1} + ∇[ perte Omega ]                    (Eq 33)
```

- `NewtonSchulz_k` approxime de l'**info du 2nd ordre** (projette le momentum `S_t`
  sur la matrice semi-orthogonale la plus proche).
- `k` (nb d'itérations) = **bouton de compute à test-time** : plus d'itérations →
  meilleure mémorisation. Premier modèle récurrent parallélisable à mémoire
  localement optimale.

**Deux modèles** : **OmegaNet** = Omega + `φ` polynomial + GD momentum ; **Atlas**
= OmegaNet + Muon.

## Bonus — DeepTransformers

La règle Omega **relie** la mémoire à l'attention softmax : globale (attention
classique) et **locale = Sliding Window Attention**. D'où les **DeepTransformers**
(et la variante fenêtre glissante **SWDT**) qui **généralisent strictement** le
Transformer. Baseline **DLA** (Deep Linear Attention) pour isoler le rôle de la
mémoire profonde.

**Résultat phare** : Atlas améliore le long contexte de Titans — **+80 %** de
précision à **10M** de contexte sur BABILong.

## Correspondance avec ce repo

Notre archi (mémoire test-time + update chunké §3.2) est la **bonne fondation** —
ATLAS est une extension, pas une réécriture.

| Brique ATLAS | État dans le repo |
|---|---|
| ② Règle Omega (fenêtre `c`) | ✅ implémentée — `NeuralMemory(window_size=c)`, portes `γ` via `gamma_head`, somme glissante des gradients par-token dans `_scan_chunked`. `c=1` = Titans bit-identique. |
| ① Feature map `φ` polynomiale | ✅ implémentée — `titans/features.py::PolynomialFeatureMap`, via `feature_degree=p` (défaut 1 = identité = Titans). `φ` est appliqué aux clés dans `_project` et aux requêtes dans `retrieve`. **Déviation** : sous-ensemble diagonal (puissances pures `x_j^i`), pas la base monomiale complète `[x^β]` — les termes croisés sont abandonnés, car la base complète fait 2145 dims à `d=64,p=2` et notre `M` est un MLP (pas d'astuce du noyau possible). Le tensor-sketch de Kacham et al. 2024a serait l'alternative fidèle. |
| ③ Optimiseur Muon (Newton-Schulz) | ✅ implémenté — `memory.py::newton_schulz5` (itération quintique), via `orthogonalize_momentum=True`. Appliqué au momentum avant qu'il ne mette à jour les poids 2D, et au readout sur le chemin cascade. |
| Recette mémoire Table 7 | ✅ implémentée — expansion 4, GELU, résiduel `M(x) = x + W₁σ(W₂x)`, via `memory_expansion` / `memory_activation` / `memory_residual`. Le résiduel est mal typé quand `φ` élargit l'entrée ; on ajoute la tranche de degré 1 (la clé non mappée). |
| DeepTransformers / SWDT | ⬜ hors scope pour l'instant. |

**L'update rule Atlas (Eq 57-58) est donc complète.** Reste ouvert : la fenêtre Omega
est tronquée en début de chunk (largeurs `[1,2,3,4,…]` au lieu d'un `c` uniforme), et
la somme d'Eq 58 gonfle le pas interne avec `c` — `window_normalize=True` (opt-in)
permet de démêler « plus de contexte » de « pas plus grand ».

**Note d'implémentation (Omega)** : le gradient étant linéaire,
`∇[Σ γ_i ‖M(k_i)−v_i‖²] = Σ γ_i ∇‖M(k_i)−v_i‖²`. On réutilise donc les gradients
par-token déjà calculés (aucun backward supplémentaire) ; la fenêtre est clampée
aux frontières de chunk (approximation documentée — le flux long-portée passe
aussi par l'état mémoire threadé). Voir `tests/test_omega.py`.
