# Prédiction — la cascade compense une dilution, écrite avant de lancer

Date : 2026-08-14. Écrit **avant** que `jobs_cas.txt` ne parte, pour la même raison que
le test en aveugle de la marge du 12/08 : une hypothèse notée après coup n'est pas
réfutable.

## L'hypothèse

Dans la topologie `chain`, le gradient n'est injecté que dans le réservoir 0, qui perd
ensuite `base_betas[0] = beta1·g⁻¹ = 0.25` de sa masse par token vers les réservoirs
profonds. Tous les readouts disponibles sauf `sum` (`shallow`, `mean`, `learned`,
`gated`) restent à l'échelle d'un filtre unique. Le momentum visible est donc
structurellement plus petit, à θ égal, qu'un momentum scalaire ordinaire.

**La boucle externe compenserait cette dilution en montant le pas effectif** — et c'est
exactement la quantité dans laquelle le critère de stabilité est écrit. La cascade ne
serait donc pas instable en soi (sa matrice `B` est doublement stochastique, rayon
spectral exactement 1 : elle ne peut pas exploser seule) : son design pousserait
l'entraînement dans le coin instable.

## L'état des mesures

58 artefacts `frozen_vv`, moyenne ± écart-type sur les graines. **`cas_none` et
`cas_r0.03` sont exclus** : le premier est l'accumulateur pur (r = 0), le second est
sous le seuil r < 0.1 où le leak ne fait rien — et les résultats cascade ne doivent pas
s'appuyer sur des accumulateurs purs (consigne du 12/08). Restent 18 artefacts cascade
à fuite réelle (`cas_r0.1`, `cas_r0.3`, `cas_r1.0`) contre 40 scalaires.

| séparateur | cascade (fuite réelle) | scalaire | Cohen d |
|---|---|---|---|
| **θ_max** | 0.0630 ± 0.0305 | 0.0165 ± 0.0132 | **1.98** |
| **‖h₀‖ max** | 2.185 ± 0.560 | 1.194 ± 0.480 | **1.90** |
| α_moyen | 0.00040 ± 0.0005 | 0.00120 ± 0.0008 | −1.20 |
| θ·λ_hidden (la quantité du critère) | 3.25 ± 3.79 | 0.144 ± 0.061 | 1.16 |

L'exclusion **renforce** la conclusion : sur l'ensemble complet θ donnait d = 1.46, sur
les seuls bras à fuite réelle d = 1.98. Les accumulateurs purs étaient les plus
dispersés (`cas_none` couvre un facteur 24 sur θ selon la graine) et diluaient l'effet.

Deux mises en garde qui comptent :

- **La dispersion par graine est énorme sur θ.** `cas_none` va de 0.006 à 0.143 selon
  la graine, un facteur 24. Les distributions cascade et scalaire **se recouvrent** sur
  θ seul. Le facteur ~19 souvent cité vaut pour θ·λ_hidden, pas pour θ.
- **Le signal le plus propre est ‖h₀‖**, pas θ — ce qui recoupe la trajectoire 40 000
  pas, où la marge chutait ×3.1 avec θ qui *baissait* et ‖h₀‖ qui montait ×3.6.

La compensation reste monotone dans le leak sur les médianes (`cas_r1.0` 0.098 >
`cas_r0.3` 0.052 > `cas_r0.1` 0.039), et `cas_r1.0` est le seul bras dont **les six
graines** dépassent toutes les médianes scalaires. C'est ce qui rend le balayage du
leak plat en survie : chaque cran de fuite est repayé en pas effectif.

## Le protocole

5 bras × 5 graines × 2500 pas, ~36 h GPU. Rien n'est bridé : θ et α restent librement
appris. On change la **structure** et on regarde où l'entraînement les emmène — les
portes choisies sont la mesure.

| bras | changement | ce qu'il teste |
|---|---|---|
| `casCTL` | aucun | contrôle apparié en batch et en pas |
| `casA_sum` | `readout=sum` | récupère la masse partie en profondeur |
| `casB_b0.1` | `beta1 = 0.1` | le réservoir 0 se vide 2.5× moins |
| `casC_sum_b0.1` | les deux | l'effet combiné |
| `casE_nh` | `normalize_hidden = 1.0` | borne ‖h₀‖, le séparateur le plus net |

**Critère principal : ‖h₀‖ max** (d = 2.11, la seule mesure où n=5 a de la puissance).
Secondaires : θ·λ_hidden, θ_max, α.

## Les prédictions

Ordre attendu sur ‖h₀‖ : `casE_nh` < `casC` < `casA` < `casB` < `casCTL`.

- `casE_nh` : ‖h₀‖ **< 1.3** — la normalisation le borne par construction ; c'est le
  contrôle positif du dispositif plus qu'une prédiction.
- `casA_sum` : ‖h₀‖ **< 1.8**, θ·λ_hidden **< 1.0**.
- `casB_b0.1` : ‖h₀‖ < 2.0, effet plus faible que `casA`.
- `casC` : le plus bas des trois bras structurels.
- `casCTL` : reproduit `cas_r0.1`, soit ‖h₀‖ ≈ 2.2 ± 0.5.

## Le falsificateur

**Si ‖h₀‖ et θ·λ_hidden restent dans l'intervalle du contrôle sur `casA`, `casB` et
`casC`, l'hypothèse de dilution est fausse.** Ce ne serait pas un résultat nul : ça
éliminerait le readout et le couplage, et renverrait vers la boucle M → ∇ℓ → u
elle-même, dont `casE_nh` est justement le test.

Prédiction secondaire, testée par les sondes `long_run` : la survie ne dépasse les
384 tokens **que si** ‖h₀‖ descend. Si un bras baisse ‖h₀‖ sans gagner en survie, le
médiateur n'est pas le bon non plus.

## Ce que ça ne teste pas

`mem_effect` n'est pas prédit ici. Retirer la dilution peut très bien coûter en
réactivité — c'est précisément le compromis qu'on cherche à quantifier, et on n'a
aucune raison de parier sur son signe. Sur le bras scalaire, `normalize_hidden` coûtait
16 % de `mem_effect` ; sur la cascade, personne n'a jamais mesuré.
