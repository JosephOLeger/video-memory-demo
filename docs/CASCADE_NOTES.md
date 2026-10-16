# Cascade métaplastique : ce qu'on a appris (août 2026)

Note pour la réimplémentation. Le code est retiré du dépôt depuis le 16/10. Il reste
récupérable à deux endroits :
- l'étiquette git `archive/cascade-2026-10-16` (ou le commit `6a044ea`) : `src/titans/metaplastic.py`
  et le chemin cascade de `src/titans/memory.py` ;
- la branche `cascade-benna-fusi`.

Référence : Benna et Fusi, *Computational principles of synaptic memory consolidation*,
Nat. Neurosci. 19(12), 2016 (`ASTAR/Biblio/cascade_synaptic_memory_2016.pdf`).

## L'objet

Le momentum scalaire `S_t = η S_{t-1} − θ g_t` est remplacé par K réservoirs `u_1..u_K` par poids :

    C_k du_k/dt = g_{k-1,k} (u_{k-1} − u_k) + g_{k,k+1} (u_{k+1} − u_k),   u_{K+1} = 0

Le gradient est injecté dans `u_1`, et la lecture est `u_1`. Les autres réservoirs ne sont
pas des échelles de temps qu'on lit séparément. Ils servent à donner à `u_1(t)` une
décroissance en loi de puissance `t^{-1/2}` : c'est la plus lente qui garde la variance
du poids finie. La chaîne doit être bidirectionnelle ; c'est ce qui fait passer la
capacité de √N à environ N.

## Le bon critère de validation

**La pente log-log de la lecture doit valoir −1/2 sur plusieurs décades.** Elle se mesure
en forme close, sans entraînement, sur la matrice de transition. Battre le scalaire sur
`memory_gain` n'est pas un critère. Le rang effectif de la banque de filtres non plus :
17 réservoirs n'en valent qu'environ 1,7 filtre indépendant, et c'est voulu par le design.

## Les trois défauts de notre ancienne implémentation

| | pente sur 10³ à 10⁵ |
|---|---|
| notre chaîne (C_k = 1, pas de fuite terminale) | −0,066 |
| + fuite terminale seule | −0,264 |
| + aires C_k = 2^k (Benna et Fusi, Fig. 2c) | **−0,500** |

1. **Pas d'aires C_k.** Le flux était appliqué de façon symétrique, ce qui revient à C_k ≡ 1.
   Règle : avec `C_k = c^k` et `g_e = β₁ g^{−e}`, il faut prendre **c = g** pour obtenir −1/2.
2. **Pas de fuite terminale.** Le réservoir le plus profond recevait de la masse sans en
   rendre. B avait alors une valeur propre exactement égale à 1 (mode conservé), et κ, le
   nombre de fois qu'une surprise est réappliquée, croissait linéairement (1 682 à
   n = 10⁴, contre 185 en momentum scalaire). C'est la cause de la divergence aux temps
   longs. **La fuite de bout de chaîne stabilise ; la fuite uniforme n'était pas en cause.**
3. **Une fuite uniforme mal placée et environ 300 fois trop forte.** Multiplier tout l'état
   par `exp(−r θ_t)` impose une enveloppe exponentielle qui coupe la loi de puissance.

## Le réglage en forme close (tout découle de l'horizon T visé, en pas)

- Pente locale de la lecture : `d log r / d log t = −1/2 − r_eff t`. Il faut donc
  **`r_eff ≤ 0,1/T`**, avec `r_eff = 1 − exp(−leak_rate ⟨θ_t⟩)`. Réglez sur la graine la plus
  chaude : ⟨θ_t⟩ varie d'un facteur 4 entre graines.
- Le réservoir le plus profond a `τ = 4^K` (pour `c = g = 2`, `β₁ = 0,5`). Il faut donc
  **`K ≥ log₄ T`**. Ce critère reproduit à ±1 les K optimaux trouvés à l'aveugle sur quatre
  décades de N.
- La capacité vient de la taille de la matrice N, pas de K : la durée de vie croît comme
  N^0,93. 128² donne environ 1 600 pas ; il faut 512² avec K = 8 pour la médiane d'EgoLifeQA
  (5,9 h).
- Point de fonctionnement retenu à 128² : chaîne, `C_k = 2^k`, fuite terminale, K = 6,
  g = 2, β₁ = 0,5, `leak_rate = 3e-4`. Avec la fuite terminale, K = 6 bat K = 12 (durée de
  vie 1 605 contre 1 060).
- La lecture n'est pas tranchée. `u_1` seule donne exactement −1/2. `mean` décroît plus
  lentement et vit 2,3 fois plus longtemps. C'est le score echo qui doit trancher.

## Stabilité (calculs validés)

- La carte de chunk du scalaire se transpose en remplaçant `η^r` par le noyau de retour
  `φ(r) = cᵀ Π^r e₁`.
- **Théorème du facteur K.** B est doublement stochastique. Avec `readout = mean`, on a
  donc `φ(r) = 1/K` pour tout r, c'est-à-dire un intégrateur parfait de coût
  `κ = m(m+1)/(2K)` : 1/K du coût du scalaire quand η → 1.
- À rétention égale, l'amplification par séquence (T = 144) vaut 1,4 pour la cascade
  contre 189 pour le scalaire.

## Résultats empiriques (sur l'ancienne chaîne, donc à refaire)

- **Elle achète du budget en θ.** Sur MQAR, 12 graines : à θ = 1, 5 runs sains sur 12 contre
  0 sur 12 (p = 0,037). Toutes valeurs de θ confondues, 13 divergences sur 48 contre 30 sur
  48 (p = 0,0009). Le `memory_gain` sur la vidéo est le même.
- **Elle déplace la borne.** Le `memory_gain` et la marge suivent une seule courbe
  monotone à travers les deux familles. La cascade la prolonge au-delà du seuil (marge < 1)
  et casse vers 200 à 600 tokens sur flux long. À convergence (10k pas), elle a le
  meilleur rappel immédiat du projet, mais aucun horizon.
- Ces deux résultats sont cohérents avec le mode conservé : κ de la cascade est plus
  petit que celui du scalaire sur séquences courtes (28,6 contre 100 à n = 100), et le
  dépasse vers n ≈ 1 000.

## Règles à garder

- Jamais de résultat sur l'accumulateur pur (r = 0, `cas_none`) ni sous le seuil de fuite
  (`cas_r0.03`) présenté comme « la cascade ».
- Ne jamais justifier K par « K échelles de temps ». Le portage à longue distance, c'est α
  et la taille N.
- La version sélective (`B_t` dépendant du token) est la vraie revendication. Une banque
  à B fixe ne peut pas la reproduire, donc la comparer à cette banque, pas seulement à
  « pas de cascade ».

## Pour la réimplémentation

Implémenter la chaîne de Benna et Fusi complète (C_k, fuite terminale, fuite uniforme
optionnelle et minuscule). La valider d'abord par la pente −1/2 en forme close, puis
régler K et `leak_rate` à partir de T. Elle se combine naturellement avec l'écriture
bornée `unit` ou `unit_squash` (θ sur NS5 de la lecture), qui n'existait pas encore
quand la cascade a été testée.
