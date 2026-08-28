# Predictions — re-ancrage sur bases saines, 2026-08-27

Decision de Josep (27/08) : les runs `rescale` (hybride maison : direction
NS + magnitude brute) ne peuvent servir d'exemple de rien ; les resultats
qui portaient ce biais sont retires du brouillon. Deux volets pour
repartir sur des configs ou l'on sait exactement ce qu'on teste.
Enregistre AVANT lancement.

## Volet T — echelle Titans-LITTERALE (diag_titans_train.py)

Config : base sane (portes theta/eta/alpha identiques a la famille,
eta_init 0.995, alpha_init 0.002, dim 128, seg 8, batch 8, horizon 24,
Adam 1e-3, BPTT plein, [V;V]), **memory_orthogonalize_momentum=False** :
M_t = (1-alpha)M_{t-1} + S_t, S_t = eta S_{t-1} - theta grad — Titans tel
que publie, aucun NS, aucun rescale. Echelle theta_scale
{0.03, 0.05, 0.08, 0.12} x graines {0,1}, 1500 pas, trace mecanisme par
chunk a chaque pas.

- **P-T1** : le mur existe pour l'objet publie, dans une bande comparable
  a celle observee sur l'hybride retire : theta* dans (0.03, 0.12], avec
  transition rasoir (un barreau survit 1500 pas, le suivant meurt en
  < 20 pas). Fondement : la courbure vient de la PERTE quadratique
  (lambda = 2||h||^2), pas de la direction de l'update.
- **P-T2** : toutes les morts sont D1 nees au forward (params sains avant
  forward, etat non fini pendant, gradients empoisonnes apres) — comme
  la theorie l'exige pour une explosion de boucle interne.
- **P-T3** : au pas fatal, la trace par chunk montre theta plat et
  theta*lambda_hidden franchissant ~2 avant la premiere non-finitude,
  lambda_keys borne (< 100) — la signature mecanisme, sur l'objet publie.
- **P-T4 (controle)** : le barreau le plus bas survit 1500 pas avec
  theta*lambda_hidden < 2 tout du long.

## Volet B — sonde A (chemin de lecture) sur les artefacts BORNES

diag_readout_probe.py inchange, sur frozen_muon/mu_unit_s*.pt,
mu_usq10k_s*.pt, mu_scl_s*.pt (les conventions du regime actuel :
ATLAS-litteral `unit`, arctan-squash calibre, softclip calibre), gaps
{0, 2, 4, 8}, n_eval 96. Le meme protocole qui a revele l'explosion du
bloc 1 sur l'hybride retire.

- **P-B1** : AUCUNE explosion d'etat a aucun gap sur les bras bornes
  (zero evenement update/retrieve non fini) — le pas borne tient au
  rejeu ; le mur du rejeu n'existe pas dans le regime actuel.
- **P-B2** : les scores l1 restent finis partout.
- **P-B3 (la question collapse)** : le pr decroit avec le gap sur les
  bras bornes aussi (l'ecrasement du rang avec la distance est un
  phenomene du regime SILENCIEUX, pas un artefact de l'hybride), avec
  un plancher plus haut que ce que l'hybride montrait ; cos_ret monte
  avec le gap.
- **P-B4 (alternative pre-enregistree)** : si un bras borne montre des
  scores NaN avec etats finis partout, ALORS il existe un second
  mecanisme de mur (lecture), a sonder separement.

## Verdicts (27/08 — runner lance 15:23, fini 15:44 ; tout est revenu en 21 min)

### Volet T — echelle Titans-litterale (8 runs, tous morts)

| run | theta | pas de mort | type | theta plat (CV) | lambda_keys max | thl_hidden franchit 2 |
|---|---|---|---|---|---|---|
| tl_t003_s0 | 0.03 | 8  | D1 forward | 2.2% | 70.6 | oui (pas 1) |
| tl_t003_s1 | 0.03 | 6  | D1 forward | 0.5% | 70.5 | non vu bloc 0 (voir note) |
| tl_t005_s0 | 0.05 | 1  | D1 forward | 2.3% | 69.2 | oui |
| tl_t005_s1 | 0.05 | 0  | D1 forward | 0.4% | 69.5 | oui (chunk 15/16) |
| tl_t008_s0 | 0.08 | 0  | D1 forward | 1.3% | 67.9 | oui (chunk 8) |
| tl_t008_s1 | 0.08 | 0  | D1 forward | 0.6% | 70.0 | oui (chunk 8) |
| tl_t012_s0 | 0.12 | 0  | D1 forward | 1.4% | 67.1 | oui (chunk 5) |
| tl_t012_s1 | 0.12 | 0  | D1 forward | 0.2% | 68.2 | oui (chunk 7) |

- **P-T1 : TOMBEE (dans le sens dur)**. Le mur existe mais il est SOUS
  l'echelle : theta* < 0.03 sur cette config. Aucun barreau ne survit ;
  la transition rasoir attendue survie/mort devient un rasoir en TEMPS
  de mort (8-6 pas a 0.03 ; 1-0 a 0.05 ; 0 partout au-dessus — la mort
  arrive DANS le premier forward des theta >= 0.08, en <= 32 ecritures).
- **P-T2 : CONFIRMEE 8/8**. Signature D1 exacte : loss du pas fatal
  encore finie (0.94-1.03 quand elle existe), first_bad_upd atteint
  PENDANT le forward, p_bad = 0 au pas fatal puis 498 184 au pas
  suivant. Aucune D2.
- **P-T3 : CONFIRMEE**. theta plat partout (CV 0.2-2.3%),
  lambda_keys < 100 partout (max 70.6 — la parametrisation borne les
  cles, le tueur est lambda_hidden), theta*lambda_hidden franchit 2
  avant la premiere non-finitude dans 7/8 traces. tl_t003_s0 est la
  demonstration loss-aveugle sur l'objet publie : w1_max fait
  18 -> 1.4e2 -> 9.1e2 -> 2.3e5 -> 5.2e7 -> 1.1e9 -> 4.9e9 -> 2.1e14
  pendant que la loss reste dans [0.94, 1.06] et que theta ne bouge pas.
- **P-T4 : TOMBEE**. Le barreau 0.03 meurt aussi (pas 6-8).
- **Note instrument (encore le trou mono-bloc)** : la trace filme le
  bloc 0 seulement (16 chunks) ; sur tl_t003_s1 le max de
  theta*lambda_hidden fini au bloc 0 est 1.5 alors que w1_max global
  atteint 3.4e18 — l'explosion vivait dans le bloc 1. Toute figure
  mecanisme doit le dire, et la sonde multi-bloc reste prioritaire.

### Volet B — sonde de lecture sur bras bornes

| bras | pr (gap 0 -> 8) | l1 (0 -> 8) | w1max max | etats |
|---|---|---|---|---|
| mu_unit (5 graines) | 44.6 -> 44.6 (plat ; par graine 14-53 !) | 0.65 -> 0.84 | 0.98 | tous finis |
| mu_usq10k (5) | 21.4 -> 31.3 (MONTE) | 0.36 -> 1.57 | 3.0 | tous finis |
| mu_scl (5 graines) | 68.7-71.3, plat a tous les gaps | 0.67 -> 0.68 | 0.27 | tous finis |

- **P-B1 : CONFIRMEE**. Zero evenement, zero batch non fini, sante finie
  partout : le mur du rejeu n'existe pas dans le regime borne.
- **P-B2 : CONFIRMEE**. Tous les l1 finis.
- **P-B3 : TOMBEE — et c'est le resultat le plus utile**. pr ne decroit
  PAS avec le gap : plat (unit, softclip) ou croissant (usq10k, ou l1 se
  degrade pourtant de 0.36 a 1.57). Le rang n'est pas une fonction de la
  distance au rejeu : c'est une propriete GELEE dans l'artefact a
  l'entrainement — mu_unit s'etale de pr 14 a pr 53 selon la GRAINE, a
  gap identique. Consequence directe pour le programme collapse : la loi
  visee est pr_infini(entrainement) = f(alpha, rho, d), a verifier sur
  les artefacts geles (les donnees existent), pas une loi de rejeu ; et
  la regle du classifieur doit etre un pr ABSOLU par artefact.
- **P-B4 : non declenchee** (aucun NaN de score sur etats finis).
