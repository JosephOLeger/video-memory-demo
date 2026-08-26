# Audit D2 interne — classification de NOS NaN d'entrainement

Enregistre le 25/08/2026, AVANT lancement de l'echelle (regle du projet).

## Contexte

- `training.py` documente : "this project's memory diverges to NaN on ~13% of
  seeds" — le corpus a auditer existe et il est a nous.
- Le `train_loop` n'a AUCUN ecretage de gradient : `backward` -> `opt.step`
  direct. Tout gradient non fini empoisonne les params immediatement.
- La chaine NS par backward est courte : [V;V] a T=64, segment 8, BPTT plein
  => ~16 appels `memory.update` (compte exact loggue par l'espion), tres sous
  le point propre 64 maillons de la serie de doses sur la reference.
- Les logs historiques (logs/nan/nd_*.log, th_d32_s0.log, regime_theta10.log)
  montrent des morts deterministes ~pas 1639/1700 a cadence 100-250 pas :
  l'ORDRE des evenements (etat interne d'abord vs params empoisonnes d'abord)
  y est illisible. D'ou le rerun instrumente pas-a-pas.

## Instrument

`scripts/diag_d2_train.py` — rerun du config famille exact (base `sane_ns`,
mirror de `mu_rescale` : alpha_init 0.002, eta_init 0.995, dim 128, seg 8,
K=24, batch 8, Adam 1e-3, [V;V] via `double`), dose = `--theta-scale`.
Log PAR PAS, dans l'ordre :

- `p_bad`   : elements de params non finis AVANT le forward (empoisonnement
              par le pas precedent) ;
- `loss`    : perte forward (finie ?) ;
- `w1_max`, `S_max` : max de l'etat interne sur tous les `memory.update` du
              pas (espion, aucun changement de calcul) ; `n_upd` = compte ;
- `g_bad`/`g_n` : elements de grads non finis APRES backward, AVANT
              `opt.step` (il n'y a pas de clip dans cette boucle) ;
- `pr`      : ratio de participation des valeurs singulieres du w1 final de
              fenetre, tous les 25 pas (ride-along question du rang).

## Regle de classification (fixee avant lecture)

- **D1** : au pas de mort, `p_bad = 0` et `w1_max`/`loss` deviennent non
  finis PENDANT le forward — l'explosion est dans la boucle interne.
  Precurseurs attendus : tendance montante de `w1_max` avant la mort.
- **D2** : au pas de mort, `p_bad = 0`, loss FINIE, `w1_max` fini, mais
  `g_bad > 0` ; le pas suivant a `p_bad > 0` (params empoisonnes par
  `opt.step`).
- Toute mort qui ne rentre dans aucune case est rapportee telle quelle.

## Echelle (6 runs sequentiels, 1500 pas cap, arret mort+10)

| run     | muon    | theta_scale | seed |
|---------|---------|-------------|------|
| ctrl_s0 | rescale | 0.03        | 0    |
| t03_s0  | rescale | 0.3         | 0    |
| t03_s1  | rescale | 0.3         | 1    |
| t10_s0  | rescale | 1.0         | 0    |
| t10_s1  | rescale | 1.0         | 1    |
| u10_s0  | unit    | 1.0         | 0    |

## Predictions

- **P-D2i-1** : ctrl_s0 ne meurt pas en 1500 pas (famille 25/25 propre a
  theta 0.03).
- **P-D2i-2** : aucune mort au pas <= 2 — la chaine NS est courte (~16
  maillons), le chemin D2 deterministe de la reference (128 maillons) ne
  s'applique pas ici.
- **P-D2i-3 (la substantielle)** : les morts a theta 0.3/1.0 porteront la
  signature **D1** (explosion de l'etat interne au forward, precurseurs
  `w1_max`), PAS la signature D2 — coherent avec "abrupt, deterministe par
  graine" et avec le mecanisme rasoir de I5. **Branche alternative,
  pre-enregistree** : si les morts sont backward-nees (loss finie +
  `g_bad > 0`), la phrase du CLAIM I5 "coherent avec nos morts par graine
  cote world model" est RETIREE du brouillon et remplacee par "nos NaN
  d'entrainement sont D2, comme ceux de la reference" — la taxonomie
  survit, l'illustration change de camp.
- **P-D2i-4** : u10_s0 (unit, theta 1.0) survit au FORWARD (etat borne par
  construction) ; s'il meurt, c'est backward-ne — et cela renforce "la borne
  ne protege pas du backward : deux territoires".
- **P-RANG-1** : sur ctrl_s0, le `pr` de l'etat produit reste >> 1 pendant
  tout l'entrainement (les artefacts finaux mesurent 30-80).
- **P-RANG-2** : a theta 0.3/1.0, chez au moins une graine, `pr` chute vers
  ~1-5 AVANT toute non-finitude — le collapse silencieux que le log
  historique regime_theta10 appelait "locked" (cos_to_seg0 = +-1). Si
  observe : le trou du classifieur (reference pr relative au gap 0, aveugle
  a un artefact DEJA collapse) devient un point du papier.

## Regles d'ecriture

Chaque phrase du brouillon sur nos NaN d'entrainement sera conditionnee au
verdict ; si la distribution est mixte (D1 et D2 selon la graine), elle est
rapportee telle quelle, sans lissage.

---

# Echelle 2 (25/08 soir) — relance meticuleuse apres les morts au pas 0

Demande de Josep : relancer en verifiant que TOUT est identique a
l'entrainement famille. Enregistre AVANT lancement.

## Verdicts de l'echelle 1 (pour memoire)

- ctrl_s0 (rescale, theta 0.03) : 1500 pas, 0 incident. P-D2i-1 CONFIRMEE.
- t03/t10 (rescale, theta 0.3 et 1.0, 2 graines chacun) : MORT AU PAS 0,
  4/4 — p_bad=0, perte ET w1 non finis pendant le premier forward, tous
  les grads nan (backward d'une perte nan, trivial). Signature D1 pure,
  a l'INIT, avant tout entrainement. P-D2i-2 confirmee (aucune mort
  backward-nee a pas <= 2 : la mort est FORWARD-nee au pas 0 — cas non
  prevu par la lettre de P-D2i-3, qui imaginait des morts en cours
  d'entrainement ; la branche D1 de P-D2i-3 est confirmee sur la
  localisation, pas sur le moment).
- u10_s0 (unit, theta 1.0) : vivant et apprenant au pas 650+ a la dose qui
  tue rescale au pas 0. P-D2i-4 confirmee cote forward.
- L'ETONNEMENT : les NaN historiques (~13% des graines, nd_*.log morts
  ~pas 1700 ; regime_theta10.log progressait a theta 1.0 sur 29 segments)
  ne ressemblent PAS a ces morts d'init. Tension a resoudre : autre banc
  (nd_* = dim32/np8 synthetique ; regime_* = variante d'architecture ?)
  ou artefact de harnais ?

## Audit d'identite du harnais (fait AVANT l'echelle 2, consigne ici)

diag_d2_train.py est le code d'entrainement lui-meme : prepare_data ->
build_cfg("sane_ns") -> WorldModel(depth=2, seg=8, dim=space k=128) ->
Adam(1e-3) -> double ([V;V]) -> tbptt_step(window=None) -> opt.step, meme
ordre de consommation RNG (manual_seed(seed) apres chargement des données,
randperm par pas identique a train_loop), model.train() actif, batch 8,
T=64 -> [V;V]=128 latents, 32 appels memory.update par pas, AUCUN
ecretage nulle part (fidele au train_loop du repo). Le dump cfg du header
est identique champ a champ au cfg de frozen_muon/mu_rescale_s0.pt.
L'espion est purement observationnel (normes et svd detaches, hors
graphe). Seuil residuel d'incertitude : l'arbitre du volet A le ferme.

## Volet A — arbitre externe (le harnais hors-jeu)

python -m titans.worldmodel.echo, NON MODIFIE, commande famille exacte,
200 pas : (A1) theta-scale 0.3, seed 0 ; (A2) theta-scale 0.03, seed 0.

- P-E2-1 : A1 rend un etat/des scores non finis (la mort au pas 0 est
  reelle, pas un artefact du harnais) ; A2 est sain. Branche alternative,
  pre-enregistree : si A1 s'entraine normalement, le harnais est INCULPE,
  toute l'echelle 1 est annulee, et l'ecart harnais/echo devient l'objet
  de l'enquete.

## Volet B — echelle fine de theta (harnais forensique)

theta_scale dans {0.05, 0.08, 0.12, 0.2} x graines {0, 1}, 1500 pas,
config par ailleurs bit-identique a ctrl_s0. Nouveau champ passif
first_bad_upd : index du premier memory.update non fini dans le pas
(localisation intra-sequence de la mort).

- P-E2-2 : le seuil theta* est dans [0.05, 0.15] (borne par ctrl stable a
  0.03 et mort instantanee a 0.3 ; la condition theta*kappa*lambda ~ 2
  avec les proxys d'init porte un facteur ~2 d'incertitude, declare).
- P-E2-3 : transition RASOIR, comme I5 : aucun barreau a divergence
  lente ; sous le seuil, w1_max plateau attracteur ; au-dessus, mort au
  pas 0. first_bad_upd DECROIT avec theta (dose plus forte = mort plus
  tot dans la sequence), miroir des morts en <= 8 chunks de I5.
- Si un barreau intermediaire donne des morts EN COURS d'entrainement
  (pas > 0), c'est le regime des NaN historiques : classification D1/D2
  au compteur, precurseurs w1_max a l'appui — le resultat le plus
  precieux possible pour le papier.

## Volet C — controles longs au budget famille exact

theta_scale 0.03, graines 1 et 2, 4000 pas (le budget famille exact),
harnais forensique.

- P-E2-4 : 0 mort (la famille etait 25/25 propre a ce budget), w1_max en
  plateau, pr >> 1 tout du long (P-RANG-1 etendue). Toute mort observee
  serait une piece rare : classification immediate au compteur.

## Regle

Si P-E2-1 tombe (branche alternative), TOUT verdict de l'echelle 1 est
suspendu le temps de l'enquete d'ecart. Sinon, la lecture au papier est :
nos morts a theta pousse sont des morts D1 d'INIT (le seuil est croise
des l'initialisation, coherent avec la transition rasoir de I5), et le
regime historique des NaN tardifs reste a capturer sur son banc d'origine.

## VERDICT Volet A (25/08 ~16h00, lu des le rendu)

P-E2-1 CONFIRMEE : l'arbitre externe (echo.py NON MODIFIE, commande famille
exacte) rend a theta 0.3 des scores TOUS nan (memory_gain nan, PR nan,
repeat/cross nan) — la meme mort que le harnais — et reste sain a theta
0.03 (scores finis, gain ~0 a 200 pas, attendu). Le harnais est INNOCENTE ;
la mort au pas 0 a theta 0.3 est reelle, portee par l'entraineur d'origine.
L'echelle 1 tient. Volets B (echelle fine, en cours depuis fine_t005_s0)
et C (controles longs) : verdicts a la fin du runner.

## Verdicts intermediaires echelle 2 (25/08 ~17h, en cours)

- Volet B : fine_t005_s0 (theta 0.05) SURVIT 1500 pas — theta* > 0.05,
  borne basse de P-E2-2 tenue. Barreaux suivants en cours.
- P-RANG-1 CONFIRMEE sur ctrl_s0 : pr in [61.7, 69.0] sur 1500 pas
  (mediane 67.7, legere hausse) — aucun collapse de rang a la dose saine.
- P-RANG-2 : non testable sur les morts au pas 0 (aucun temps pour
  collapser) ; a lire sur les barreaux fins proches de theta*.

## VERDICTS FINAUX echelle 2 (lus le 26/08 matin ; runner fini 25/08 21h28)

- **P-E2-2 CONFIRMEE, seuil epingle : theta* in (0.05, 0.08]** — dans la
  bande pre-enregistree [0.05, 0.15], cote bas. 0.05 : survit 1500 pas
  (2/2) ; 0.08 : mort (2/2, pas 7 et 2).
- **P-E2-3 CONFIRMEE dans les deux sens.** (i) Transition rasoir : aucun
  barreau a divergence lente ; x1.6 sur theta_scale fait passer de "survit
  1500 pas" a "mort en <= 7 pas". (ii) Latence dosee : pas de mort 7/2
  (theta 0.08) -> 0 (0.12+) ; first_bad_upd (index intra-sequence du
  premier update non fini) : 30/15 (0.08) -> 15 (0.12) -> 10 (0.2) — la
  mort remonte dans la sequence avec la dose, miroir des <= 8 chunks de I5.
- **P-E2-4 CONFIRMEE** : long003 s1/s2 (theta 0.03, 4000 pas = budget
  famille exact) : 0 mort, 2/2.
- **P-D2i-4 CONFIRMEE en entier** : u10 (unit, theta 1.0) survit les 1500
  pas — la borne tient l'entrainement a >= 12x la dose du mur rescale.
- **CLASSIFICATION FINALE : toutes nos morts d'entrainement sont D1 —
  nees au FORWARD dans la boucle interne** (p_bad=0 au pas fatal, w1 non
  fini PENDANT le forward, gradients empoisonnes en consequence, params
  au pas suivant). ZERO mort de signature D2 (etat fini + grads
  empoisonnes) sur toute l'echelle. La phrase du CLAIM I5 ("coherent avec
  nos morts par graine cote world model") SURVIT et monte en preuve
  directe.
- **RESULTAT NEUF, a ecrire : la loss est AVEUGLE a D1.** fine_t008_s0 :
  w1_max fait 34 -> 371 -> 5.4e5 -> 123 -> 214 -> 1.8e7 -> 5.1e15 -> NaN
  sur les pas 0-7 pendant que la loss reste 0.97-1.06 A CHAQUE PAS, y
  compris le pas fatal. Cause : la lecture est l2-normalisee
  (normalize_retrieval), donc la perte externe ne voit pas ||W||. Les
  precurseurs D1 existent et sont enormes (spikes intermittents,
  dependants du batch) mais INVISIBLES dans la courbe de loss — il faut
  la sonde d'etat. Argument direct pour l'instrumentation du papier.
- **P-RANG-2 : pas de collapse a pr~1 avant mort, mais degradation
  monotone du rang avec la dose** : pr du premier etat produit 62-69
  (theta 0.03) -> 39/32 (0.08) -> 20 (0.12) -> nan (0.2).

## VERDICT chasse Figure 2 (P-F2, consigne aussi dans predictions_famille.md)

- **P-F2a TOMBE** : 0 ligne divergente sur le protocole exact (96 lignes,
  rep et crs), etat surveille. MAIS le diagnostic famscore est repris :
  dans retention_fam_mu_rescale.json, aux gaps >= 256 latents, les scores
  (l1_repeat/l1_cross/gain/mem_effect) sont nan avec **health_finite=True**
  et un pr qui S'EFFONDRE avec la distance (16.9 a gap 0 -> 3.2 a 2048).
  Le "mur" du rejeu n'est PAS une explosion d'etat : c'est une
  non-finitude du READOUT sur etat fini et rang degenere — la chasse
  guettait l'etat, elle ne pouvait rien trouver. Prochaine sonde : meme
  protocole, guetter la PREMIERE non-finitude dans le chemin de
  prediction (retrieval normalise ? attention ?) au gap 256, par ligne.
