# Predictions — echelle I (inference pure sur la pile de reference)

Enregistre le 2026-08-24, AVANT lancement. Objet: combler la cellule manquante
de la matrice de preuves D1/D2 — D1 (boucle de courbure, risque de deploiement)
n'a jamais ete demontre SUR LEUR PILE. L'echelle R (entrainement) est contaminee
par D2 par construction; l'echelle I retire le backward entierement: tout mur
observe ici ne peut etre que D1 (ou un mecanisme forward inconnu — a discuter
si observe la ou nous ne le predisons pas).

## Dispositif

`probe_iladder.py` sur le box, `~/repro/titans-pytorch`, venv `repro`.
NeuralMemory de lucidrains pilote directement (sans le trunk MAC), config
miroir de l'echelle R: dim 384, dim_head 64, heads 4, MemoryMLP(64, depth 4),
chunk 4, ttt batch 4, max_lr 1e-1, qk_rmsnorm ON, momentum ordre 1,
attn_pool_chunks ON, per_parameter_lr_modulation ON. Poids et gates a
l'initialisation (pas d'entrainement externe — les gates entraines sont de
toute facon erases, cf. eta_init/alpha erasure + replication Planquart).
Flux: enwik8 reel (leur fichier de donnees), embedding gele seed-fixe,
262144 octets en fenetres de 512, ETAT PORTE EN CONTINU (aucun reset).
`torch.inference_mode()` — aucun backward, D2 physiquement impossible.
2 graines par barreau (offsets de flux differents + init differente).

Simplifications documentees (a l'identique sur tous les barreaux, donc jamais
un confondant inter-barreaux): qkv_receives_diff_views OFF, weight_residual
OFF (features du MAC multi-couches), pre/qk rmsnorm ON (amortisseurs de
reference CONSERVES — tout mur trouve l'est malgre eux).

Instrumentation par fenetre: ||W_l||_F par couche depuis l'etat retourne,
rms des caches h_l par une passe-sonde sur cles fixes, theta effectif
(adaptive_lr moyen/max), rms du retrieved. Arret sur non-fini ou
||W|| > 1e6 (mur atteint).

## Barreaux

| barreau     | spec_norm | ResidualNorm | biais alpha      | max_lr |
|-------------|-----------|--------------|------------------|--------|
| I0_defaut   | ON        | ON           | 0 (alpha=0.5)    | 0.1    |
| I1_alpha    | ON        | ON           | -8 (alpha~3.4e-4)| 0.1    |
| I2_brut_rn  | OFF       | ON           | -8               | 0.1    |
| I3_fidele   | OFF       | OFF          | -8               | 0.1    |
| I3_dose     | OFF       | OFF          | -8               | 0.01   |
| I4_amnesie  | OFF       | OFF          | 0 (alpha=0.5)    | 0.1    |

I3 = la memoire fidele au papier (ni spec_norm ni ResidualNorm, tous deux
absents de Titans/ATLAS), configuree pour la portee: alpha~3.4e-4/chunk =
demie-vie ~2000 chunks (~8k octets), exactement ce que notre loi
alpha* ~ 1/(2d) impose a qui veut une memoire qui porte. C'est LE barreau.

## Predictions

- **P-I0** (defauts): plateau amnesique. ||W|| se stabilise en <100 chunks
  (oubli 0.5/chunk => serie geometrique), aucune croissance ensuite, aucun
  mur sur 65k chunks. La memoire n'accumule rien: portee ~1 chunk.
- **P-I1** (alpha portee, depots bornes): accumulation VERS UN PLATEAU plus
  haut que I0 (regime somme bornee, famille softclip/unit), pas de mur.
- **P-I2** (depots bruts, ResidualNorm garde): croissance de ||W|| visible,
  issue INCERTAINE declaree — soit derive lente sans mur a 65k chunks, soit
  mur tardif. Les deux issues sont informatives (isole le role du
  ResidualNorm, l'amortisseur B de l'audit).
- **P-I3** (fidele + portee): **mur D1** — croissance super-lineaire de
  ||W|| et des h_l visible dans les 5k premiers chunks, non-fini ou
  ||W||>1e6 avant 65k chunks, theta_max*lambda_proxy croisant O(1) AVANT le
  mur. Position du mur stable entre graines a un facteur ~2 (D1 est une
  derive deterministe, pas un accident stochastique).
- **P-I3_dose** (idem, theta/10): le mur recule d'un facteur >=5 en position
  ou disparait du budget — dose-reponse dans le sens de theta*kappa*lambda>2.
- **P-I4** (depots bruts mais alpha=0.5): PAS de mur — l'amnesie protege
  meme sans aucune borne sur les depots. C'est l'identification du VRAI
  amortisseur de deploiement: le gate d'oubli au repos.

## Regles de decision (pre-enregistrees)

- P-I3 ET P-I3_dose confirmes => D1 est demontre dans la pile de reference,
  avec dose-reponse, en un regime ou D2 est impossible par construction.
  Le passage a risque du papier passe de la defense a l'attaque: le papier
  PREDIT la survie de R0 (regime amnesique) et PRODUIT la divergence de
  deploiement la ou sa theorie la place.
- P-I3 confirme mais P-I3_dose infirme (le mur ne recule pas) => le mur
  n'est pas gouverne par theta comme la theorie le dit; consigner comme
  contradiction, NE Pas citer l'echelle I en soutien du theoreme.
- P-I3 infirme (pas de mur nulle part, y compris I3 prolonge x4) => la
  conclusion honnete: la memoire de reference est deploiement-stable a
  cette echelle meme configuree pour la portee; le brouillon RESTREINT
  explicitement D1 a nos piles (world model, graft) et la section devient:
  l'amortisseur identifie est l'amnesie (si I0/I4 plats et I1 plafonne),
  qk/pre-rmsnorm restant les suspects a discuter. Pas de sauvetage
  rhetorique.
- Si un mur apparait en I0 ou I4 (non predit): instrument suspect d'abord
  (verifier wnorms/probe), sinon mecanisme forward inconnu — consigner,
  ne rien publier avant comprehension.

## Replication de graines de l'echelle R (meme file)

Les scripts de reference n'ont pas de graine fixe: tout re-run est un tirage
independant. ctrl512 (en cours) = 2e realisation de R1 (prediction: NaN au
pas <=2, gradcheck non-fini au pas 1). R2b et R3b (1000 pas chacun,
gradcheck insere, queues derriere l'echelle I): prediction — survie 2/2,
0 gradient non-fini sur toute la course. Si R2b ou R3b meurt: la survie de
R2/R3 etait un accident de graine, le brouillon retire le CLAIM
survie-comme-prediction et le remplace par la distribution observee.

## OUTCOMES PARTIELS (24/08 ~17h, 3 barreaux/12)

- **I0_defaut s0+s1 : P-I0 CONFIRME** (et durci) : etat a zero exact sur tout
  le flux (mort au chunk ~96 mesuree en calibration), ret_rms constant 0.306
  (la lecture vit sur le chemin residuel). Pas amnesique : mort.
- **I1_alpha s0 : P-I1 PARTIELLEMENT INFIRME.** Pas de plateau d'accumulation :
  l'etat meurt au rythme exact configure ((1-3.4e-4)^n a quelques % pres —
  biais -8 verifie actif), depots invisibles <=1e-5 meme a poids sains.
  Lecture : sous spec-norm les depots n'ont la magnitude pour soutenir un etat
  a AUCUN alpha ; baisser alpha ralentit l'horloge de la mort. Les barreaux
  decisifs restent I2/I3 (depots bruts), comme concu.

## ANNULATION v1 + RELANCE v2 (24/08 soir)

Les "OUTCOMES PARTIELS" ci-dessus sont ANNULES : artefact d'instrument.
La sonde v1 tournait sous torch.inference_mode(), qui zerote SILENCIEUSEMENT
les gradients torch.func du chemin d'ecriture (verifie a config/graine
identiques : NS input 0.0 sous inference_mode, 0.34 sous no_grad et en mode
normal ; symptome retrospectif : trajectoires bit-identiques I1/I2/I3 = pure
decroissance de l'init). Ce qui survit de la v1 : alpha repos ~0.5/chunk
(non sature), biais -8 actif, qk-norm = gamma+1 (gain 1 a l'init, pas un
silencieux), gradient nul en W=0 exact (fait mathematique, MLP sans biais).
Sous no_grad, config I1 : l'etat CROIT 22.7 -> 242 en une fenetre (128
chunks) — le chemin d'ecriture est vivant.

**v2 relancee** : probe_iladder2.py (no_grad ; toujours aucun backward
externe -> D2 toujours impossible par construction), results_iladder2/,
memes 6 barreaux x 2 graines, MEMES predictions et regles de decision
qu'en tete de ce fichier. P-I1 (plateau borne) redevient plausible ;
P-I3 (mur D1) est a ce jour NON TESTE.

## AMENDEMENT 25/08 matin — v2 lue, extension + echelle de gain I5 (AVANT lancement)

v2 rendue (12/12, aucun mur): I0 plateau amnesique ~17 (P-I0 v-plateau OK);
I1 plateau borne ~790 avec theta*lambda ~ 144 >> 2 — le regime borne ignore le
seuil du gradient brut = le resultat famille reproduit sur leur pile (P-I1 OK);
I2/I3 contraction vers 7-9, theta*kappa*lambda ~ 0.2 << 2 — la stabilite
observee est PREDITE par la condition (P-I2 issue "ni derive ni mur");
I4 mort reelle (zero exact, gradients vivants — preuve interne: I0 meme
harnais plateau 17): la memoire fidele au papier meurt au repos, c'est la
spec-norm qui la maintient en vie. P-I3 (mur) infirme A CETTE ECHELLE — la
regle exige l'extension x4 avant verdict.

**I3ext (lance maintenant)**: I3_fidele 2 graines a 1,048,576 octets (x4).
PREDICTION: le plateau persiste, pas de mur.

**I5 — echelle de gain (lance maintenant)**: config I3 (fidele, alpha portee,
max_lr 0.1) avec init des poids memoire x g, g dans {2, 4, 8}, 2 graines.
Mecanisme: h0 ~ g (1re couche), lambda ~ g^2, theta*kappa*lambda(g=1) ~ 0.2
=> franchissement de 2 attendu vers g* ~ 3.
PREDICTIONS:
- P-I5a: l'instabilite est monotone en g (croissance de ||W||/h0 d'autant
  plus rapide que g est grand).
- P-I5b: il existe un g* fini dans [2, 8]: mur (non-fini ou ||W||>1e6) a
  g >= g*, plateau/contraction en dessous; le mur est precede par la
  croissance de h0 (signature D1).
- Si aucun mur a g=8: P-I5 infirme => restriction honnete maintenue (la
  contraction GELU de leur init bat la boucle a toute echelle atteignable);
  identifier l'amortisseur via la trajectoire h.

## VERDICT I3ext + I5 (25/08 ~12h30) + affinage (AVANT lancement)

- I3ext: plateau 8.6-9.0 sur 262k chunks x2 graines — la regle x4 est purgee,
  P-I3 infirme A g=1 proprement.
- I5_g2: 53 -> erosion -> MEME plateau ~8.5 (attracteur sous le seuil), 2/2.
- **I5_g4 et I5_g8: MUR — non-fini des la premiere fenetre, 4/4 realisations.**
  P-I5a (monotonie) CONFIRME; P-I5b (g* fini) CONFIRME avec g* dans (2, 4].
  Quantitatif: theta*kappa*lambda(g) ~ 0.2 g^2 => seuil 2 franchi a
  g* ~ 3.2 — DANS le bracket mesure. D1 est demontre sur leur pile en
  inference pure.

**Affinage (lance maintenant):**
- I5_g3 (2 graines): theta*kappa*lambda(3) ~ 1.8, a un facteur ~2 du seuil —
  prediction VOLONTAIREMENT non tranchee (les estimations kappa/lambda ont
  des incertitudes de facteur 2): sa valeur est de RESSERRER g*, pas de
  tester. Les deux issues sont consignables telles quelles.
- I5_g4fin (1 graine, fenetre 32 tokens, 16k tokens): prediction — la
  signature D1 est visible AVANT le non-fini: h0 croit de facon monotone
  sur les chunks precedant le mur (boucle somme, pas bascule instantanee).
  Si le non-fini arrive sans croissance visible meme a ce grain: consigner
  comme bascule rapide, ne pas presenter la trajectoire comme signature.

## AFFINAGE 2 (25/08 ~12h45, AVANT lancement) — la zone proche seuil

Verdicts precedents: g=3 MUR 2/2 des la fenetre 0 => g* dans (2, 3].
g=4 a grain 32: non-fini dans les 8 premiers chunks => bascule rapide,
consignee comme telle (pas de trajectoire-signature a ce gain).

Lance: g dans {2.25, 2.5, 2.75}, 1 graine, fenetre 32, 65536 octets.
PREDICTION: g* etant dans (2, 3], au moins un de ces barreaux est juste
au-dessus du seuil et diverge LENTEMENT — croissance monotone de h0/||W||
visible sur des dizaines a centaines de chunks avant le non-fini (la
signature somme de D1), la ou g=4 bascule en <8 chunks. Les barreaux sous
g* montrent l'erosion vers le plateau ~8.5. Si TOUS basculent vite ou
tous plateauent (g* a bord raide): consigner la raideur du seuil comme
resultat — la transition est alors quasi discontinue en g, ce qui est
aussi une signature interpretable (la boucle multiplicative par couche
domine la somme).

## VERDICT FINAL I5 (25/08 ~13h)

- g=3: MUR 2/2 des la fenetre 0 => bracket resserre.
- Affinage fin (fenetre 32): g=2.25, 2.5, 2.75 — TOUS mur fenetre 0
  (non-fini en <=8 chunks). Branche alternative pre-enregistree appliquee:
  le seuil est RASOIR, aucun barreau a divergence lente n'existe —
  signature de la boucle multiplicative par couche (profondeur 4,
  gain de boucle ~ g^4).
- Controle croise d'instrument: g=2.25 a fenetre 512 => mur (idem);
  g=2.0 a fenetre 32 => survit 2048 fenetres (idem). La fenetre n'est
  pas un confondant.
- **g* dans (2.0, 2.25]**. theta*kappa*lambda(g) ~ 0.2 g^2 (facteur ~2
  d'incertitude declare) => franchissement predit dans la bande 2-3:
  mesure DANS la bande. D1 est demontre sur la pile de reference, en
  inference pure, dose par l'echelle d'etat.
- g=4 grain fin: bascule en <8 chunks — consignee comme bascule rapide,
  pas de trajectoire-signature presentable a ce gain (regle appliquee).
