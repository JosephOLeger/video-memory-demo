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
